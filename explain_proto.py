"""Prototype-level explanation of a LogiX-GIN-with-prototypes run.

Division of labour: **all rule extraction lives in `latent_logic.py`**, which is the
port of `nbs/LayerWiseRules.ipynb`. This file answers the questions that are about
the *prototypes* rather than about a single latent component:

  1. IDENTITY    which population of training nodes (or graphs) each prototype
                 stands for, described as chemistry: atom types, 1-hop
                 environments, degree, and the class balance of the molecules the
                 matching nodes come from.
  2. COMPONENTS  which latent components a prototype actually keys on, and - via
                 latent_logic.explain_component - the logical formula of each one.
  3. REDUNDANCY  whether the K prototypes are K distinct binary patterns.
  4. HEAD        which prototypes each class reads, via latent_logic.explain_head.
  5. EVIDENCE    whether each readout column separates the classes on held-out data.

These runs are trained with `--push_every 0`, so a prototype is not a real training
example: it is a free binary vector. Section 1 therefore characterises each one by
the population it fires on, and reports the single nearest node only as an
illustration.

Two splits are used, deliberately:
  --split        (default train) for the prototype chemistry, since the prototypes
                 were fit on the training set.
  --rule_split   (default val) for rule extraction and its support figures, which
                 is the notebook's own choice and makes support a held-out number.

Usage:
    python explain_proto.py --run_path results_proto/<dataset>/<args>/<baseline>/<seed>
    python explain_proto.py --run_path ... --decode        # + component formulas
"""
import argparse
import json
import os
import pickle
from collections import Counter

import torch
from torch_geometric.loader import DataLoader

from latent_logic import (ATOMS, CLASSES, collect_activations, explain_component,
                          explain_head)
from utils.utils import get_dataset


def atom(x_row):
    i = int(x_row.argmax())
    return ATOMS[i] if i < len(ATOMS) else f'?{i}'


def describe_node(data, n):
    """Atom plus its 1-hop environment, e.g. 'N bonded to [C, O, O]'."""
    nbrs = data.edge_index[1][data.edge_index[0] == n]
    env = sorted(atom(data.x[int(j)]) for j in nbrs)
    return atom(data.x[n]), env


def env_string(sym, env):
    return f'{sym}[{",".join(env)}]'


def readout_columns(model, level):
    """Column layout of s, matching GINTELLProtoBase.readout().

    _pool concatenates one full block of K per pooling op, so the ordering is
    op-major: [op0 x K, op1 x K, ...].
    """
    cols = []
    if level == 'node':
        for op in model.pool_ops:
            cols += [('node', k, op) for k in range(model.num_prototypes)]
    elif level == 'graph':
        cols = [('graph', k, 'pooled') for k in range(model.num_prototypes)]
    elif level == 'both':
        for op in model.node_pool_ops:
            cols += [('node', k, op) for k in range(model.num_prototypes)]
        cols += [('graph', k, 'pooled') for k in range(model.num_graph_prototypes)]
    return cols


def proto_dim_map(model, level, li):
    """Prototype dim -> (pool_op | None, conv layer, hidden unit).

    The prototype input is torch.hstack(xs), i.e. layer-major blocks of hidden_dim,
    optionally preceded by a pooling-op block for the graph-level layers.
    """
    L, h = model.num_layers, model.hidden_dim
    node_level = (level == 'node') or (level == 'both' and li == 0)
    if node_level:
        return [(None, d // h, d % h) for d in range(L * h)]
    ops = model.pool_ops if level == 'graph' else model.graph_pool_ops
    return [(op, (d % (L * h)) // h, d % h) for op in ops for d in range(L * h)]


# ---------------------------------------------------------------------------
# what the prototype analysis needs, on top of latent_logic's activations
# ---------------------------------------------------------------------------

def provenance(dataset):
    """Per-node bookkeeping that needs no model pass.

    Returns node_graph / node_local (indices back into `dataset`, in loader order),
    graph_y, A (each node's atom index) and NB (closed-neighbourhood atom counts).
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    node_graph, node_local, graph_y, A, NB = [], [], [], [], []
    g_offset = 0
    for data in loader:
        xf = data.x.float()
        b = data.batch
        node_graph.append(b + g_offset)
        counts = torch.bincount(b, minlength=data.num_graphs)
        node_local.append(torch.cat([torch.arange(int(c)) for c in counts]))
        graph_y.append(data.y.reshape(-1))
        A.append(xf.argmax(1))
        nb = torch.zeros_like(xf)
        nb.index_add_(0, data.edge_index[1], xf[data.edge_index[0]])
        NB.append(nb + xf)
        g_offset += data.num_graphs
    return (torch.cat(node_graph), torch.cat(node_local), torch.cat(graph_y),
            torch.cat(A), torch.cat(NB))


def derive(model, acts):
    """Concepts and prototype inputs, read off latent_logic's activations.

    acts[l]['y'] is conv layer l's output, so hstack over layers is exactly the
    concept matrix the prototype layers are compared against - no second model pass.
    """
    xs = [acts[l]['y'] for l in range(len(model.convs))]
    batch = acts[0]['batch']
    H = torch.hstack(xs)
    feats = [f.cpu() for f in model.prototype_inputs(xs, batch)]
    return xs, batch, H, feats


# ---------------------------------------------------------------------------
# 1. IDENTITY
# ---------------------------------------------------------------------------

def identity(model, level, feats, node_graph, node_local, graph_y, dataset, top, fire_thr):
    """What population of nodes (or graphs) each prototype stands for."""
    out = {}
    for li, layer in enumerate(model.proto_layers):
        lname = 'node' if (level == 'node' or (level == 'both' and li == 0)) else 'graph'
        x = feats[li]
        s = layer.similarity(x.to(layer.proto_logits.device)).cpu()   # [rows x K]
        print(f'\n[{lname} prototypes]  {x.shape[0]} rows x {x.shape[1]} dims '
              f'(continuous in [0,1])')
        out[lname] = []
        for k in range(s.shape[1]):
            sk = s[:, k]
            order = sk.argsort(descending=True)
            fired = int((sk >= fire_thr).sum())
            q99 = float(torch.quantile(sk, torch.tensor([0.99]))[0])
            entry = {'k': k, 'max_sim': float(sk.max()), 'mean_sim': float(sk.mean()),
                     'p99_sim': q99, 'n_fired': fired, 'frac_fired': fired / len(sk)}
            if lname == 'graph':
                gi = int(order[0])
                d = dataset[gi]
                entry['nearest'] = f'graph #{gi} (y={CLASSES[int(d.y)]}, {d.num_nodes} atoms)'
                lab = Counter(CLASSES[int(graph_y[int(g)])] for g in order[:top])
                entry['class_mix'] = dict(lab)
                print(f'  p{k:<3d} sim max/p99/mean={entry["max_sim"]:.3f}/{q99:.3f}/'
                      f'{entry["mean_sim"]:.3f} fired={entry["frac_fired"]:6.2%}'
                      f'  top-{top} classes={dict(lab)}  nearest={entry["nearest"]}')
            else:
                envs, syms, degs, labs = Counter(), Counter(), [], Counter()
                for r in order[:top].tolist():
                    gi, ni = int(node_graph[r]), int(node_local[r])
                    d = dataset[gi]
                    sym, env = describe_node(d, ni)
                    syms[sym] += 1
                    envs[env_string(sym, env)] += 1
                    degs.append(len(env))
                    labs[CLASSES[int(d.y)]] += 1
                entry['atoms'] = dict(syms.most_common())
                entry['envs'] = dict(envs.most_common(4))
                entry['mean_degree'] = sum(degs) / max(len(degs), 1)
                entry['class_mix'] = dict(labs)
                r0 = int(order[0])
                sym0, env0 = describe_node(dataset[int(node_graph[r0])], int(node_local[r0]))
                entry['nearest'] = env_string(sym0, env0)
                atom_s = ' '.join(f'{a}:{c/top:.0%}' for a, c in syms.most_common(3))
                env_s = ' '.join(f'{e}({c})' for e, c in envs.most_common(2))
                print(f'  p{k:<3d} sim max/p99/mean={entry["max_sim"]:.3f}/{q99:.3f}/'
                      f'{entry["mean_sim"]:.3f} fired={entry["frac_fired"]:6.2%} '
                      f'| atoms {atom_s:<24s} | deg {entry["mean_degree"]:.1f} '
                      f'| {dict(labs)} | top env {env_s}')
            out[lname].append(entry)
    return out


# ---------------------------------------------------------------------------
# 2. COMPONENTS - which latent components a prototype keys on, and their formulas
# ---------------------------------------------------------------------------

def components(model, level, feats, H, A, NB, rule_acts, top_dims,
               max_rule_len, max_rules, min_support):
    """Per prototype, its most characteristic components and each one's formula.

    A prototype is a full binary vector, so about half its dims are 1 and most carry
    no information. A component is *characteristic* when the nodes matching the
    prototype hold it far more than the population does, so active dims are ranked
    by lift = agreement on the prototype's top nodes minus the population rate.
    (Ranking by rarity instead selects dims that never fire, which describe nothing.)

    The formula of each selected component comes from latent_logic.explain_component,
    so it is the notebook's extraction, with its support and precision.
    """
    rate = H.mean(0)
    # The student's LogicalLayer output is never hard-thresholded, so concepts are
    # CONTINUOUS in [0,1]: a dim can have rate > 0 yet never cross 0.5.
    above = (H > 0.5).float().mean(0)
    const = int(((rate <= 1e-6) | (rate >= 1 - 1e-6)).sum())
    never_half, always_half = int((above <= 0).sum()), int((above >= 1).sum())
    print(f'    concept dims: {len(rate)} total | {const} exactly constant | '
          f'{never_half} never exceed 0.5 | {always_half} always exceed 0.5 '
          f'-> {(never_half + always_half) / len(rate):.0%} effectively uninformative')

    cache, out = {}, {}
    for li, layer in enumerate(model.proto_layers):
        lname = 'node' if ((level == 'node') or (level == 'both' and li == 0)) else 'graph'
        dmap = proto_dim_map(model, level, li)
        p = (layer.proto_logits > 0).float().cpu()
        sim = layer.similarity(feats[li].to(layer.proto_logits.device)).cpu()

        wanted = {}
        for k in range(p.shape[0]):
            act = (p[k] > 0.5).nonzero(as_tuple=True)[0]
            if lname == 'node':
                tn = sim[:, k].argsort(descending=True)[:200]
                lift = H[tn][:, act].mean(0) - rate[act]
                wanted[k] = act[lift.argsort(descending=True)][:top_dims].tolist()
            else:               # graph dims index pooled features, not H
                wanted[k] = act[:top_dims].tolist()

        print(f'\n[{lname} prototypes]  formulas from latent_logic '
              f'(counts over the closed neighbourhood; support on the rule split)')
        out[lname] = []
        for k in range(p.shape[0]):
            top_nodes = sim[:, k].argsort(descending=True)[:200]
            print(f'\n  p{k}:')
            entries = []
            for d in wanted[k]:
                op, lay, u = dmap[d]
                if (lay, u) not in cache:
                    e = explain_component(model, lay, u, rule_acts,
                                          max_rule_len=max_rule_len, max_rules=max_rules,
                                          min_support=min_support, recursive=False)
                    cache[(lay, u)] = e['components'][(lay, u)]['rules']
                rules = cache[(lay, u)]
                tag = f'L{lay}u{u}' + (f'[{op}]' if op else '')

                entry = {'dim': int(d), 'layer': lay, 'unit': u, 'pool': op,
                         'rate': float(rate[d]) if lname == 'node' else None,
                         'rules': [{'literals': [nm for _, _, nm in r['literals']],
                                    'support': r['support'], 'precision': r['precision'],
                                    'always_true': r['always_true']} for r in rules]}
                if lname == 'node':
                    m = H[:, d] > 0.5
                    if int(m.sum()) > 0:
                        ac = torch.bincount(A[m], minlength=len(ATOMS)).float()
                        ac = ac / ac.sum()
                        atop = ' '.join(f'{ATOMS[i]}:{ac[i]:.0%}'
                                        for i in ac.argsort(descending=True)[:2] if ac[i] > 0.05)
                        nbm = NB[m].mean(0)
                        ntop = ' '.join(f'{ATOMS[i]}:{nbm[i]:.1f}'
                                       for i in nbm.argsort(descending=True)[:3] if nbm[i] > 0.15)
                        cov = float((H[top_nodes, d] > 0.5).float().mean())
                        entry.update({'atoms': atop, 'nbrs': ntop, 'cov_top200': cov})
                        print(f'    {tag:<8s} fires {rate[d]:5.1%} | {atop:<14s} '
                              f'| nbrs {ntop:<22s} | in top-200 {cov:4.0%}')
                    else:
                        print(f'    {tag:<8s} never exceeds 0.5')
                else:
                    print(f'    {tag:<14s}')
                if not rules:
                    print(f'             = <no rule with support >= {min_support} '
                          f'within {max_rule_len} literals>')
                for r in rules[:2]:
                    if r['always_true']:
                        print('             = ALWAYS TRUE on the observed data')
                        continue
                    prec = '-' if r['precision'] is None else f"{r['precision']:.0%}"
                    print(f"             = {' AND '.join(nm for _, _, nm in r['literals'])}")
                    print(f"               [support {r['support']:.2%} / precision {prec}]")
                entries.append(entry)
            out[lname].append({'k': k, 'components': entries})
    return out


# ---------------------------------------------------------------------------
# 3. REDUNDANCY
# ---------------------------------------------------------------------------

def redundancy(model, level):
    """Are the K prototypes actually K distinct binary patterns?"""
    out = {}
    for li, layer in enumerate(model.proto_layers):
        lname = 'node' if (level == 'node' or (level == 'both' and li == 0)) else 'graph'
        p = (layer.proto_logits > 0).float().cpu()
        K, d = p.shape
        agree = (p @ p.t() + (1 - p) @ (1 - p).t()) / d
        uniq = {}
        for k in range(K):
            uniq.setdefault(tuple(p[k].tolist()), []).append(k)
        groups = [g for g in uniq.values() if len(g) > 1]
        off = ~torch.eye(K, dtype=torch.bool)
        out[lname] = {'n_distinct': len(uniq), 'K': K, 'duplicate_groups': groups,
                      'mean_pairwise_agreement': float(agree[off].mean()),
                      'max_pairwise_agreement': float(agree[off].max())}
        print(f'  [{lname}] {len(uniq)}/{K} distinct patterns; '
              f'pairwise agreement mean {agree[off].mean():.3f} '
              f'max {agree[off].max():.3f}')
        if groups:
            print(f'        collapsed groups: {groups}')
    return out


# ---------------------------------------------------------------------------
# 5. EVIDENCE
# ---------------------------------------------------------------------------

@torch.no_grad()
def evidence(model, dataset, device, col_desc, top):
    """Does each readout column separate the classes on held-out data?"""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    S, Y = [], []
    for data in loader:
        data = data.to(device)
        x = data.x.float() if data.x is not None else torch.ones(
            (data.num_nodes, model.num_features), device=device)
        xs = []
        for conv in model.convs:
            x = conv(torch.hstack([x, 1 - x]), data.edge_index)
            xs.append(x)
        S.append(model.readout(xs, data.batch).cpu())
        Y.append(data.y.reshape(-1).cpu())
    S, Y = torch.cat(S), torch.cat(Y)
    m0, m1 = S[Y == 0].mean(0), S[Y == 1].mean(0)
    gap = (m0 - m1) / S.std(0).clamp(min=1e-8)
    print(f'  (mean over {CLASSES[0]} minus mean over {CLASSES[1]}, in std units)')
    for i in gap.abs().argsort(descending=True)[:top].tolist():
        arrow = CLASSES[0] if gap[i] > 0 else CLASSES[1]
        print(f'    {gap[i]:+7.3f} sd -> {arrow:<11s} {col_desc[i]:<20s} '
              f'(mut {m0[i]:.3f} / non {m1[i]:.3f})')
    return gap.tolist(), m0.tolist(), m1.tolist()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_path', required=True, help='a seed directory under results_proto/')
    ap.add_argument('--dataset', default='Mutagenicity')
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                    help='split the prototype chemistry is measured on')
    ap.add_argument('--rule_split', default='val', choices=['train', 'val', 'test'],
                    help="split the rules' support is measured on (the notebook uses val)")
    ap.add_argument('--top', type=int, default=40, help='nodes per prototype to characterise')
    ap.add_argument('--top_cols', type=int, default=10, help='columns to list per class')
    ap.add_argument('--fire_thr', type=float, default=0.65,
                    help='similarity counted as "fires". Hamming agreement concentrates '
                         'near 0.5 (chance), so without --push_every even the best node '
                         'rarely exceeds ~0.7.')
    ap.add_argument('--decode', action='store_true',
                    help='also give the logical formula of each prototype component')
    ap.add_argument('--decode_top', type=int, default=5,
                    help='most characteristic components to decode per prototype')
    ap.add_argument('--max_rule_len', type=int, default=4)
    ap.add_argument('--max_rules', type=int, default=5)
    ap.add_argument('--min_support', type=int, default=2,
                    help='minimum number of examples a rule must jointly hold on '
                         '(see latent_logic; 0 leaves ~45%% of layer-0 rules vacuous)')
    a = ap.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = json.load(open(os.path.join(a.run_path, 'args.json')))
    model = torch.load(os.path.join(a.run_path, a.ckpt), map_location=device,
                       weights_only=False).eval()

    dataset = get_dataset(a.dataset)
    split = pickle.load(open(os.path.join(a.run_path, 'data.pkl'), 'rb'))
    ds_id = dataset[split[f'{a.split}_indices']]
    ds_rule = dataset[split[f'{a.rule_split}_indices']]
    ds_test = dataset[split['test_indices']]

    level = args['proto_level']
    cols = readout_columns(model, level)
    col_desc = [f'{n}.p{k}.{op}' for n, k, op in cols]

    print(f'== {a.run_path}')
    print(f'== level={level}  K={args["num_prototypes"]}  readout_dim={model.readout_dim}')
    print(f'== chemistry on {a.split} ({len(ds_id)} graphs) / '
          f'rules on {a.rule_split} ({len(ds_rule)} graphs)')

    acts = collect_activations(model, DataLoader(ds_id, batch_size=64, shuffle=False), device)
    _, _, H, feats = derive(model, acts)
    node_graph, node_local, graph_y, A, NB = provenance(ds_id)

    print('\n--- 1. what each prototype stands for (top nodes by Hamming agreement) ---')
    ident = identity(model, level, feats, node_graph, node_local, graph_y,
                     ds_id, a.top, a.fire_thr)

    rule_acts = acts if a.rule_split == a.split else None
    comp = None
    if a.decode:
        print('\n--- 2. the components each prototype keys on, and their formulas ---')
        if rule_acts is None:
            rule_acts = collect_activations(
                model, DataLoader(ds_rule, batch_size=64, shuffle=False), device)
        comp = components(model, level, feats, H, A, NB, rule_acts, a.decode_top,
                          a.max_rule_len, a.max_rules, a.min_support)

    print('\n--- 3. are the prototypes distinct? ---')
    red = redundancy(model, level)

    print('\n--- 4. which prototypes each class reads (latent_logic.explain_head) ---')
    if rule_acts is None:
        rule_acts = collect_activations(
            model, DataLoader(ds_rule, batch_size=64, shuffle=False), device)
    head = explain_head(model, level, rule_acts, max_rule_len=a.max_rule_len,
                        max_rules=a.max_rules, min_support=a.min_support)
    for cls, entries in head.items():
        print(f'  {cls}:')
        if not entries:
            print(f'    (no rule with support >= {a.min_support} '
                  f'within {a.max_rule_len} literals)')
        for e in entries:
            if e['always_true']:
                print('    ALWAYS TRUE (default class)')
            else:
                print(f"    {' AND '.join(e['literals'])}   [support {e['support']:.2%}]")

    print('\n--- 5. is each column discriminative? (test split) ---')
    gap, m0, m1 = evidence(model, ds_test, device, col_desc, 2 * a.top_cols)

    out = os.path.join(a.run_path, 'explanations.json')
    with open(out, 'w') as f:
        json.dump({'level': level, 'columns': col_desc, 'split': a.split,
                   'rule_split': a.rule_split, 'identity': ident, 'components': comp,
                   'redundancy': red, 'head': head, 'class_gap_sd': gap,
                   'mean_mutagen': m0, 'mean_nonmutagen': m1}, f, indent=1)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
