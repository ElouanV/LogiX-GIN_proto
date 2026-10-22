"""Interpretability metrics of a prototype LogiX-GIN checkpoint, next to its accuracy and AUC.

Works on prototype runs (results_proto/) and on classic LogiX-GIN runs (results_logic/,
models/model.py GINTELL), whose head reads the mean / max / sum pooled trunk units
directly: there a head literal is one condition on one trunk unit, so expl_bits counts the
explanation's literals, protos_cited is None, and the NO2/NH2 evidence of a literal is the
set of nodes where its unit holds.

Every metric is read off the *symbolic* model: the same weights, with every truth value
binarised the way the rules read it (conv literals phi_in(neighbourhood sum) >= 0.5, conv
units W.lit + b >= 0, head literals phi_in([s, 1-s]) >= 0.5). The prototype step is kept
as the model computes it (Hamming agreement, or exp(-mismatches/T) with --proto_mask),
on those binary node states. So a number here describes the explanation a reader gets,
not the network.

    acc, auc, balanced_acc   the network (utils.evaluation)
    logic_acc                accuracy of the symbolic model
    logic_fidelity           share of graphs where symbolic and network predictions agree
    head_fidelity            same, with the network's own trunk and only the head binarised,
                             so 1 - head_fidelity is the head's share of the infidelity
    head_acc                 accuracy of that reading (network trunk, binarised head): the
                             layer-local reading of the authors' notebook applied to the head
                             (docs/SYMBOLIC_READINGS.md)
    trunk_bit_agreement      share of node bits (all conv layers) on which the symbolic trunk
                             agrees with the network's states >= 0.5
    rule_backed              share of graphs whose predicted class rule actually fires
                             (argmax of the head margins can pick a class whose rule is false)
    expl_literals            head literals in the shortest sufficient explanation of a
                             prediction (true literals by decreasing weight until the
                             threshold is reached: exact minimum cardinality, weights >= 0)
    expl_bits                prototype bits those literals cite: sum over the distinct
                             prototypes cited of their cared bits (d without a mask), i.e.
                             the explanation's length in trunk-unit literals
    protos_cited             distinct prototypes cited over all explanations of the split
    bits_per_proto           cared bits of a cited prototype (median; d without a mask)
    units_cited              distinct trunk units the cited prototypes' cared bits read
    units_cited_L0           share of them in conv layer 0, the only layer whose units
                             decode to short rules (CLAUDE.md, rule-extraction limits)
    literal_purity           class purity of the graphs where a cited head literal holds,
                             weighted by how often it is cited
    gt_precision / gt_chance datasets with a node-level ground truth (configs/datasets.yaml), node
                             prototypes and classic models: share of the best-matching nodes of
                             the cited positive prototype literals that lie in the ground-truth
                             region, on correctly predicted graphs of the scored classes that
                             contain one, and the same share for a uniformly random node (the
                             baseline). Regions: Mutagenicity mutagens, within one hop of an
                             NO2 / NH2 group; BA2Motifs, the planted house / 5-cycle (both
                             classes); BAMultiShapes class 1, its two planted motifs
    mask_temp                temperature stored in the checkpoint (1 = strict AND)
    head_literals_constant   share of head literals with the same value on every input
                             (head_literal_constants: true / false on all of [0, 1], a
                             prototype with no cared bit, or a masked max true for every
                             mismatch count)
    <metric>_evidence        expl_literals, expl_bits, protos_cited, bits_per_proto_median,
                             units_cited, units_cited_L0, literal_purity and gt_precision /
                             gt_chance / gt_n again, on the *evidence*: the explanation in
                             input-dependent literals only (evidence_explanation: the
                             constant-true literals are a bias, taken off the threshold).
                             The plain metrics count constant literals, which hold on every
                             graph and so explain nothing about this one (Mutagenicity QAT
                             NMP fold 3: 5.1 literals, 1.6 of them evidence)

Per-prediction metrics are over rule-backed graphs, as mean and median.

Usage:
    python interp_metrics.py --run_path <seed dir> [<seed dir> ...] --split test --out table.csv
Writes <run_path>/interp_<split>.json next to each checkpoint and prints one row per run.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from explain_proto import proto_dim_map, readout_columns
from min_covers import phi_true_intervals
from models_proto.model_proto import MODELS
from models_proto.model_proto_count import GINTELLProtoCount
from utils.datasets import spec
from utils.evaluation import evaluate, predict
from utils.runs import EVAL_BATCH_SIZE, RunDir, get_device

LEVELS = {cls.__name__: level for level, cls in MODELS.items()} | {GINTELLProtoCount.__name__: 'node'}


def level_of(model):
    """'node' | 'graph' | 'both' for prototype models, 'base' for classic LogiX-GIN (GINTELL)."""
    return LEVELS.get(type(model).__name__, 'base' if not hasattr(model, 'proto_layers') else None)


BASE_POOLS = ('mean', 'max', 'sum')              # models/model.py GINTELL readout order


def aggregate(conv, x, edge_index):
    """GINConv aggregation (1 + eps)·x + sum of the neighbours: the count a LogicalLayer reads."""
    out = torch.zeros_like(x)
    out.index_add_(0, edge_index[1], x[edge_index[0]])
    return out + (1 + conv.eps) * x


@torch.no_grad()
def trunk_states(model, x, edge_index, symbolic=True):
    """Per conv layer node states: binary rule outputs, or the network's own (symbolic=False)."""
    xs = []
    for conv in model.convs:
        if symbolic:
            ll = conv.nn[0]
            lit = (ll.phi_in(aggregate(conv, torch.hstack([x, 1 - x]), edge_index)) >= 0.5).float()
            x = (lit @ ll.weight.t() + ll.b >= 0).float()
        else:
            x = conv(torch.hstack([x, 1 - x]), edge_index)
            if hasattr(model, 'quant_out'):     # QAT: the binarisation the network's forward applies
                x = model.quant_out(x)
        xs.append(x)
    return xs


@torch.no_grad()
def symbolic_head(model, xs, batch):
    """Readout of node states xs, then the binarised head.

    Returns the head margins W.lit + b per class [G × C], the head literals [G × 2R],
    and the node-prototype similarities [N × K] (None without node prototypes).
    """
    s, s_node = readout_values(model, xs, batch)
    head_in = model.head_input(s) if hasattr(model, 'head_input') else torch.hstack([s, 1 - s])
    head_lit = model.fc.phi_in(head_in) >= 0.5
    margin = head_lit.float() @ model.fc.weight.t() + model.fc.b
    return margin, head_lit, s_node


@torch.no_grad()
def readout_values(model, xs, batch):
    """The head's positive inputs s [G × R] (the model feeds it [s, 1 - s], or with sum
    pooling [s, 1 - s_bounded]), and the node-prototype similarities [N × K] or None.
    A graph-level sum block is read through phi_sum, binarised like every literal."""
    level = level_of(model)
    h = torch.hstack(xs)
    s_node = None
    if level == 'base':
        # classic LogiX-GIN: the head reads the pooled trunk units directly, with the same
        # 1 - x negation the model applies (also to the unbounded sum, as upstream does)
        if hasattr(model, 'readout'):                 # GINTELLPool: its own pooling ops
            s = model.readout(xs, batch)
        else:
            s = torch.hstack([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)])
    elif level == 'node' and hasattr(model, 'predicate'):
        # model_proto_count: the binarised node predicates q [N × K], pooled as counts / shares
        s_node = model.predicate(h, symbolic=True)
        s = model.pool_predicate(s_node, batch)
    elif level == 'node':
        s_node = model.proto.similarity(h)
        s = model._pool(s_node, batch, model.pool_ops)
    elif level == 'graph':
        s = model.proto.similarity(model.graph_input(h, batch, symbolic=True))
    else:
        s_node = model.proto_node.similarity(h)
        s = torch.hstack([model._pool(s_node, batch, model.node_pool_ops),
                          model.proto_graph.similarity(model.graph_input(h, batch, symbolic=True))])
    return s, s_node


def shortest_explanation(w, threshold, true_lits):
    """Fewest true literals whose weights reach the threshold (weights >= 0, so greedy by
    decreasing weight is exact). Returns their indices, or None if they cannot reach it."""
    idx = np.flatnonzero(true_lits)
    order = idx[np.argsort(-w[idx], kind='stable')]
    if threshold <= 0:
        return order[:0]
    reached = np.flatnonzero(np.cumsum(w[order]) >= threshold)
    return order[:reached[0] + 1] if len(reached) else None


def prototype_table(model):
    """Per prototype layer: cared bits [K × d], and the trunk unit (layer·h + unit) of every bit."""
    level = level_of(model)
    return [{'care': (p.care.detach().cpu() > 0.5).numpy(),
             'unit_of_bit': np.array([l * model.hidden_dim + u for _, l, u in proto_dim_map(model, level, li)])}
            for li, p in enumerate(model.proto_layers)]


def trunk_shape(model):
    """(num_layers, hidden_dim) of the trunk, for prototype and classic models alike."""
    return len(model.convs), model.convs[0].nn[0].out_features


def base_literal_map(model):
    """Classic LogiX-GIN head literal i -> (trunk unit, pool op, positive?).

    The head reads hstack([p, 1 - p]) with p = [mean | max | sum] of the L·h trunk units
    (or the pooling ops of a GINTELLPool).
    """
    L, h = trunk_shape(model)
    n = L * h
    ops = getattr(model, 'pool_ops', BASE_POOLS)
    R = len(ops) * n
    return [((i % R) % n, ops[(i % R) // n], i < R) for i in range(2 * R)]


def head_literal_map(model):
    """Head literal i -> (prototype layer, prototype k, pool op, positive?)."""
    level = level_of(model)
    cols = readout_columns(model, level)
    lits = []
    for c, pos in model.head_columns():       # sum-pooled columns have no negation
        kind, k, op = cols[c]
        li = 1 if (level == 'both' and kind == 'graph') else 0
        lits.append((li, k, op, pos))
    return lits


def head_literal_constants(model):
    """Per head literal: True / False when it has that value on every input, else None.

    Exact for bounded readout columns (a sum-pooled count is never declared constant): the
    literal is true on all of [0, 1] or nowhere (min_covers.literal_true_intervals), its
    prototype cares about no bit (similarity 1 on every graph), or it reads a masked node
    prototype's max similarity exp(-m*/T) and is true (or false) for every possible
    mismatch count m* in 0..d. A constant-true literal acts as a bias of every class rule
    that weighs it; explanations in evidence (input-dependent literals) leave it out."""
    level = level_of(model)
    phi = model.fc.phi_in
    if level == 'base':
        lits = base_literal_map(model)
        R = len(lits) // 2
        cols = [(i % R, i < R) for i in range(2 * R)]
        layout = [{'op': op} for _, op, _ in lits[:R]]
        bounded = [col['op'] != 'sum' for col in layout]
    else:
        cols = [(int(c), bool(p)) for c, p in model.head_columns()]
        layout = [{'proto_layer': 1 if (level == 'both' and kind == 'graph') else 0, 'k': int(k), 'op': op}
                  for kind, k, op in readout_columns(model, level)]
        b = getattr(model, '_bounded', None)
        bounded = [b is None or bool(b[c]) for c in range(len(layout))]
    out = []
    for i, (c, pos) in enumerate(cols):
        if not bounded[c]:
            out.append(None)
            continue
        ivs = phi_true_intervals(phi, i, 0.0, 1.0)
        if not ivs:
            out.append(False)
            continue
        if len(ivs) == 1 and ivs[0][0] <= 1e-6 and ivs[0][1] >= 1 - 1e-6:
            out.append(True)
            continue
        col = layout[c]
        if 'k' not in col:
            out.append(None)
            continue
        li, k = col['proto_layer'], col['k']
        p = model.proto_layers[li]
        if not bool((p.care[k] > 0.5).any()):
            v = 1.0 if pos else 0.0
            out.append(any(a - 1e-9 <= v <= e + 1e-9 for a, e in ivs))
            continue
        node = level == 'node' or (level == 'both' and li == 0)
        if col['op'] == 'max' and node and p.masked:
            s_ivs = ivs if pos else [(1 - e, 1 - a) for a, e in ivs][::-1]
            T, d = float(p.mask_temp), int((p.care[k] > 0.5).sum())
            ms = [m for m in range(d + 1) if any(a - 1e-9 <= np.exp(-m / T) <= e + 1e-9 for a, e in s_ivs)]
            if not ms:
                out.append(False)
                continue
            if len(ms) == d + 1:
                out.append(True)
                continue
        out.append(None)
    return out


def evidence_explanation(w, threshold, true_lits, constants):
    """shortest_explanation in input-dependent literals only: the constant-true literals
    are counted in as a bias (their weight is taken off the threshold), then the fewest
    true non-constant literals reaching what remains. None exactly when the rule does not
    fire; [] when the bias alone fires it."""
    const_true = np.array([v is True for v in constants])
    varying = np.array([v is None for v in constants])
    return shortest_explanation(w, threshold - w[const_true].sum(), np.asarray(true_lits, dtype=bool) & varying)


@torch.no_grad()
def interp_metrics(model, dataset, device, ds_name=None, records=False):
    """The metrics of the module docstring. records=True also returns, per graph, the
    rule model's class, the evidence explanation (literal indices, None when the class
    rule does not fire) and the nodes it cites (cited_nodes; None for graph prototypes)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    net = evaluate(model, loader, device)
    out, y = predict(model, loader, device)
    net_pred = out.argmax(1)

    W = model.fc.weight.detach().cpu().numpy()
    S = (-model.fc.b).detach().cpu().numpy()
    base = level_of(model) == 'base'
    lit_map = base_literal_map(model) if base else head_literal_map(model)
    protos = None if base else prototype_table(model)
    truth = spec(ds_name).ground_truth if ds_name else None
    gt = truth is not None and level_of(model) in ('node', 'both', 'base')
    gt_classes = truth.classes if gt else ()
    constants = head_literal_constants(model)
    KINDS = ('', '_evidence')                         # every literal / input-dependent literals only

    node_level = base or level_of(model) in ('node', 'both')
    predicate = hasattr(model, 'predicate')
    margins, lits, head_pred, bit_agree, n_nodes, recs = [], [], [], [], [], []
    gt_hits, gt_chance = {k: [] for k in KINDS}, {k: [] for k in KINDS}
    expl = {k: [] for k in KINDS}                     # per graph: literal indices or None

    def cited_nodes(e, nodes):
        """Per positive node-level literal of explanation e, the nodes of the graph (mask over
        `nodes`) it points at: the best-matching nodes of its prototype ("some node
        matches"), or for classic models the nodes where its trunk unit holds."""
        out = []
        for i in e:
            if base:
                u, op, pos = lit_map[i]
                best = (h[nodes, u] >= 0.5).cpu()
                if not pos or not best.any():
                    continue
            else:
                li, k, op, pos = lit_map[i]
                if not pos or li != 0 or s_node is None:
                    continue
                sim = s_node[nodes, k].cpu()
                if predicate:               # model_proto_count: the nodes satisfying P_k
                    best = sim >= 0.5
                    if not best.any():
                        continue
                else:
                    best = sim >= sim.max() - 1e-6
            out.append(best)
        return out

    g0 = 0
    for data in loader:
        data = data.to(device)
        xf = data.x.float()
        xs = trunk_states(model, xf, data.edge_index)
        xs_net = trunk_states(model, xf, data.edge_index, symbolic=False)
        bit_agree.append(torch.stack([((n >= 0.5).float() == b).float().mean() for n, b in zip(xs_net, xs)]))
        n_nodes.append(len(xf))
        head_pred.append(symbolic_head(model, xs_net, data.batch)[0].argmax(1).cpu().numpy())
        margin, head_lit, s_node = symbolic_head(model, xs, data.batch)
        margin, head_lit = margin.cpu().numpy(), head_lit.cpu().numpy()
        h = torch.hstack(xs)
        for g in range(data.num_graphs):
            c = int(margin[g].argmax())
            es = {'': shortest_explanation(W[c], S[c], head_lit[g]),
                  '_evidence': evidence_explanation(W[c], S[c], head_lit[g], constants)}
            for kind, e in es.items():
                expl[kind].append(e)
            nodes = torch.nonzero(data.batch == g).flatten()
            if records:
                e = es['_evidence']
                cited = None if not node_level else (
                    [] if e is None else sorted({int(n) for m in cited_nodes(e, nodes) for n in torch.nonzero(m).flatten()}))
                recs.append({'pred': c, 'expl': None if e is None else [int(i) for i in e], 'nodes': cited})
            if not gt or es[''] is None or c not in gt_classes or int(y[g0 + g]) != c:
                continue
            region = spec(ds_name).region(xf[nodes].cpu(), _subgraph_edges(data.edge_index, nodes).cpu())
            if not region.any():
                continue
            for kind, e in es.items():
                for best in cited_nodes(e, nodes):
                    gt_hits[kind].append(float(region[best].float().mean()))
                    gt_chance[kind].append(float(region.float().mean()))
        margins.append(margin)
        lits.append(head_lit)
        g0 += data.num_graphs
    margin = np.vstack(margins)
    head_lit = np.vstack(lits)
    logic_pred = margin.argmax(1)

    def bits(li, k):
        return int(protos[li]['care'][k].sum())

    def stat(v, name):
        return {f'{name}_mean': float(np.mean(v)) if len(v) else None,
                f'{name}_median': float(np.median(v)) if len(v) else None}

    def explanation_metrics(expls, sfx):
        """Size, citations and purity of one kind of explanation (every literal, or evidence)."""
        backed = [e for e in expls if e is not None]
        cites = Counter(int(i) for e in backed for i in e)
        if base:
            # every head literal is one condition on one pooled trunk unit
            cited_protos = set()
            expl_bits = [len(e) for e in backed]
            units = {int(lit_map[i][0]) for i in cites}
        else:
            cited_protos = {(lit_map[i][0], lit_map[i][1]) for i in cites}
            expl_bits = [sum(bits(*pk) for pk in {(lit_map[i][0], lit_map[i][1]) for i in e}) for e in backed]
            units = {int(u) for li, k in cited_protos for u in protos[li]['unit_of_bit'][protos[li]['care'][k]]}
        unit_layers = [u // trunk_shape(model)[1] for u in units]
        purity = []
        for i, n in cites.items():
            holds = head_lit[:, i]
            purity += [float(np.bincount(y[holds], minlength=W.shape[0]).max() / holds.sum())] * n
        return {**stat([len(e) for e in backed], f'expl_literals{sfx}'),
                **stat(expl_bits, f'expl_bits{sfx}'),
                f'protos_cited{sfx}': None if base else len(cited_protos),
                f'bits_per_proto_median{sfx}': float(np.median([bits(*pk) for pk in cited_protos])) if cited_protos else None,
                f'units_cited{sfx}': len(units),
                f'units_cited_L0{sfx}': float(np.mean([l == 0 for l in unit_layers])) if units else None,
                f'literal_purity{sfx}': float(np.mean(purity)) if purity else None}

    backed = [e for e in expl[''] if e is not None]
    res = {**net,
           'logic_acc': float((logic_pred == y).mean()),
           'logic_fidelity': float((logic_pred == net_pred).mean()),
           'head_fidelity': float((np.concatenate(head_pred) == net_pred).mean()),
           'head_acc': float((np.concatenate(head_pred) == y).mean()),
           'trunk_bit_agreement': float((torch.stack(bit_agree).mean(1).cpu().numpy() * n_nodes).sum() / sum(n_nodes)),
           'rule_backed': float(len(backed) / len(expl[''])),
           **explanation_metrics(expl[''], ''),
           'mask_temp': None if base or not model.proto_layers[0].masked else float(model.proto_layers[0].mask_temp),
           'head_literals_constant': float(np.mean([v is not None for v in constants])),
           **explanation_metrics(expl['_evidence'], '_evidence')}
    if gt:
        for kind in KINDS:
            res.update({f'gt_precision{kind}': float(np.mean(gt_hits[kind])) if gt_hits[kind] else None,
                        f'gt_chance{kind}': float(np.mean(gt_chance[kind])) if gt_chance[kind] else None,
                        f'gt_n{kind}': len(gt_hits[kind])})
    return (res, recs) if records else res


def _subgraph_edges(edge_index, nodes):
    """Edges among `nodes` (a contiguous batch block), relabelled from 0."""
    lo, hi = int(nodes[0]), int(nodes[-1]) + 1
    keep = (edge_index[0] >= lo) & (edge_index[0] < hi)
    return edge_index[:, keep] - lo


def run(run_path, split, ckpt, device, dataset=None):
    run = RunDir(run_path, dataset)
    res = interp_metrics(run.model(device, ckpt), run.pickled(split), device, run.dataset)
    with open(os.path.join(run_path, f'interp_{split}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', nargs='+', required=True, help='seed directories holding best.pt and data.pkl')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--dataset', default=None, help='default: read from the run path')
    ap.add_argument('--out', default=None, help='also write the table as CSV here')
    a = ap.parse_args()
    device = get_device()
    rows = []
    for p in a.run_path:
        rows.append({'run_path': p, **run(p, a.split, a.ckpt, device, a.dataset)})
    df = pd.DataFrame(rows)
    cols = ['acc', 'auc', 'logic_acc', 'logic_fidelity', 'head_fidelity', 'trunk_bit_agreement', 'rule_backed', 'expl_literals_median',
            'expl_bits_median', 'protos_cited', 'bits_per_proto_median', 'units_cited', 'units_cited_L0',
            'literal_purity', 'gt_precision', 'gt_chance', 'mask_temp']
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.precision', 3):
        print(df[[c for c in cols if c in df]].rename(index=lambda i: f'#{i}').to_string())
    for i, p in enumerate(a.run_path):
        print(f'#{i} {p}')
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
