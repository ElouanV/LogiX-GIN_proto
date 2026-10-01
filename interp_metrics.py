"""Interpretability metrics of a prototype LogiX-GIN checkpoint, next to its accuracy and AUC.

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
    gt_precision / gt_chance Mutagenicity, node prototypes only: share of the best-matching
                             nodes of the cited positive prototype literals that lie within
                             one hop of an NO2 / NH2 group, on mutagens that contain one,
                             and the same share for a uniformly random node (the baseline)
    mask_temp                temperature stored in the checkpoint (1 = strict AND)

Per-prediction metrics are over rule-backed graphs, as mean and median.

Usage:
    python interp_metrics.py --run_path <seed dir> [<seed dir> ...] --split test --out table.csv
Writes <run_path>/interp_<split>.json next to each checkpoint and prints one row per run.
"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from explain_proto import proto_dim_map, readout_columns
from latent_logic import ATOMS
from models_proto.model_proto import MODELS
from unpack_rules import dataset_from_path
from utils.evaluation import evaluate, predict

LEVELS = {cls.__name__: level for level, cls in MODELS.items()}
MUTAGEN = 0                       # Mutagenicity label 0 = mutagen (latent_logic.CLASSES)


def level_of(model):
    return LEVELS[type(model).__name__]


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
        xs.append(x)
    return xs


@torch.no_grad()
def symbolic_head(model, xs, batch):
    """Prototype readout of node states xs, then the binarised head.

    Returns the head margins W.lit + b per class [G × C], the head literals [G × 2R],
    and the node-prototype similarities [N × K] (None without node prototypes).
    """
    level = level_of(model)
    h = torch.hstack(xs)
    s_node = None
    if level == 'node':
        s_node = model.proto.similarity(h)
        s = model._pool(s_node, batch, model.pool_ops)
    elif level == 'graph':
        s = model.proto.similarity(model._pool(h, batch, model.pool_ops))
    else:
        s_node = model.proto_node.similarity(h)
        s = torch.hstack([model._pool(s_node, batch, model.node_pool_ops),
                          model.proto_graph.similarity(model._pool(h, batch, model.graph_pool_ops))])
    head_lit = model.fc.phi_in(torch.hstack([s, 1 - s])) >= 0.5
    margin = head_lit.float() @ model.fc.weight.t() + model.fc.b
    return margin, head_lit, s_node


def shortest_explanation(w, threshold, true_lits):
    """Fewest true literals whose weights reach the threshold (weights >= 0, so greedy by
    decreasing weight is exact). Returns their indices, or None if they cannot reach it."""
    idx = np.flatnonzero(true_lits)
    order = idx[np.argsort(-w[idx], kind='stable')]
    if threshold <= 0:
        return order[:0]
    reached = np.flatnonzero(np.cumsum(w[order]) >= threshold)
    return order[:reached[0] + 1] if len(reached) else None


def toxicophore_nodes(x, edge_index, hops=1):
    """Nodes within `hops` of an NO2 or NH2 group (N with >= 2 O, resp. >= 2 H, neighbours)."""
    atom = x.argmax(1)
    N, O, H = ATOMS.index('N'), ATOMS.index('O'), ATOMS.index('H')
    src, dst = edge_index
    region = torch.zeros(len(atom), dtype=torch.bool)
    for n in torch.nonzero(atom == N).flatten().tolist():
        nb = dst[src == n]
        for other in (O, H):
            grp = nb[atom[nb] == other]
            if len(grp) >= 2:
                region[n] = True
                region[grp] = True
    for _ in range(hops):
        grow = region.clone()
        grow[dst[region[src]]] = True
        region = grow
    return region


def prototype_table(model):
    """Per prototype layer: cared bits [K × d], and the trunk unit (layer·h + unit) of every bit."""
    level = level_of(model)
    return [{'care': (p.care.detach().cpu() > 0.5).numpy(),
             'unit_of_bit': np.array([l * model.hidden_dim + u for _, l, u in proto_dim_map(model, level, li)])}
            for li, p in enumerate(model.proto_layers)]


def head_literal_map(model):
    """Head literal i -> (prototype layer, prototype k, pool op, positive?)."""
    level = level_of(model)
    cols = readout_columns(model, level)
    R = len(cols)
    lits = []
    for i in range(2 * R):
        kind, k, op = cols[i % R]
        li = 1 if (level == 'both' and kind == 'graph') else 0
        lits.append((li, k, op, i < R))
    return lits


@torch.no_grad()
def interp_metrics(model, dataset, device, ds_name=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    net = evaluate(model, loader, device)
    out, y = predict(model, loader, device)
    net_pred = out.argmax(1)

    W = model.fc.weight.detach().cpu().numpy()
    S = (-model.fc.b).detach().cpu().numpy()
    lit_map = head_literal_map(model)
    protos = prototype_table(model)
    gt = ds_name == 'Mutagenicity' and level_of(model) in ('node', 'both')

    margins, lits, head_pred, bit_agree, n_nodes = [], [], [], [], []
    gt_hits, gt_chance = [], []
    expl = []                                         # per graph: literal indices or None
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
        for g in range(data.num_graphs):
            c = int(margin[g].argmax())
            e = shortest_explanation(W[c], S[c], head_lit[g])
            expl.append(e)
            if not gt or e is None or c != MUTAGEN or int(y[g0 + g]) != MUTAGEN:
                continue
            nodes = torch.nonzero(data.batch == g).flatten()
            region = toxicophore_nodes(xf[nodes].cpu(), _subgraph_edges(data.edge_index, nodes).cpu())
            if not region.any():
                continue
            for i in e:
                li, k, op, pos = lit_map[i]
                if not pos or li != 0:                # node prototypes, "some node matches"
                    continue
                sim = s_node[nodes, k].cpu()
                best = sim >= sim.max() - 1e-6
                gt_hits.append(float(region[best].float().mean()))
                gt_chance.append(float(region.float().mean()))
        margins.append(margin)
        lits.append(head_lit)
        g0 += data.num_graphs
    margin = np.vstack(margins)
    head_lit = np.vstack(lits)
    logic_pred = margin.argmax(1)

    backed = [e for e in expl if e is not None]
    cites = Counter(int(i) for e in backed for i in e)
    cited_protos = {(lit_map[i][0], lit_map[i][1]) for i in cites}

    def bits(li, k):
        return int(protos[li]['care'][k].sum())

    expl_bits = [sum(bits(*pk) for pk in {(lit_map[i][0], lit_map[i][1]) for i in e}) for e in backed]
    units = {int(u) for li, k in cited_protos for u in protos[li]['unit_of_bit'][protos[li]['care'][k]]}
    unit_layers = [u // model.hidden_dim for u in units]

    purity = []
    for i, n in cites.items():
        holds = head_lit[:, i]
        purity += [float(np.bincount(y[holds], minlength=W.shape[0]).max() / holds.sum())] * n

    def stat(v, name):
        return {f'{name}_mean': float(np.mean(v)) if len(v) else None,
                f'{name}_median': float(np.median(v)) if len(v) else None}

    res = {**net,
           'logic_acc': float((logic_pred == y).mean()),
           'logic_fidelity': float((logic_pred == net_pred).mean()),
           'head_fidelity': float((np.concatenate(head_pred) == net_pred).mean()),
           'trunk_bit_agreement': float((torch.stack(bit_agree).mean(1).cpu().numpy() * n_nodes).sum() / sum(n_nodes)),
           'rule_backed': float(len(backed) / len(expl)),
           **stat([len(e) for e in backed], 'expl_literals'),
           **stat(expl_bits, 'expl_bits'),
           'protos_cited': len(cited_protos),
           'bits_per_proto_median': float(np.median([bits(*pk) for pk in cited_protos])) if cited_protos else None,
           'units_cited': len(units),
           'units_cited_L0': float(np.mean([l == 0 for l in unit_layers])) if units else None,
           'literal_purity': float(np.mean(purity)) if purity else None,
           'mask_temp': float(model.proto_layers[0].mask_temp) if model.proto_layers[0].masked else None}
    if gt:
        res.update({'gt_precision': float(np.mean(gt_hits)) if gt_hits else None,
                    'gt_chance': float(np.mean(gt_chance)) if gt_chance else None,
                    'gt_n': len(gt_hits)})
    return res


def _subgraph_edges(edge_index, nodes):
    """Edges among `nodes` (a contiguous batch block), relabelled from 0."""
    lo, hi = int(nodes[0]), int(nodes[-1]) + 1
    keep = (edge_index[0] >= lo) & (edge_index[0] < hi)
    return edge_index[:, keep] - lo


def run(run_path, split, ckpt, device, dataset=None):
    model = torch.load(os.path.join(run_path, ckpt), map_location=device, weights_only=False)
    data = pickle.load(open(os.path.join(run_path, 'data.pkl'), 'rb'))
    res = interp_metrics(model, data[f'{split}_dataset'], device, dataset or dataset_from_path(run_path))
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
