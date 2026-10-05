"""Figures that open one trained prototype LogiX-GIN: its unit vocabulary, its prototypes,
and worked examples of its explanations.

Everything is read off the *symbolic* model, as in interp_metrics.py (binary trunk
states, binarised head literals, shortest sufficient explanation of each prediction), so
a figure shows the explanation a reader gets, not the soft network.

    explain_graphs      per graph: label, network / rule prediction, explanation literals,
                        node-prototype similarities
    plot_vocabulary     prototypes x trunk units they care about (must be 1 / must be 0)
    plot_prototypes     one card per cited prototype: its AND rule, the decoded layer-0
                        units, how often explanations cite it, and its best-matching
                        training node drawn in its graph
    plot_examples       test graphs with the nodes matching each cited prototype marked,
                        the NO2/NH2 region shaded (Mutagenicity) and the explanation written
"""
import os
import pickle
import textwrap
from collections import Counter

import matplotlib
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from explain_proto import readout_columns
from interp_metrics import (MUTAGEN, _subgraph_edges, head_literal_map, level_of, shortest_explanation,
                            symbolic_head, toxicophore_nodes, trunk_states)
from latent_logic import ATOMS, CLASSES, collect_activations, explain_component
from utils.evaluation import predict
from utils.viz import AQUA, BLUE, INK, INK2, ORANGE, save  # noqa: F401  (save re-exported for callers)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PROTO_COLORS = (BLUE, ORANGE, AQUA)        # at most three prototypes marked per drawing
NODE_FILL, REGION_FILL, DONT_CARE = '#e9e8e3', '#f6e3a1', '#f4f3ef'


def load_run(run_path, ckpt='best.pt', device='cpu'):
    model = torch.load(os.path.join(run_path, ckpt), map_location=device, weights_only=False)
    model.eval()
    data = pickle.load(open(os.path.join(run_path, 'data.pkl'), 'rb'))
    return model, data


def class_name(c, ds):
    return CLASSES[c] if ds == 'Mutagenicity' and c < len(CLASSES) else f'class {c}'


# ---------------------------------------------------------------------------
# explanations per graph
# ---------------------------------------------------------------------------

@torch.no_grad()
def explain_graphs(model, dataset, device='cpu'):
    """One record per graph of `dataset` (node-level prototype models).

    Keys: i, y, net, rule, backed, expl (head literal indices of the shortest sufficient
    explanation, or None), x (atom one-hot), edge_index (local), s_node [n x K]
    (similarity of each node to each prototype, on the binary trunk states).
    """
    if level_of(model) != 'node':
        raise ValueError('worked examples need node-level prototypes')
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    out, y = predict(model, loader, device)
    net = out.argmax(1)
    W = model.fc.weight.detach().cpu().numpy()
    S = (-model.fc.b).detach().cpu().numpy()
    recs, g0 = [], 0
    for data in loader:
        data = data.to(device)
        xf = data.x.float()
        xs = trunk_states(model, xf, data.edge_index)
        margin, head_lit, s_node = symbolic_head(model, xs, data.batch)
        margin, head_lit = margin.cpu().numpy(), head_lit.cpu().numpy()
        for g in range(data.num_graphs):
            nodes = torch.nonzero(data.batch == g).flatten()
            c = int(margin[g].argmax())
            e = shortest_explanation(W[c], S[c], head_lit[g])
            recs.append({'i': g0 + g, 'y': int(y[g0 + g]), 'net': int(net[g0 + g]), 'rule': c,
                         'backed': e is not None, 'expl': None if e is None else [int(i) for i in e],
                         'x': xf[nodes].cpu(), 'edge_index': _subgraph_edges(data.edge_index, nodes).cpu(),
                         's_node': s_node[nodes].cpu().numpy()})
        g0 += data.num_graphs
    return recs


def citations(model, recs):
    """Prototype k -> number of explanations citing it, and per predicted class."""
    lit_map = head_literal_map(model)
    total, by_class = Counter(), {}
    for r in recs:
        for k in {lit_map[i][1] for i in r['expl'] or []}:
            total[k] += 1
            by_class.setdefault(k, Counter())[r['rule']] += 1
    return total, by_class


def literal_citations(model, recs):
    """Prototype k -> Counter of the head literals (indices) citing it in explanations."""
    lit_map = head_literal_map(model)
    out = {}
    for r in recs:
        for i in r['expl'] or []:
            out.setdefault(lit_map[i][1], Counter())[i] += 1
    return out


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

def unit_name(model, d):
    l, u = divmod(int(d), model.hidden_dim)
    return f'L{l}u{u}'


def prototype_rule(model, k):
    """The AND rule of (masked) prototype k: 'L0u3 ∧ ¬L1u12 ∧ ...' (cared bits, by layer)."""
    p = model.proto
    care = (p.care.detach()[k] > 0.5).cpu().numpy()
    bits = (p.prototypes.detach()[k] > 0.5).cpu().numpy()
    terms = [('' if bits[d] else '¬') + unit_name(model, d) for d in np.flatnonzero(care)]
    return ' ∧ '.join(terms) if terms else 'TRUE (cares about no bit)'


def head_literal_interval(model, i):
    """Interval (lo, hi) of the readout value s on which head literal i is true.

    The head literal reads phi([s, 1 - s])_i >= 0.5; phi is evaluated on a grid of its
    input, then mapped back to s (literals of the 1 - s half flip the interval).
    """
    if getattr(model, 'has_unbounded_readout', False):
        raise NotImplementedError('sum-pooled readout: counts are not in the [0, 1] grid read here')
    _, _, _, pos = head_literal_map(model)[i]
    phi = model.fc.phi_in
    grid = torch.linspace(0, 1, 1001)
    with torch.no_grad():
        true = (phi(grid[:, None].expand(-1, phi.w.shape[0]))[:, i] >= 0.5).numpy()
    s = grid.numpy() if pos else 1 - grid.numpy()
    return (float(s[true].min()), float(s[true].max())) if true.any() else (np.nan, np.nan)


def literal_kind(model, i):
    """'always' (true for every s: a bias), 'present' (needs high similarity: some node /
    the nodes on average match the prototype), 'absent' (needs low similarity), 'band'."""
    lo, hi = head_literal_interval(model, i)
    if lo <= 1e-3 and hi >= 0.999:
        return 'always'
    if hi >= 0.999:
        return 'present'
    if lo <= 1e-3:
        return 'absent'
    return 'band'


def head_literal_text(model, i):
    """Readable form of head literal i, e.g. 'some node matches P3'."""
    _, k, op, _ = head_literal_map(model)[i]
    lo, hi = head_literal_interval(model, i)
    kind = literal_kind(model, i)
    if kind == 'always':
        return f'{op} match(P{k}) ≥ 0: always true, acts as a bias'
    strict = model.proto.masked and float(model.proto.mask_temp) <= 1.0
    if op == 'max' and strict:
        if kind == 'present' and lo > np.exp(-1):
            return f'some node matches P{k}'
        if kind == 'absent' and hi < 0.999:
            return f'no node matches P{k}'
    what = f'{op} over nodes of match(P{k})'
    if kind == 'present':
        return f'{what} ≥ {lo:.2f}'
    if kind == 'absent':
        return f'{what} ≤ {hi:.2f}'
    return f'{what} in [{lo:.2f}, {hi:.2f}]'


def decode_units(model, units, train_dataset, device='cpu', max_graphs=1500, max_rules=2):
    """Atom-count rules of layer-0 units (latent_logic.explain_component, non-recursive).

    Deeper units are not decoded: their minimal covers number ~1e5-1e10 (CLAUDE.md).
    """
    l0 = [d for d in units if int(d) // model.hidden_dim == 0]
    if not l0:
        return {}
    loader = DataLoader(train_dataset[:max_graphs], batch_size=128, shuffle=False)
    acts = collect_activations(model, loader, device)
    out = {}
    for d in l0:
        u = int(d) % model.hidden_dim
        comp = explain_component(model, 0, u, acts, max_rule_len=3, max_rules=max_rules, recursive=False)
        rules = comp['components'][(0, u)]['rules']
        terms = []
        for r in rules[:max_rules]:
            if r.get('always_true'):
                terms.append('always true')
            elif r['literals']:
                terms.append(' ∧ '.join(nm.replace('#', 'n_') for _, _, nm in r['literals']))
        out[unit_name(model, d)] = ' ∨ '.join(f'({t})' if len(terms) > 1 else t for t in terms) or 'no short rule'
    return out


# ---------------------------------------------------------------------------
# drawing a graph
# ---------------------------------------------------------------------------

def _layout(edge_index, n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edge_index.t().tolist())
    return g, nx.kamada_kawai_layout(g) if n > 1 else {0: (0, 0)}


def draw_graph(ax, x, edge_index, marks=None, region=None, title=None, node_size=150):
    """Atoms as grey discs with their symbol; marks = {node: [colours]} concentric rings;
    region shaded."""
    marks, region = marks or {}, set(int(v) for v in (region if region is not None else []))
    n = len(x)
    g, pos = _layout(edge_index, n)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='#b8b7b1', width=1.2)
    if region:
        nx.draw_networkx_nodes(g, pos, nodelist=sorted(region), ax=ax, node_color=REGION_FILL,
                               node_size=node_size * 3.2, linewidths=0)
    labels = {v: ATOMS[int(x[v].argmax())] if int(x[v].argmax()) < len(ATOMS) else '?' for v in range(n)}
    small = [v for v in range(n) if labels[v] == 'H']
    big = [v for v in range(n) if labels[v] != 'H']
    nx.draw_networkx_nodes(g, pos, nodelist=big, ax=ax, node_color=NODE_FILL, node_size=node_size,
                           edgecolors='#8d8c86', linewidths=0.6)
    nx.draw_networkx_nodes(g, pos, nodelist=small, ax=ax, node_color=NODE_FILL, node_size=node_size * 0.45,
                           edgecolors='#8d8c86', linewidths=0.4)
    for v, colors in marks.items():
        for j, color in enumerate([colors] if isinstance(colors, str) else colors):
            nx.draw_networkx_nodes(g, pos, nodelist=[v], ax=ax, node_color='none',
                                   node_size=node_size * (2.2 + 1.6 * j), edgecolors=color, linewidths=2.2)
    nx.draw_networkx_labels(g, pos, labels={v: labels[v] for v in big}, ax=ax, font_size=6.5, font_color=INK)
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc='left', fontsize=8.5)


def best_nodes(s_col, tol=1e-6):
    return np.flatnonzero(s_col >= s_col.max() - tol)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_vocabulary(models, cited=None, title=None):
    """Prototypes x the trunk units any prototype cares about, per model (masked models).

    Cell colour: must be 1 (blue), must be 0 (orange), don't care (pale). Columns are
    grouped by conv layer; rows of prototypes cited by at least one test explanation
    carry a '●'.
    """
    cited = cited or {}
    mats = {}
    for name, m in models.items():
        care = (m.proto.care.detach() > 0.5).cpu().numpy()
        bits = (m.proto.prototypes.detach() > 0.5).cpu().numpy()
        used = np.flatnonzero(care.any(0))
        mats[name] = (np.where(care, np.where(bits, 1, -1), 0)[:, used], used, m)
    widths = [max(len(v[1]), 4) for v in mats.values()]
    fig, axes = plt.subplots(1, len(mats), figsize=(min(18, 2.2 + 0.16 * sum(widths)), 3.6),
                             gridspec_kw={'width_ratios': widths, 'wspace': 0.25})
    cmap = ListedColormap([ORANGE, DONT_CARE, BLUE])
    for ax, (name, (mat, used, m)) in zip(np.atleast_1d(axes), mats.items()):
        ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect='auto', interpolation='none')
        ax.set_xticks(range(len(used)), [unit_name(m, d) for d in used], rotation=90, fontsize=6)
        cset = cited.get(name, set())
        ax.set_yticks(range(mat.shape[0]), [f'{"● " if k in cset else ""}P{k}' for k in range(mat.shape[0])],
                      fontsize=7)
        layers = used // m.hidden_dim
        for b in np.flatnonzero(np.diff(layers)) + 0.5:
            ax.axvline(b, color=INK, lw=1)
        ax.grid(False)
        ax.set_title(f'{name}: {len(used)} units, {int((mat != 0).sum())} cared bits', loc='left', fontsize=9)
    from matplotlib.patches import Patch
    np.atleast_1d(axes)[-1].legend(handles=[Patch(color=BLUE, label='must be 1'), Patch(color=ORANGE, label='must be 0'),
                                            Patch(facecolor=DONT_CARE, edgecolor=INK2, label="don't care")],
                                   loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7.5)
    fig.suptitle(title or 'Trunk units each prototype cares about (● = cited by a test explanation; '
                 'vertical lines separate conv layers)', x=0.01, ha='left', fontsize=10, y=1.04)
    return fig


@torch.no_grad()
def train_matches(model, train_dataset, device='cpu'):
    """Per prototype and class: (graph index, local node) of the first best-matching training
    node in a graph of that class, and the share of training graphs of each class having a
    node with similarity 1 (full match)."""
    loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    K = model.num_prototypes
    C = model.fc.weight.shape[0]
    best = [[(-1.0, None)] * C for _ in range(K)]
    full = np.zeros((K, C))
    count = np.zeros(full.shape[1])
    g0 = 0
    for data in loader:
        data = data.to(device)
        s = model.proto.similarity(torch.hstack(trunk_states(model, data.x.float(), data.edge_index))).cpu().numpy()
        batch = data.batch.cpu().numpy()
        ys = data.y.reshape(-1).cpu().numpy().astype(int)
        for g in range(data.num_graphs):
            sg = s[batch == g]
            count[ys[g]] += 1
            full[:, ys[g]] += sg.max(0) >= 1 - 1e-6
            for k in range(K):
                v = int(sg[:, k].argmax())
                if sg[v, k] > best[k][ys[g]][0]:
                    best[k][ys[g]] = (float(sg[v, k]), (g0 + g, v))
        g0 += data.num_graphs
    return [[b[1] for b in row] for row in best], full / np.maximum(count, 1)


def plot_prototypes(model, recs, train_dataset, ds='Mutagenicity', top=6, device='cpu'):
    """One row per most-cited prototype: rule, decoded L0 units, citations, train match."""
    total, by_class = citations(model, recs)
    protos = [k for k, _ in total.most_common(top)]
    if not protos:
        raise ValueError('no explanation cites a prototype')
    best, full = train_matches(model, train_dataset, device)
    care = (model.proto.care.detach() > 0.5).cpu().numpy()
    decoded = decode_units(model, np.flatnonzero(care[protos].any(0)), train_dataset, device)
    lit_cites = literal_citations(model, recs)
    fig, axes = plt.subplots(len(protos), 2, figsize=(12, 2.2 * len(protos)),
                             gridspec_kw={'width_ratios': [2.3, 1]})
    axes = np.atleast_2d(axes)
    for row, k in zip(axes, protos):
        tax, gax = row
        tax.set_axis_off()
        cls = ', '.join(f'{class_name(c, ds)} {n}' for c, n in by_class[k].most_common())
        lines = [f'P{k}   cited by {total[k]} test explanations ({cls})',
                 f'rule:  {prototype_rule(model, k)}',
                 'training graphs with a fully matching node: ' +
                 ', '.join(f'{class_name(c, ds)} {full[k, c]:.0%}' for c in range(full.shape[1]))]
        lines += [f'cited as: {head_literal_text(model, i)}  (x{n})' for i, n in lit_cites[k].most_common(3)]
        units = [unit_name(model, d) for d in np.flatnonzero(care[k]) if int(d) // model.hidden_dim == 0]
        for u in units:
            if u in decoded:
                lines += textwrap.wrap(f'{u} = {decoded[u]}', 125, initial_indent='   ', subsequent_indent='       ')
        tax.text(0, 1, lines[0], fontsize=9.5, fontweight='bold', va='top', color=INK, transform=tax.transAxes)
        tax.text(0, 0.8, '\n'.join(lines[1:]), fontsize=7.8, va='top', color=INK, family='monospace',
                 transform=tax.transAxes, wrap=True)
        c = by_class[k].most_common(1)[0][0]              # the class its explanations argue for
        gi, v = best[k][c] if best[k][c] is not None else next(b for b in best[k] if b is not None)
        d = train_dataset[gi]
        draw_graph(gax, d.x.float(), d.edge_index, marks={v: BLUE},
                   title=f'a best-matching training node (graph {gi}, {class_name(int(d.y), ds)})')
    fig.suptitle('Most-cited prototypes. Units: L<layer>u<unit>; ¬ = must be 0; layer-0 units decoded '
                 'into atom-count rules (n_X = number of X among the node and its neighbours)',
                 x=0.01, ha='left', fontsize=10, y=1.0)
    fig.tight_layout()
    return fig


def pick_examples(recs, ds='Mutagenicity'):
    """One graph per case: correct mutagen with an NO2/NH2 group, correct non-mutagen,
    wrong prediction, and rules disagreeing with the network (all rule-backed when possible)."""
    def first(cond):
        for r in recs:
            if cond(r):
                return r
        return None

    tox = lambda r: ds == 'Mutagenicity' and bool(toxicophore_nodes(r['x'], r['edge_index']).any())  # noqa: E731
    cases = [('correct, mutagen with NO2/NH2', lambda r: r['backed'] and r['y'] == r['net'] == r['rule'] == MUTAGEN
              and tox(r)),
             ('correct, non-mutagen', lambda r: r['backed'] and r['y'] == r['net'] == r['rule'] != MUTAGEN),
             ('wrong prediction', lambda r: r['backed'] and r['net'] == r['rule'] != r['y']),
             ('rules disagree with the network', lambda r: r['rule'] != r['net'])]
    return [(name, first(c)) for name, c in cases if first(c) is not None]


def plot_examples(model, recs, ds='Mutagenicity', examples=None):
    """Worked examples: each graph with the nodes matching its cited prototypes ringed."""
    examples = examples or pick_examples(recs, ds)
    lit_map = head_literal_map(model)
    fig, axes = plt.subplots(2, len(examples), figsize=(4.4 * len(examples), 5.6),
                             gridspec_kw={'height_ratios': [3, 0.8]})
    axes = np.atleast_2d(axes).reshape(2, -1)
    for (case, r), gax, tax in zip(examples, axes[0], axes[1]):
        marks, lines = {}, []
        protos = []
        for i in r['expl'] or []:
            k = lit_map[i][1]
            if literal_kind(model, i) == 'present' and k not in protos:
                protos.append(k)
        colors = dict(zip(protos, PROTO_COLORS))
        for k, color in colors.items():
            for v in best_nodes(r['s_node'][:, k]):
                marks.setdefault(int(v), []).append(color)
        region = np.flatnonzero(toxicophore_nodes(r['x'], r['edge_index']).numpy()) if ds == 'Mutagenicity' else []
        draw_graph(gax, r['x'], r['edge_index'], marks=marks, region=region,
                   title=f'{case}\ntrue {class_name(r["y"], ds)} · network {class_name(r["net"], ds)} · '
                         f'rules {class_name(r["rule"], ds)}')
        tax.set_axis_off()
        if r['expl'] is None:
            lines.append(f'no rule fires for "{class_name(r["rule"], ds)}": no sufficient explanation')
        else:
            lines.append(f'"{class_name(r["rule"], ds)}" because:')
            for i in r['expl']:
                _, k, _, _ = lit_map[i]
                ring = k in colors and literal_kind(model, i) == 'present'
                tag = f' [ring {["blue", "orange", "aqua"][protos.index(k)]}]' if ring else ''
                lines.append(f' • {head_literal_text(model, i)}{tag}')
        tax.text(0, 1, '\n'.join(lines[:9]) + ('\n …' if len(lines) > 9 else ''), va='top', fontsize=7.8,
                 color=INK, transform=tax.transAxes)
    fig.suptitle('Worked examples (test split). Rings: best-matching nodes of the prototypes whose presence '
                 'the explanation requires; shaded: within one hop of an NO2/NH2 group' if ds == 'Mutagenicity' else
                 'Worked examples (test split). Rings: best-matching nodes of the cited prototypes',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    return fig
