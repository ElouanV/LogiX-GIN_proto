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
    plot_trunk_match    one graph against one prototype: node x cared-unit trunk states,
                        and the full trunk of its best-matching node
    plot_unfolding      one prediction unfolded from the class down to atom counts: head
                        literal, prototype, its cared bits, and each true unit's shortest
                        sufficient literals, followed to a witness neighbour
    plot_examples       test graphs with the nodes matching each cited prototype marked,
                        the ground-truth region shaded (configs/datasets.yaml: NO2/NH2 on
                        Mutagenicity, the planted motifs on BA2Motifs / BAMultiShapes) and the
                        explanation written
"""
import textwrap
from collections import Counter

import matplotlib
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from interp_metrics import (_subgraph_edges, aggregate, evidence_explanation, head_literal_constants, head_literal_map,
                            level_of, shortest_explanation, symbolic_head, trunk_states)
from latent_logic import collect_activations, explain_component, literal_name
from min_covers import literal_true_intervals
from utils.datasets import spec
from utils.runs import EVAL_BATCH_SIZE
from utils.evaluation import predict
from utils.viz import AQUA, BLUE, INK, INK2, ORANGE, save  # noqa: F401  (save re-exported for callers)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PROTO_COLORS = (BLUE, ORANGE, AQUA)        # at most three prototypes marked per drawing
NODE_FILL, REGION_FILL, DONT_CARE = '#e9e8e3', '#f6e3a1', '#f4f3ef'


def class_name(c, ds):
    return spec(ds).class_name(c)


# ---------------------------------------------------------------------------
# explanations per graph
# ---------------------------------------------------------------------------

@torch.no_grad()
def explain_graphs(model, dataset, device='cpu', evidence=False):
    """One record per graph of `dataset` (node-level prototype models).

    Keys: i, y, net, rule, backed, expl (head literal indices of the shortest sufficient
    explanation, or None), x (atom one-hot), edge_index (local), s_node [n x K]
    (similarity of each node to each prototype, on the binary trunk states).
    evidence: expl in input-dependent literals only (interp_metrics.evidence_explanation:
    the constant-true head literals are a bias), so citations count real evidence.
    """
    if level_of(model) != 'node':
        raise ValueError('worked examples need node-level prototypes')
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    out, y = predict(model, loader, device)
    net = out.argmax(1)
    W = model.fc.weight.detach().cpu().numpy()
    S = (-model.fc.b).detach().cpu().numpy()
    constants = head_literal_constants(model) if evidence else None
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
            e = (evidence_explanation(W[c], S[c], head_lit[g], constants) if evidence
                 else shortest_explanation(W[c], S[c], head_lit[g]))
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


def head_literal_intervals(model, i):
    """Intervals [(lo, hi), ...] of the readout value s on which head literal i is true,
    in increasing s; [] if it is never true.

    The head literal reads phi([s, 1 - s])_i >= 0.5, true exactly on the intervals of
    min_covers.literal_true_intervals, mapped back to s (literals of the 1 - s half flip
    them). A periodic phi can be true on two bands, e.g. [0, .26] and [.75, 1].
    """
    if getattr(model, 'has_unbounded_readout', False):
        raise NotImplementedError('sum-pooled readout: counts are not in the [0, 1] range read here')
    _, _, _, pos = head_literal_map(model)[i]
    phi = model.fc.phi_in
    ivs = literal_true_intervals(float(phi.w[i]), float(phi.b[i]), 0.0, 1.0)
    return ivs if pos else [(1 - c, 1 - a) for a, c in ivs][::-1]


def head_literal_interval(model, i):
    """Hull (lo, hi) of the intervals of head_literal_intervals; (nan, nan) if never true."""
    runs = head_literal_intervals(model, i)
    return (runs[0][0], runs[-1][1]) if runs else (np.nan, np.nan)


def literal_kind(model, i):
    """'always' (true for every s: a bias), 'never', 'present' (needs high similarity: some
    node / the nodes on average match the prototype), 'absent' (needs low similarity),
    'band' (one interior interval), 'outside' (two bands: s low or s high), 'multi'."""
    runs = head_literal_intervals(model, i)
    if not runs:
        return 'never'
    if len(runs) == 1:
        lo, hi = runs[0]
        if lo <= 1e-3 and hi >= 0.999:
            return 'always'
        if hi >= 0.999:
            return 'present'
        if lo <= 1e-3:
            return 'absent'
        return 'band'
    if len(runs) == 2 and runs[0][0] <= 1e-3 and runs[1][1] >= 0.999:
        return 'outside'
    return 'multi'


def head_literal_text(model, i):
    """Readable form of head literal i, e.g. 'some node is within 3 mismatches of P3'.

    With masked prototypes s = exp(-m/T) (m = mismatches on cared bits), so a threshold
    on max over nodes of s is a threshold on the closest node's mismatch count m*.
    """
    _, k, op, _ = head_literal_map(model)[i]
    runs = head_literal_intervals(model, i)
    kind = literal_kind(model, i)
    if kind == 'always':
        return f'{op} match(P{k}) ≥ 0: always true, acts as a bias'
    if kind == 'never':
        return f'{op} match(P{k}): never true'
    if op == 'max' and model.proto.masked:
        T = float(model.proto.mask_temp)
        m_le = lambda lo: int(np.floor(-T * np.log(lo) + 1e-6)) if lo > 0 else None  # s ≥ lo ⇔ m* ≤ m_le
        m_ge = lambda hi: int(np.ceil(-T * np.log(hi) - 1e-6))                       # s ≤ hi ⇔ m* ≥ m_ge
        if kind == 'present':
            m_hi = m_le(runs[0][0])
            return f'some node matches P{k}' if m_hi == 0 else f'some node is within {m_hi} mismatch{"es" * (m_hi > 1)} of P{k}'
        if kind == 'absent':
            m_lo = m_ge(runs[0][1])
            return f'no node matches P{k}' if m_lo <= 1 else f'every node has ≥ {m_lo} mismatches with P{k}'
        if kind == 'band':
            return f'closest node to P{k} has {m_ge(runs[0][1])}-{m_le(runs[0][0])} mismatches'
        if kind == 'outside':
            return (f'closest node to P{k} has ≤ {m_le(runs[1][0])} or ≥ {m_ge(runs[0][1])} mismatches')
        return f'closest node to P{k}: ' + ' or '.join(f'{m_ge(hi)}-{m_le(lo)} mismatches' for lo, hi in runs)
    what = f'{op} over nodes of match(P{k})'
    if kind == 'present':
        return f'{what} ≥ {runs[0][0]:.2f}'
    if kind == 'absent':
        return f'{what} ≤ {runs[0][1]:.2f}'
    if kind == 'outside':
        return f'{what} ≤ {runs[0][1]:.2f} or ≥ {runs[1][0]:.2f}'
    return f'{what} in ' + ' ∪ '.join(f'[{lo:.2f}, {hi:.2f}]' for lo, hi in runs)


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


def draw_graph(ax, x, edge_index, marks=None, region=None, title=None, node_size=150, ds='Mutagenicity'):
    """Atoms as grey discs with their symbol (the feature name of ``ds``); marks = {node:
    [colours]} concentric rings; region shaded."""
    marks, region = marks or {}, set(int(v) for v in (region if region is not None else []))
    n = len(x)
    g, pos = _layout(edge_index, n)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='#b8b7b1', width=1.2)
    if region:
        nx.draw_networkx_nodes(g, pos, nodelist=sorted(region), ax=ax, node_color=REGION_FILL,
                               node_size=node_size * 3.2, linewidths=0)
    if n and bool((x == x[0]).all()):          # constant features (BA datasets): no symbol to show
        labels = {v: '' for v in range(n)}
    else:
        labels = {v: spec(ds).feature_name(int(x[v].argmax())) for v in range(n)}
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
        draw_graph(gax, d.x.float(), d.edge_index, marks={v: BLUE}, ds=ds,
                   title=f'a best-matching training node (graph {gi}, {class_name(int(d.y), ds)})')
    fig.suptitle('Most-cited prototypes. Units: L<layer>u<unit>; ¬ = must be 0; layer-0 units decoded '
                 'into atom-count rules (n_X = number of X among the node and its neighbours)',
                 x=0.01, ha='left', fontsize=10, y=1.0)
    fig.tight_layout()
    return fig


def pick_examples(recs, ds='Mutagenicity'):
    """One graph per case: correct, of a class scored against the ground truth and containing
    its region (a mutagen with an NO2/NH2 group, a BA graph with its motifs); correct, of
    another class; wrong prediction; rules disagreeing with the network (all rule-backed)."""
    def first(cond):
        for r in recs:
            if cond(r):
                return r
        return None

    truth = spec(ds).ground_truth
    scored = truth.classes if truth else ()
    correct = lambda r: r['backed'] and r['y'] == r['net'] == r['rule']  # noqa: E731
    gt = first(lambda r: correct(r) and r['y'] in scored and bool(spec(ds).region(r['x'], r['edge_index']).any()))
    gt_name = f'{class_name(gt["y"], ds)} with {truth.motif}' if gt else ''
    # a correct graph of a class without ground truth, else of another class than the first case
    other = first(lambda r: correct(r) and r['y'] not in scored) or \
        first(lambda r: correct(r) and (gt is None or r['y'] != gt['y']))
    cases = [(f'correct, {gt_name}', gt),
             (f'correct, {class_name(other["y"], ds)}' if other else '', other),
             ('wrong prediction', first(lambda r: r['backed'] and r['net'] == r['rule'] != r['y'])),
             ('rules disagree with the network', first(lambda r: r['rule'] != r['net']))]
    return [(name, r) for name, r in cases if r is not None]


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
        region = spec(ds).region(r['x'], r['edge_index'])
        region = [] if region is None else np.flatnonzero(region.numpy())
        draw_graph(gax, r['x'], r['edge_index'], marks=marks, region=region, ds=ds,
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
                 'the explanation requires; shaded: ' + spec(ds).ground_truth.shading if spec(ds).ground_truth else
                 'Worked examples (test split). Rings: best-matching nodes of the cited prototypes',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# inside one prediction: trunk states and the unfolded reasoning
# ---------------------------------------------------------------------------

@torch.no_grad()
def layer_literals(model, x, edge_index):
    """Binary trunk states [n x h] per layer, and per conv layer the neighbourhood counts its
    literals read [n x 2 d_in] with the literals' truth values (the symbolic model)."""
    xs = trunk_states(model, x, edge_index)
    lits = []
    for l, conv in enumerate(model.convs):
        inp = x if l == 0 else xs[l - 1]
        agg = aggregate(conv, torch.hstack([inp, 1 - inp]), edge_index)
        lits.append((agg, conv.nn[0].phi_in(agg) >= 0.5))
    return xs, lits


def _node_label(x, v, ds='Mutagenicity'):
    if len(x) and bool((x == x[0]).all()):
        return f'node {v}'
    return f'{spec(ds).feature_name(int(x[v].argmax()))}{v}'


def plot_trunk_match(model, rec, k, ds='Mutagenicity', max_nodes=14):
    """One graph against prototype k: the graph with the best-matching node ringed; the
    binary trunk states of the nodes closest to P_k on the units P_k cares about, under the
    values P_k requires (mismatches crossed); the full trunk of the best node, cared bits
    outlined."""
    xs = torch.hstack(trunk_states(model, rec['x'], rec['edge_index'])).numpy()
    care = (model.proto.care.detach()[k] > 0.5).cpu().numpy()
    want = (model.proto.prototypes.detach()[k] > 0.5).cpu().numpy().astype(float)
    cols = np.flatnonzero(care)
    if not len(cols):
        raise ValueError(f'P{k} cares about no bit')
    sim = rec['s_node'][:, k]
    order = np.argsort(-sim, kind='stable')[:max_nodes]
    v = int(order[0])
    L, h = len(model.convs), model.hidden_dim
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, max(len(cols), 6) * 0.11 + 0.6, 2.6],
                          height_ratios=[1, 1], wspace=0.35, hspace=0.5)
    gax = fig.add_subplot(gs[:, 0])
    region = spec(ds).region(rec['x'], rec['edge_index'])
    draw_graph(gax, rec['x'], rec['edge_index'], marks={v: BLUE}, ds=ds,
               region=[] if region is None else np.flatnonzero(region.numpy()),
               title=f'test graph {rec["i"]}\ntrue {class_name(rec["y"], ds)}, rules {class_name(rec["rule"], ds)}')
    hax = fig.add_subplot(gs[:, 1])
    mat = np.vstack([want[cols], xs[order][:, cols]])
    hax.imshow(mat, cmap=ListedColormap([NODE_FILL, INK2]), vmin=0, vmax=1, aspect='auto', interpolation='none')
    bad = xs[order][:, cols] != want[cols]
    for r, c in zip(*np.nonzero(bad)):
        hax.text(c, r + 1, '×', ha='center', va='center', color=ORANGE, fontsize=8, fontweight='bold')
    hax.axhline(0.5, color=BLUE, lw=2)
    hax.set_xticks(range(len(cols)), [unit_name(model, d) for d in cols], rotation=90, fontsize=6)
    hax.set_yticks(range(len(order) + 1), [f'P{k} requires'] +
                   [f'{_node_label(rec["x"], int(n), ds)}  {sim[n]:.2f}' for n in order], fontsize=6.5)
    for b in np.flatnonzero(np.diff(cols // h)) + 0.5:
        hax.axvline(b, color=INK, lw=0.8)
    hax.grid(False)
    hax.set_title(f'nodes by similarity to P{k}\n(dark = 1, × = mismatch)', loc='left', fontsize=8.5)
    tax = fig.add_subplot(gs[:, 2])
    full = xs[v].reshape(L, h)
    tax.imshow(full, cmap=ListedColormap([NODE_FILL, INK2]), vmin=0, vmax=1, aspect='auto', interpolation='none')
    from matplotlib.patches import Rectangle
    for d in cols:
        l, u = divmod(int(d), h)
        tax.add_patch(Rectangle((u - 0.5, l - 0.5), 1, 1, fill=False, lw=1.6, ec=BLUE if want[d] else ORANGE))
    tax.set_yticks(range(L), [f'L{l}' for l in range(L)], fontsize=8)
    tax.set_xticks(range(0, h, 8), [f'u{u}' for u in range(0, h, 8)], fontsize=6.5)
    tax.grid(False)
    tax.set_title(f'full trunk of {_node_label(rec["x"], v, ds)} ({int(full.sum())} of {L * h} units on)\n'
                  f'outlined: P{k} needs 1 (blue) / 0 (orange)', loc='left', fontsize=8.5)
    fig.suptitle(textwrap.fill(f'Trunk activation against prototype P{k} ({len(cols)} cared bits): '
                               f'{prototype_rule(model, k)}', 200), x=0.01, ha='left', fontsize=9, y=1.02)
    return fig


def _box_text(model, xs, lits, A, l, u, w, max_lits):
    """Shortest sufficient literals of unit (l, u) at node w, as (text, witness) pairs: the
    witness is a neighbour (or w) where the counted lower unit holds, for positive literals
    of layers >= 1. A unit that is 0 gets its exact reason instead: the weights of its true
    literals sum below its threshold. A: dense adjacency [n x n] (bool)."""
    ll = model.convs[l].nn[0]
    agg, true = lits[l]
    W, thr, t = ll.weight.detach()[u].cpu().numpy(), float(-ll.b.detach()[u]), true[w].cpu().numpy()
    e = shortest_explanation(W, thr, t)
    if e is None:                                  # the unit is 0: its true literals fall short
        return [(f'true literals sum {float(W[t].sum()):.1f} < {thr:.1f}', None)], 0
    n_in = ll.in_features // 2
    out = []
    for i in e[:max_lits]:
        i = int(i)
        txt = f'{literal_name(model, l, i).replace("#", "n_")} = {float(agg[w, i]):.0f}'
        wit = None
        if l > 0 and i < n_in:                     # a count of a lower unit being 1: find a node
            on = np.flatnonzero(xs[l - 1][:, i].cpu().numpy() > 0.5)
            nb = set(np.flatnonzero(A[w].numpy())) | {w}
            cand = [int(n) for n in on if n in nb]
            wit = ((l - 1, i), cand[0]) if cand else None
        out.append((txt, wit))
    return out, max(len(e) - max_lits, 0)


@torch.no_grad()
def plot_unfolding(model, rec, ds='Mutagenicity', k=None, max_bits=8, max_lits=3, depth=3):
    """One prediction unfolded down to atom counts.

    class <- the cited head literal 'some node is within m mismatches of P_k' <- P_k's cared bits at its
    best-matching node v <- for each true unit, its shortest sufficient set of literals
    (neighbourhood counts of lower units, or of atoms at L0) <- for a count of a lower unit,
    one witness neighbour where that unit holds, unfolded the same way. Everything is the
    symbolic model: the boxes are exact reasons, truncated for room (max_bits cared bits,
    required-1 bits first; max_lits literals per unit, '+n' = literals not drawn). A cared
    bit whose value at v differs from P_k's is shaded (a partial match)."""
    lit_map = head_literal_map(model)
    if k is None:
        present = [lit_map[i][1] for i in rec['expl'] or [] if literal_kind(model, i) == 'present']
        if not present:
            raise ValueError('the explanation cites no prototype presence')
        k = present[0]
    x, ei = rec['x'], rec['edge_index']
    xs, lits = layer_literals(model, x, ei)
    n = len(x)
    A = torch.zeros(n, n, dtype=torch.bool)
    A[ei[0], ei[1]] = True
    v = int(best_nodes(rec['s_node'][:, k])[0])
    h, L = model.hidden_dim, len(model.convs)
    care = (model.proto.care.detach()[k] > 0.5).cpu().numpy()
    want = (model.proto.prototypes.detach()[k] > 0.5).cpu().numpy()
    bits = [int(d) for d in np.flatnonzero(care)]
    shown = sorted(bits, key=lambda d: (not want[d], -(d // h)))[:max_bits]   # required 1 first
    n_pos = int(want[bits].sum())

    # boxes: (layer, unit, node) -> {'lines': [...], 'children': [...]}; layer L = prototype
    boxes, edges = {}, []
    frontier = [((d // h, d % h), v) for d in shown]
    for (l, u), w in frontier:
        boxes.setdefault((l, u, w), None)
    seen = set()
    stack = list(frontier)
    while stack:
        (l, u), w = stack.pop()
        if (l, u, w) in seen:
            continue
        seen.add((l, u, w))
        lines, more = _box_text(model, xs, lits, A, l, u, w, max_lits)
        boxes[(l, u, w)] = (lines, more)
        for txt, wit in lines:
            if wit is not None and l - 1 >= L - depth:
                (lc, uc), wc = wit
                edges.append(((lc, uc, wc), (l, u, w), txt))
                stack.append(((lc, uc), wc))

    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4.2], wspace=0.02)
    gax = fig.add_subplot(gs[0])
    layer_col = [AQUA, ORANGE, BLUE]
    marks = {}
    for (l, u, w) in boxes:
        marks.setdefault(w, [])
        c = layer_col[l % 3]
        if c not in marks[w]:
            marks[w].append(c)
    region = spec(ds).region(x, ei)
    draw_graph(gax, x, ei, marks=marks, region=[] if region is None else np.flatnonzero(region.numpy()), ds=ds,
               title=f'test graph {rec["i"]}: true {class_name(rec["y"], ds)} · rules {class_name(rec["rule"], ds)}\n'
                     f'rings = nodes used below (L0 aqua, L1 orange, L2 blue)')
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-0.2, L + 2.2)
    ax.set_axis_off()
    cols = {l: [b for b in boxes if b[0] == l] for l in range(L)}
    xy = {}
    for l in reversed(range(L)):                   # top layer by unit, lower ones under their parents
        def key(b):
            ys = [xy[c][1] for a, c, _ in edges if a == b and c in xy]
            return (-np.mean(ys) if ys else 0.0, b[1], b[2])
        bs = sorted(cols[l], key=key)
        for j, b in enumerate(bs):
            xy[b] = (l, 1 - (j + 0.5) / max(len(bs), 1))
    ax.set_ylim(-0.05, 1.08)
    for b, (bx, by) in xy.items():
        l, u, w = b
        lines, more = boxes[b]
        body = '\n'.join(t for t, _ in lines) + (f'\n+{more} more' if more else '')
        val = int(xs[l][w, u] > 0.5)
        miss = w == v and l * h + u in bits and val != int(want[l * h + u])
        head_line = f'L{l}u{u} = {val} at {_node_label(x, w, ds)}' + (f'  ✗ P{k} needs {1 - val}' if miss else '')
        ax.text(bx, by, f'{head_line}\n{body}', ha='center', va='center', fontsize=6.3,
                color=INK, family='monospace',
                bbox=dict(boxstyle='round,pad=0.35', fc='#fdeee6' if miss else 'white',
                          ec=layer_col[l % 3], lw=1.3))
    for a, b, _ in edges:
        if a in xy and b in xy:
            ax.annotate('', xy=(xy[b][0] - 0.33, xy[b][1]), xytext=(xy[a][0] + 0.33, xy[a][1]),
                        arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8, shrinkA=2, shrinkB=2))
    # prototype, head, class
    sim = float(rec['s_node'][v, k])
    ptxt = (f'P{k} at {_node_label(x, v, ds)}: similarity {sim:.2f}\n'
            f'needs {n_pos} units = 1, {len(bits) - n_pos} = 0 (shown: {len(shown)})\n'
            + textwrap.fill(prototype_rule(model, k), 46))
    ax.text(L, 0.55, ptxt, ha='center', va='center', fontsize=6.6, family='monospace', color=INK,
            bbox=dict(boxstyle='round,pad=0.4', fc='#eef4fb', ec=BLUE, lw=1.6))
    cared_at_v = {(d // h, d % h, v) for d in shown}
    for b in xy:
        if b in cared_at_v:
            ax.annotate('', xy=(L - 0.42, 0.55), xytext=(xy[b][0] + 0.33, xy[b][1]),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.9, alpha=0.6))
    head = [head_literal_text(model, i) for i in rec['expl'] or []]
    ax.text(L + 1, 0.55, 'head explanation\n' + '\n'.join(textwrap.fill('• ' + t, 30) for t in head),
            ha='center', va='center', fontsize=6.6, color=INK,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=INK2, lw=1.2))
    ax.annotate('', xy=(L + 0.6, 0.55), xytext=(L + 0.42, 0.55), arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
    ax.text(L + 1.9, 0.55, f'"{class_name(rec["rule"], ds)}"', ha='center', va='center', fontsize=12,
            fontweight='bold', color=INK)
    ax.annotate('', xy=(L + 1.65, 0.55), xytext=(L + 1.4, 0.55), arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
    for l in range(L):
        ax.text(l, 1.06, f'conv layer {l}', ha='center', fontsize=9, color=layer_col[l % 3], fontweight='bold')
    ax.text(L, 1.06, 'prototype', ha='center', fontsize=9, color=BLUE, fontweight='bold')
    ax.text(L + 1, 1.06, 'head', ha='center', fontsize=9, color=INK, fontweight='bold')
    fig.suptitle('One prediction unfolded (rule model, exact). A box is a unit at a node: if 1, the fewest true '
                 'literals that make it fire (n_X = count over the node and its neighbours; L0 counts atoms);\n'
                 'if 0, its true literals fall short of the threshold. Arrows: a neighbour where the counted unit '
                 'holds. Shaded box: the node differs from the prototype on that bit.', x=0.01, ha='left',
                 fontsize=9.5)
    return fig
