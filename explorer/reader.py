"""The symbolic reading of one trained LogiX-GIN, as plain data for the explorer.

Same reading as interp_metrics.py (and through its functions): every conv literal is
``phi_in(neighbourhood count) >= 0.5``, every conv unit is ``W.lit + b >= 0``, prototypes
compare the binary node states, every head literal is ``phi_in(head input) >= 0.5`` and the
class is the argmax of the head margins ``W.lit + b``. A literal's truth set is exact
(min_covers.literal_true_intervals: ``phi_in`` is true where ``(w*c + b) mod 2*pi`` lies in
[0, pi]), so the step functions drawn by the app are the ones the rules use.

Two products:

    card()   what the model *is*, independent of any input: per conv layer the literals
             (what they count, their step function, the counts on which they are true),
             the unit weights and thresholds S = -b, each unit's firing rate, its shortest
             possible rule (greedy minimum cardinality, exact) and its number of minimal
             covers (min_covers.count_min_covers, an estimate on a 2048-step grid); the
             observed rules of layer-0 units (deeper units have ~1e9 covers, CLAUDE.md);
             the prototypes (required bits, cared bits, best-matching reference nodes and
             the class mix of the graphs they match exactly); the head literals and the
             class rules
    trace()  what the model *does* on one graph: per layer and node every count, literal,
             margin, binary unit and the network's own soft value, the shortest sufficient
             literals of every true unit, the prototype similarities and mismatches, the
             readout, the head literals, the class margins, the shortest sufficient
             explanation of the prediction and the network's prediction
"""
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from explain_proto import proto_dim_map, readout_columns
from interp_metrics import (base_literal_map, head_literal_map, level_of, readout_values, shortest_explanation,
                            symbolic_head, trunk_shape, trunk_states)
from min_covers import count_min_covers, literal_true_intervals, supported_min_covers, unique_rows
from unpack_rules import fmt_counts
from utils.datasets import spec
from utils.runs import EVAL_BATCH_SIZE
from utils.viz_proto import _layout, layer_literals

TOP_MATCHES = 6           # best-matching reference nodes / graphs listed per prototype
L0_RULES = 8              # observed rules listed per layer-0 unit


def bits(row):
    return ''.join('1' if b else '0' for b in np.asarray(row, dtype=bool).ravel())


def rnd(t, nd=3):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return np.round(a.astype(np.float64), nd).tolist()


def min_cardinality(w, S):
    """Fewest literals any rule of a unit needs: its heaviest weights until S is reached
    (weights >= 0, so greedy is exact). 0: always on; None: can never fire."""
    if S <= 0:
        return 0
    c = np.cumsum(np.sort(np.asarray(w, dtype=np.float64))[::-1])
    hit = np.flatnonzero(c >= S)
    return int(hit[0]) + 1 if len(hit) else None


def rule_stats(W, S):
    """Per unit: shortest possible rule and estimated number of minimal covers (None when
    the count overflows a float, e.g. on a head of ~1000 literals)."""
    W, S = np.asarray(W, dtype=np.float64), np.asarray(S, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        n = [float(count_min_covers(W[u], S[u]).sum()) for u in range(len(S))]
    return [min_cardinality(W[u], S[u]) for u in range(len(S))], [x if np.isfinite(x) else None for x in n]


class SymbolicReader:
    """One trained model (prototype node / graph / both, or classic GINTELL) read for display."""

    def __init__(self, model, ds_name, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.level = level_of(model)
        if self.level is None:
            raise ValueError(f'{type(model).__name__} is not a LogiX-GIN student')
        self.L, self.h = trunk_shape(model)
        self.n_feat = model.convs[0].nn[0].in_features // 2
        self.spec = spec(ds_name)
        self.dataset = ds_name
        self.features = self.spec.feature_names(self.n_feat)
        self.n_classes = model.fc.out_features
        self.classes = [self.spec.class_name(c) for c in range(self.n_classes)]
        self.W_head = model.fc.weight.detach().cpu().numpy()
        self.S_head = (-model.fc.b).detach().cpu().numpy()
        self.head_const = [self.literal_constant(i) for i in range(len(self.head_literals()))]

    # ------------------------------------------------------------------ structure

    @property
    def proto_layers(self):
        return [] if self.level == 'base' else list(self.model.proto_layers)

    def proto_kind(self, li):
        return 'node' if self.level == 'node' or (self.level == 'both' and li == 0) else 'graph'

    def readout_layout(self):
        """Per readout column: what it pools. Proto: {proto_layer, k, op}; base: {unit, op}."""
        if self.level == 'base':
            lits = base_literal_map(self.model)
            R = len(lits) // 2
            return [{'unit': int(u), 'op': op} for u, op, _ in lits[:R]]
        out = []
        for kind, k, op in readout_columns(self.model, self.level):
            out.append({'proto_layer': 1 if (self.level == 'both' and kind == 'graph') else 0, 'k': int(k), 'op': op})
        return out

    def head_literals(self):
        """Per head literal: (readout column, positive?)."""
        if self.level == 'base':
            R = len(self.readout_layout())
            return [(i % R, i < R) for i in range(2 * R)]
        return [(int(c), bool(p)) for c, p in self.model.head_columns()]

    def subject(self, l, i):
        """What conv literal i of layer l counts over the closed neighbourhood N[v]."""
        n = self.n_feat if l == 0 else self.h
        base, neg = i % n, i >= n
        name = self.features[base] if l == 0 else f'L{l - 1}u{base}'
        return ('¬' if neg else '') + name

    def node_x(self, data):
        if data.x is None:
            return torch.ones((data.num_nodes, self.n_feat), device=self.device)
        return data.x.float()

    # ------------------------------------------------------------------ forward

    @torch.no_grad()
    def run_batch(self, data):
        """Every quantity of the symbolic model (and the network's prediction) on a batch."""
        m = self.model
        data = data.to(self.device)
        x, ei, batch = self.node_x(data), data.edge_index, data.batch
        xs, lits = layer_literals(m, x, ei)
        layers = []
        for l, conv in enumerate(m.convs):
            ll = conv.nn[0]
            count, lit = lits[l]
            layers.append({'count': count, 'lit': lit, 'margin': lit.float() @ ll.weight.t() + ll.b, 'out': xs[l]})
        xs_net = trunk_states(m, x, ei, symbolic=False)
        s, s_node = readout_values(m, xs, batch)
        head_in = m.head_input(s) if hasattr(m, 'head_input') else torch.hstack([s, 1 - s])
        margin, head_lit, _ = symbolic_head(m, xs, batch)
        protos = []
        for li, p in enumerate(self.proto_layers):
            if self.proto_kind(li) == 'node':
                inp = torch.hstack(xs)
                sim = s_node if s_node is not None else p.similarity(inp)
            else:
                inp = m.graph_input(torch.hstack(xs), batch, symbolic=True)
                sim = p.similarity(inp)
            c, P = (p.care > 0.5).float(), (p.prototypes > 0.5).float()
            mism = inp @ (c * (1 - P)).t() + (1 - inp) @ (c * P).t()
            protos.append({'input': inp, 'sim': sim, 'mism': mism})
        return {'x': x, 'layers': layers, 'net_states': xs_net, 's': s, 'head_in': head_in, 'head_lit': head_lit,
                'margin': margin, 'protos': protos, 'prob': m(x, ei, batch),
                'head_only': symbolic_head(m, xs_net, batch)[0].argmax(1)}

    def explanation(self, c, head_lit_row):
        """Shortest sufficient explanation of class c (interp_metrics: the heaviest true head
        literals until S is reached), literal indices or None if the rule does not fire."""
        e = shortest_explanation(self.W_head[c], self.S_head[c], np.asarray(head_lit_row, dtype=bool))
        return None if e is None else [int(i) for i in e]

    def evidence(self, c, head_lit_row):
        """The same explanation in input-dependent literals only: the always-true literals
        hold on every graph, so they are counted in (a bias) and the fewest true non-constant
        literals reaching what remains of S are returned (greedy by weight: exact). None if
        the rule does not fire."""
        const_true = np.array([v is True for v in self.head_const])
        varying = np.array([v is None for v in self.head_const])
        need = self.S_head[c] - self.W_head[c][const_true].sum()
        e = shortest_explanation(self.W_head[c], need, np.asarray(head_lit_row, dtype=bool) & varying)
        return None if e is None else [int(i) for i in e]

    # ------------------------------------------------------------------ the model card

    def _conv_literal_texts(self, l, lo, hi):
        ll = self.model.convs[l].nn[0]
        w = ll.phi_in.w.detach().double().cpu().numpy()
        b = ll.phi_in.b.detach().double().cpu().numpy()
        out = []
        for i in range(len(w)):
            counts = np.arange(int(lo[i]), int(hi[i]) + 1)
            true = counts[np.remainder(w[i] * counts + b[i], 2 * np.pi) <= np.pi]
            cond = fmt_counts(true, int(lo[i]), int(hi[i]))
            subj = f'#{{n ∈ N[v] : {self.subject(l, i)}(n)}}'
            out.append({'true_counts': true.tolist(),
                        'text': f'{subj} {cond}' if cond is not None else f'{subj}: true on every observed count'})
        return out

    @torch.no_grad()
    def card(self, ref_dataset, ref_split='train', max_graphs=None):
        """The model card, with the observed ranges, rates and prototype matches measured on
        ``ref_dataset`` (the training split by default: the prototypes were fit on it)."""
        m = self.model
        ds = ref_dataset if max_graphs is None else ref_dataset[:max_graphs]
        loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)
        lo = [None] * self.L
        hi = [None] * self.L
        on = [0.0] * self.L
        l0_rows, n_nodes = [], 0
        head_lo = head_hi = None
        node_sims, graph_sims, node_where, graph_y = [], [], [], []
        g0 = 0
        for data in loader:
            st = self.run_batch(data)
            for l, lay in enumerate(st['layers']):
                c = lay['count']
                lo[l] = c.min(0).values if lo[l] is None else torch.minimum(lo[l], c.min(0).values)
                hi[l] = c.max(0).values if hi[l] is None else torch.maximum(hi[l], c.max(0).values)
                on[l] = on[l] + lay['out'].sum(0)
            l0_rows.append(st['layers'][0]['lit'].cpu().numpy())
            n_nodes += len(st['x'])
            hm, hM = st['head_in'].min(0).values, st['head_in'].max(0).values
            head_lo = hm if head_lo is None else torch.minimum(head_lo, hm)
            head_hi = hM if head_hi is None else torch.maximum(head_hi, hM)
            b = data.batch.cpu()
            local = torch.cat([torch.arange(int(k)) for k in torch.bincount(b, minlength=data.num_graphs)])
            node_where.append(torch.stack([b + g0, local], 1))
            graph_y.append(data.y.view(-1).cpu())
            for li, p in enumerate(st['protos']):
                (node_sims if self.proto_kind(li) == 'node' else graph_sims).append((li, p['sim'].cpu()))
            g0 += data.num_graphs
        node_where, graph_y = torch.cat(node_where), torch.cat(graph_y)

        conv = []
        stats = [rule_stats(c.nn[0].weight.detach().cpu().numpy(), (-c.nn[0].b).detach().cpu().numpy())
                 for c in m.convs]
        for l, c in enumerate(m.convs):
            ll = c.nn[0]
            W, S = ll.weight.detach().cpu().numpy(), (-ll.b).detach().cpu().numpy()
            lo_l, hi_l = (torch.floor(lo[l] + 1e-6).cpu().numpy(), torch.ceil(hi[l] - 1e-6).cpu().numpy())
            entry = {'l': l, 'n_in': self.n_feat if l == 0 else self.h,
                     'subjects': [self.subject(l, i) for i in range(ll.in_features)],
                     'phi_w': rnd(ll.phi_in.w, 5), 'phi_b': rnd(ll.phi_in.b, 5),
                     'phi_tau': getattr(ll.phi_in, 'tau', None) or 0,
                     'count_lo': lo_l.astype(int).tolist(), 'count_hi': hi_l.astype(int).tolist(),
                     'literals': self._conv_literal_texts(l, lo_l, hi_l),
                     'W': rnd(W, 4), 'S': rnd(S, 4), 'rate': rnd(on[l] / max(n_nodes, 1), 4),
                     'min_card': stats[l][0], 'n_covers': stats[l][1]}
            if l == 0:
                pats, mult = unique_rows(np.vstack(l0_rows))
                rules = {}
                for u in range(self.h):
                    found = supported_min_covers(W[u], S[u], pats, counts=mult)[:L0_RULES]
                    rules[u] = [{'literals': list(cv), 'support': sup / n_nodes} for cv, sup in found]
                entry['rules'] = rules
            conv.append(entry)

        protos = []
        for li, p in enumerate(self.proto_layers):
            kind = self.proto_kind(li)
            dmap = proto_dim_map(m, self.level, li)
            P = (p.prototypes.detach().cpu().numpy() > 0.5)
            care = (p.care.detach().cpu().numpy() > 0.5)
            labels = [(f'{op}:' if op else '') + f'L{l}u{u}' for op, l, u in dmap]
            sims = torch.cat([s for i, s in (node_sims if kind == 'node' else graph_sims) if i == li])
            top, exact = [], []
            for k in range(p.num_prototypes):
                col = sims[:, k]
                order = torch.argsort(col, descending=True, stable=True)[:TOP_MATCHES].tolist()
                if kind == 'node':
                    top.append([{'graph': int(node_where[r, 0]), 'node': int(node_where[r, 1]), 'sim': round(float(col[r]), 4)}
                                for r in order])
                    hit = torch.unique(node_where[col >= 1 - 1e-6, 0])
                else:
                    top.append([{'graph': int(r), 'sim': round(float(col[r]), 4)} for r in order])
                    hit = torch.nonzero(col >= 1 - 1e-6).flatten()
                mix = torch.bincount(graph_y[hit], minlength=self.n_classes).tolist() if len(hit) else [0] * self.n_classes
                exact.append({'graphs': int(len(hit)), 'class_mix': mix})
            protos.append({'kind': kind, 'K': p.num_prototypes, 'd': p.in_features, 'masked': bool(p.masked),
                           'T': float(p.mask_temp) if p.masked else None, 'bit_labels': labels,
                           'bit_unit': [l * self.h + u for _, l, u in dmap],
                           'bits': [bits(r) for r in P], 'care': [bits(r) for r in care],
                           'rule': [' ∧ '.join(('' if P[k, d] else '¬') + labels[d] for d in np.flatnonzero(care[k]))
                                    or 'TRUE (cares about no bit)' for k in range(p.num_prototypes)],
                           'top': top, 'exact': exact})

        head = self._head_card(head_lo.cpu().numpy(), head_hi.cpu().numpy())
        mc, nc = rule_stats(self.W_head, self.S_head)
        head.update({'W': rnd(self.W_head, 4), 'S': rnd(self.S_head, 4), 'min_card': mc, 'n_covers': nc})
        return {'level': self.level, 'model_class': type(m).__name__, 'dataset': self.dataset,
                'num_layers': self.L, 'hidden': self.h, 'features': self.features, 'classes': self.classes,
                'ground_truth': None if self.spec.ground_truth is None else
                {'classes': list(self.spec.ground_truth.classes), 'motif': self.spec.ground_truth.motif,
                 'shading': self.spec.ground_truth.shading},
                'ref_split': ref_split, 'ref_graphs': len(ds), 'ref_nodes': n_nodes,
                'conv': conv, 'protos': protos, 'readout': self.readout_layout(), 'head': head}

    def _bounded_col(self, c):
        """Does readout column c stay in [0, 1] (mean / max), so its literals read [0, 1]?"""
        if self.level == 'base':
            return self.readout_layout()[c]['op'] != 'sum'
        b = getattr(self.model, '_bounded', None)
        return b is None or bool(b[c])

    def _head_intervals(self, i, dlo=0.0, dhi=1.0):
        phi = self.model.fc.phi_in
        return literal_true_intervals(float(phi.w[i]), float(phi.b[i]), dlo, dhi)

    @staticmethod
    def _s_intervals(ivs, pos):
        """Intervals of a head input as intervals of the readout value s (negative: 1 - s)."""
        return ivs if pos else [(1 - c, 1 - a) for a, c in ivs][::-1]

    def _mismatch_counts(self, li, k, s_ivs):
        """Max over nodes of a masked node prototype's similarity is exp(-m*/T), m* the
        closest node's mismatches on the d cared bits: only d + 1 values occur. Returns the
        values of m* on which the literal is true, and d."""
        p = self.proto_layers[li]
        T, d = float(p.mask_temp), int((p.care[k] > 0.5).sum())
        return [m for m in range(d + 1) if any(a - 1e-9 <= np.exp(-m / T) <= c + 1e-9 for a, c in s_ivs)], d

    def _discrete_max(self, col):
        return (col.get('op') == 'max' and 'k' in col and self.proto_kind(col['proto_layer']) == 'node'
                and self.proto_layers[col['proto_layer']].masked)

    def literal_constant(self, i):
        """True / False when head literal i has that value on every input, else None. Exact
        for bounded columns (a sum-pooled count is never declared constant): the literal is
        true on all of [0, 1] or nowhere, its prototype cares about no bit (similarity 1
        everywhere), or it reads a masked prototype's max similarity and is true (or false)
        for every possible mismatch count."""
        c, pos = self.head_literals()[i]
        if not self._bounded_col(c):
            return None
        ivs = self._head_intervals(i)
        if not ivs:
            return False
        if len(ivs) == 1 and ivs[0][0] <= 1e-6 and ivs[0][1] >= 1 - 1e-6:
            return True
        col = self.readout_layout()[c]
        if 'k' not in col:
            return None
        p = self.proto_layers[col['proto_layer']]
        if not bool((p.care[col['k']] > 0.5).any()):
            v = 1.0 if pos else 0.0
            return any(a - 1e-9 <= v <= c2 + 1e-9 for a, c2 in ivs)
        if self._discrete_max(col):
            ms, d = self._mismatch_counts(col['proto_layer'], col['k'], self._s_intervals(ivs, pos))
            if not ms:
                return False
            if len(ms) == d + 1:
                return True
        return None

    def _head_card(self, lo, hi):
        """Head literals: what each one reads, its step function, where it is true and
        whether it is a constant (true or false on every input)."""
        phi = self.model.fc.phi_in
        w, b = phi.w.detach().double().cpu().numpy(), phi.b.detach().double().cpu().numpy()
        cols = self.readout_layout()
        lits = []
        for i, (c, pos) in enumerate(self.head_literals()):
            col = cols[c]
            bounded = self._bounded_col(c)
            dlo, dhi = (0.0, 1.0) if bounded else (float(lo[i]), float(hi[i]))
            ivs = literal_true_intervals(w[i], b[i], dlo, dhi)
            s_ivs = self._s_intervals(ivs, pos) if bounded else ivs
            what = (f"{col['op']}(L{col['unit'] // self.h}u{col['unit'] % self.h})" if self.level == 'base'
                    else f"{col['op']} sim(P{col['k']})" if col.get('proto_layer', 0) == 0 or self.level == 'graph'
                    else f"sim(G{col['k']})")
            const = self.head_const[i]
            if const is not None:
                nocare = 'k' in col and not bool((self.proto_layers[col['proto_layer']].care[col['k']] > 0.5).any())
                text = f"{what}: {'always' if const else 'never'} true" + (' (a bias)' if const else '') + \
                    (' · the prototype cares about no bit, similarity ≡ 1' if nocare else '')
            elif self._discrete_max(col):
                text = self._mismatch_text(col['k'], *self._mismatch_counts(col['proto_layer'], col['k'], s_ivs))
            else:
                span = ' ∪ '.join(f'[{a:.2f}, {c2:.2f}]' for a, c2 in s_ivs)
                text = f'{what} ∈ {span}' + ('' if bounded else f'  (observed {dlo:.2f}–{dhi:.2f})')
            lits.append({'col': c, 'pos': pos, 'what': what, 'text': text, 'constant': const, 'domain': [dlo, dhi],
                         'intervals': [list(v) for v in ivs], 's_intervals': [list(v) for v in s_ivs],
                         'phi_w': round(float(w[i]), 5), 'phi_b': round(float(b[i]), 5)})
        return {'literals': lits, 'phi_tau': getattr(phi, 'tau', None) or 0}

    @staticmethod
    def _mismatch_text(k, ms, d):
        """A condition on the closest node's mismatch count m* with prototype k."""
        if ms == list(range(ms[-1] + 1)):
            return f'some node matches P{k} exactly' if ms[-1] == 0 else \
                f'some node is within {ms[-1]} mismatch{"es" * (ms[-1] > 1)} of P{k}'
        if ms == list(range(ms[0], d + 1)):
            return f'no node matches P{k} exactly' if ms[0] == 1 else \
                f'every node has ≥ {ms[0]} mismatches with P{k}'
        return f'closest node to P{k} has m* {fmt_counts(ms, 0, d)} mismatches (of {d} cared bits)'

    # ------------------------------------------------------------------ one graph

    @torch.no_grad()
    def trace(self, data, split=None, index=None):
        """Everything the symbolic model computes on one graph (see the module docstring)."""
        b = Batch.from_data_list([data])
        st = self.run_batch(b)
        n = data.num_nodes
        ei = data.edge_index.cpu()
        x = st['x'].cpu()
        const = n > 0 and bool((x == x[0]).all())
        labels = [''] * n if const else [self.spec.feature_name(int(r.argmax())) for r in x]
        _, pos = _layout(ei, n)
        region = self.spec.region(x, ei)
        edges = sorted({(min(a, c), max(a, c)) for a, c in ei.t().tolist() if a != c})
        layers = []
        for l, (lay, net) in enumerate(zip(st['layers'], st['net_states'])):
            ll = self.model.convs[l].nn[0]
            W, S = ll.weight.detach().cpu().numpy(), (-ll.b).detach().cpu().numpy()
            lit, out = lay['lit'].cpu().numpy(), lay['out'].cpu().numpy() > 0.5
            why = []
            for v in range(n):
                row = []
                for u in range(self.h):
                    if not out[v, u]:
                        row.append(None)
                        continue
                    e = shortest_explanation(W[u], S[u], lit[v])
                    if e is None:            # float rounding at the threshold: every true literal
                        e = np.flatnonzero(lit[v])[np.argsort(-W[u][lit[v]], kind='stable')]
                    row.append([int(i) for i in e])
                why.append(row)
            layers.append({'count': np.rint(lay['count'].cpu().numpy()).astype(int).tolist(),
                           'lit': [bits(r) for r in lit], 'out': [bits(r) for r in out],
                           'margin': rnd(lay['margin'], 3), 'net': rnd(net, 3), 'why': why})
        protos = [{'kind': self.proto_kind(li), 'sim': rnd(p['sim'][0] if self.proto_kind(li) == 'graph' else p['sim'], 4),
                   'mism': rnd(p['mism'][0] if self.proto_kind(li) == 'graph' else p['mism'], 3),
                   'input': rnd(p['input'][0], 3) if self.proto_kind(li) == 'graph' else None}
                  for li, p in enumerate(st['protos'])]
        margin = st['margin'][0].cpu().numpy()
        c = int(margin.argmax())
        head_lit = st['head_lit'][0].cpu().numpy()
        expl = self.explanation(c, head_lit)
        evid = self.evidence(c, head_lit)
        prob = st['prob'][0].cpu()
        bit_agree = [float(((nt >= 0.5).float() == lay['out']).float().mean())
                     for nt, lay in zip(st['net_states'], st['layers'])]
        return {'split': split, 'i': index, 'y': int(data.y.view(-1)[0]) if data.y is not None else None,
                'n': n, 'labels': labels, 'x': [bits(r > 0.5) for r in x.numpy()], 'pos': [rnd(pos[v], 4) for v in range(n)], 'edges': edges,
                'region': None if region is None else [int(r) for r in region.tolist()],
                'layers': layers, 'protos': protos, 's': rnd(st['s'][0], 4), 'head_in': rnd(st['head_in'][0], 4),
                'head_lit': bits(head_lit), 'margin': rnd(margin, 4), 'rule': c, 'backed': expl is not None,
                'expl': expl, 'evidence': evid, 'net': {'prob': rnd(prob, 4), 'pred': int(prob.argmax()),
                                      'head_only': int(st['head_only'][0])},
                'bit_agreement': bit_agree}

    # ------------------------------------------------------------------ a whole split

    @torch.no_grad()
    def index(self, dataset):
        """One row per graph (label, predictions, explanation) and the split's summary:
        accuracy of the network and of the rules, fidelity, rule-backed share, and how
        often each head literal / prototype is cited by an explanation, per predicted class."""
        loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
        rows, agree, n_nodes = [], [], 0
        n_lit = len(self.head_literals())
        lit_cites = np.zeros((n_lit, self.n_classes), dtype=int)
        g0 = 0
        for data in loader:
            st = self.run_batch(data)
            prob = st['prob'].cpu().numpy()
            margin = st['margin'].cpu().numpy()
            head_lit = st['head_lit'].cpu().numpy()
            sizes = torch.bincount(data.batch.cpu(), minlength=data.num_graphs).tolist()
            ys = data.y.view(-1).cpu().tolist()
            for g in range(data.num_graphs):
                c = int(margin[g].argmax())
                e = self.explanation(c, head_lit[g])
                ev = self.evidence(c, head_lit[g])
                for i in e or []:
                    lit_cites[i, c] += 1
                rows.append({'i': g0 + g, 'y': ys[g], 'n': sizes[g], 'net': int(prob[g].argmax()),
                             'p': round(float(prob[g].max()), 3), 'rule': c, 'backed': e is not None,
                             'expl_len': None if e is None else len(e),
                             'evidence_len': None if ev is None else len(ev),
                             'head_only': int(st['head_only'][g])})
            for nt, lay in zip(st['net_states'], st['layers']):
                agree.append(float(((nt >= 0.5).float() == lay['out']).float().sum()))
            n_nodes += len(st['x'])
            g0 += data.num_graphs
        y = np.array([r['y'] for r in rows])
        net = np.array([r['net'] for r in rows])
        rule = np.array([r['rule'] for r in rows])
        head_only = np.array([r['head_only'] for r in rows])
        summary = {'graphs': len(rows), 'acc': float((net == y).mean()), 'logic_acc': float((rule == y).mean()),
                   'logic_fidelity': float((rule == net).mean()), 'head_fidelity': float((head_only == net).mean()),
                   'rule_backed': float(np.mean([r['backed'] for r in rows])),
                   'trunk_bit_agreement': float(sum(agree) / (n_nodes * self.L * self.h)),
                   'constant_head_literals': int(sum(v is not None for v in self.head_const))}
        for key in ('expl_len', 'evidence_len'):
            v = [r[key] for r in rows if r[key] is not None]
            summary[f'{key}_mean'] = float(np.mean(v)) if v else None
            summary[f'{key}_median'] = float(np.median(v)) if v else None
        proto_cites = []
        if self.level != 'base':
            cols, lits = self.readout_layout(), self.head_literals()
            for li, p in enumerate(self.proto_layers):
                pc = np.zeros((p.num_prototypes, self.n_classes), dtype=int)
                for i, (c, _) in enumerate(lits):
                    if cols[c]['proto_layer'] == li:
                        pc[cols[c]['k']] += lit_cites[i]
                proto_cites.append(pc.tolist())
        return {'rows': rows, 'summary': summary, 'literal_cites': lit_cites.tolist(), 'proto_cites': proto_cites}
