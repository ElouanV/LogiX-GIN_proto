"""Extract every rule of a trained LogiX-GIN (base or node-level prototype) and write it as readable predicates.

For each LogicalLayer unit the complete rule set is its minimal covers
(min_covers.py): the unit fires iff at least one listed conjunction holds. Every
literal is then unpacked into what it states about the graph:

    conv layer 0   #{n in N[v] : atom(n) = C} in {1, 2}
    conv layer l   #{n in N[v] : L(l-1)u5(n)} >= 2          (L(l-1)u5 defined below)
    head (proto)   max over nodes of sim(n, P3) in [0.71, 1.00]
    head (base)    mean over nodes of L2u7 in [0.10, 0.35]

N[v] is the closed neighbourhood (v and its neighbours): GINConv with eps=0 sums it.
The count sets come from the closed form of phi_in (a literal is true iff
(w*count + b) mod 2*pi is in [0, pi]), evaluated on the integer counts observed on
the data. A negated unit NOT u is needed wherever a higher layer reads "#{n: NOT u(n)}";
its rules are the minimal sets of FALSE literals whose weight exceeds W - S (W = sum
of the unit's weights), i.e. the same enumeration on the dual threshold.

Three reductions keep the output readable (rules.md) and small (rules.json):

1. Observed rules only. Rules no example satisfies are formally valid but say nothing
   about the data. When a unit has more than EXHAUSTIVE_LIMIT formal rules, each
   observed example is explained instead by its shortest sufficient rule (its
   heaviest true literals until the threshold is reached - provably the fewest
   literals any rule can use for that example).
2. Covering set. From the observed rules, a greedy set cover keeps the fewest rules
   that together hold on every example where the unit is on (resp. off) and some
   observed rule holds. The other rules are counted, and stored with --all_rules.
3. Data simplification (--simplify_tol). Each kept rule is also shortened by dropping
   literals as long as its precision on the data does not fall by more than the
   tolerance: literals implied by the others on real graphs (degree proxies, a count
   and its complement...) disappear. The simplified rule is grounded in the data, not
   formally: the exact minimal rule stays next to it in rules.json.

Units computing the same function on the data (identical observed rule sets) are
reported once ("identical to L0u7").

    python unpack_rules.py --run_path "<results_{proto,logic}/.../<seed>[/sparse/<cfg>]>"
    -> <run_path>/rules/rules.md (readable) and rules.json (complete)
"""
import argparse
import json
import math
import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from latent_logic import ATOMS, collect_activations
from min_covers import count_min_covers, literal_true_intervals, supported_min_covers
from rule_eval import logical_layers
from utils.utils import BBBP_FEATURES, get_dataset

TWO_PI = 2 * math.pi
EXHAUSTIVE_LIMIT = 5e6      # formal rules per unit above which the shortest rule per example is used
POOL = 20000                # most supported observed rules offered to the covering-set selection
MAX_STORED = 20000          # rules per unit written with --all_rules


def dataset_from_path(run_path):
    parts = os.path.normpath(run_path).split(os.sep)
    for stage in ('results_proto', 'results_logic'):
        if stage in parts:
            return parts[parts.index(stage) + 1]
    raise ValueError(f'{run_path} is neither under results_proto/ nor results_logic/')


def feature_names(ds_name, n):
    names = {'Mutagenicity': ATOMS, 'BBBP': BBBP_FEATURES}.get(ds_name, [])
    return [names[i] if i < len(names) else f'f{i}' for i in range(n)]


# ----------------------------------------------------------------------------- literals

def fmt_counts(true_set, lo, hi):
    """Integer set within [lo, hi] as a short statement; None if it is all of [lo, hi]."""
    s = sorted(int(c) for c in true_set)
    if not s:
        return 'never (on observed counts)'
    if len(s) == hi - lo + 1:
        return None
    runs, start, prev = [], s[0], s[0]
    for c in s[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        if c is not None:
            start = prev = c
    if len(runs) == 1:
        a, b = runs[0]
        if b == hi:
            return f'>= {a}'
        if a == lo:
            return f'<= {b}'
        return f'in [{a}, {b}]' if b > a else f'= {a}'
    parts = [f'{a}' if a == b else (f'>={a}' if b == hi else f'{a}-{b}') for a, b in runs]
    return 'in {' + ', '.join(parts) + '}'


class ConvLiterals:
    """Names and truth sets of the input literals of conv layer l."""

    def __init__(self, model, l, acts, feat_names):
        ll = model.convs[l].nn[0]
        self.l, self.n = l, ll.in_features // 2
        w = ll.phi_in.w.detach().double().cpu().numpy()
        b = ll.phi_in.b.detach().double().cpu().numpy()
        xs = acts[l]['x_sum'].double().numpy()
        self.lo = np.floor(xs.min(0) + 1e-6).astype(int)
        self.hi = np.ceil(xs.max(0) - 1e-6).astype(int)
        self.true_sets = []
        for i in range(2 * self.n):
            c = np.arange(self.lo[i], self.hi[i] + 1)
            self.true_sets.append(set(c[np.remainder(w[i] * c + b[i], TWO_PI) <= math.pi].tolist()))
        self.feat_names = feat_names

    def subject(self, i):
        base, neg = i % self.n, i >= self.n
        if self.l == 0:
            return f'atom(n) {"!=" if neg else "="} {self.feat_names[base]}'
        return f'{"NOT " if neg else ""}L{self.l - 1}u{base}(n)'

    def text(self, i, value=True):
        """The literal (value=True) or its negation (value=False) as a statement."""
        full = set(range(self.lo[i], self.hi[i] + 1))
        s = self.true_sets[i] if value else full - self.true_sets[i]
        cond = fmt_counts(s, self.lo[i], self.hi[i])
        return None if cond is None else f'#{{n in N[v] : {self.subject(i)}}} {cond}'

    def child(self, i):
        """(layer, unit, polarity) of the lower-layer predicate this literal counts."""
        if self.l == 0:
            return None
        return (self.l - 1, i % self.n, i < self.n)


class _HeadBase:
    """Head inputs [s, 1-s]: a literal reads s (or 1-s) through phi_in's intervals."""

    def __init__(self, model, acts):
        ll = model.fc
        self.n = ll.in_features // 2
        self.w = ll.phi_in.w.detach().double().cpu().numpy()
        self.b = ll.phi_in.b.detach().double().cpu().numpy()
        x = acts[-1]['x'].double().numpy()
        self.lo, self.hi = x.min(0), x.max(0)

    def _intervals(self, i, value):
        ivs = literal_true_intervals(self.w[i], self.b[i], self.lo[i], self.hi[i])
        if not value:                                   # complement within the observed range
            edges, cur = [], self.lo[i]
            for a, c in ivs:
                if a > cur:
                    edges.append((cur, a))
                cur = c
            if cur < self.hi[i]:
                edges.append((cur, self.hi[i]))
            ivs = edges
        if i >= self.n:                                 # the literal reads 1 - s
            ivs = [(1 - c, 1 - a) for a, c in ivs][::-1]
        return ivs

    def _fmt(self, name, ivs, integer=False, lo=None, hi=None):
        if not ivs:
            return f'{name}: never'
        if integer:                                     # a sum over nodes is a count
            lo_i, hi_i = math.ceil(lo - 1e-6), math.floor(hi + 1e-6)
            counts = {k for a, c in ivs for k in range(math.ceil(a - 1e-9), math.floor(c + 1e-9) + 1)
                      if lo_i <= k <= hi_i}
            if counts:
                cond = fmt_counts(counts, lo_i, hi_i)
                return None if cond is None else f'{name} {cond}'
            # the unit's outputs are soft, so its summed "count" can sit between integers
        return f'{name} in ' + ' U '.join(f'[{a:.2f}, {c:.2f}]' for a, c in ivs)


class ProtoHeadLiterals(_HeadBase):
    """Node-level prototype head: s = [mean_k sim(n, P_k) | max_k sim(n, P_k)]."""

    def __init__(self, model, acts):
        super().__init__(model, acts)
        self.K, self.ops = model.num_prototypes, model.pool_ops

    def proto(self, i):
        return (i % self.n) % self.K

    def child(self, i):
        return None

    def text(self, i, value=True):
        base = i % self.n
        name = f'{self.ops[base // self.K]} over nodes of sim(n, P{base % self.K})'
        return self._fmt(name, self._intervals(i, value))


class ConceptHeadLiterals(_HeadBase):
    """Base LogiX-GIN head: s = [mean(h) | max(h) | sum(h)], h = every conv unit of every layer."""
    OPS = ('mean', 'max', 'sum')

    def __init__(self, model, acts):
        super().__init__(model, acts)
        self.h = model.convs[0].nn[0].out_features
        self.C = len(model.convs) * self.h

    def child(self, i):
        l, u = divmod((i % self.n) % self.C, self.h)
        return (l, u, i < self.n)

    def text(self, i, value=True):
        base = i % self.n
        op = self.OPS[base // self.C]
        l, u = divmod(base % self.C, self.h)
        if op == 'sum':
            # observed range of the count itself (the literal may read 1 - count)
            j = base if i < self.n else i - self.n
            name = f'#{{nodes n : L{l}u{u}(n)}}'
            return self._fmt(name, self._intervals(i, value), integer=True, lo=self.lo[j], hi=self.hi[j])
        return self._fmt(f'{op} over nodes of L{l}u{u}', self._intervals(i, value))


# ----------------------------------------------------------------------------- one unit

def patterns(X):
    """Distinct rows of a bool matrix and, for every row, the index of its pattern."""
    X = np.ascontiguousarray(X, dtype=bool)
    if X.shape[1] == 0:
        return X[:1], np.zeros(len(X), dtype=np.int64)
    packed = np.ascontiguousarray(np.packbits(X, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    return X[first], inv.ravel()


def shortest_per_pattern(w, T, full):
    """For every pattern whose true literals reach T, its shortest sufficient rule:
    the heaviest true literals until the threshold is crossed (a minimal cover, and
    the fewest literals any rule contained in the pattern can have)."""
    out = set()
    order = np.argsort(-w, kind='stable')
    for p in full:
        idx = order[p[order] & (w[order] > 0)]
        c = np.cumsum(w[idx])
        if len(c) and c[-1] >= T:
            out.add(tuple(int(i) for i in idx[:int(np.searchsorted(c, T)) + 1]))
    return list(out)


def greedy_cover(M, hits):
    """Fewest rows of M (rules x patterns) whose union holds all the target mass it can.
    Classic greedy set cover, gains updated incrementally."""
    gains = M.astype(np.float64) @ hits
    covered = np.zeros(M.shape[1], dtype=bool)
    chosen = []
    while len(gains):
        i = int(np.argmax(gains))
        if gains[i] <= 1e-9:
            break
        chosen.append(i)
        new = M[i] & ~covered
        covered |= new
        gains -= M[:, new].astype(np.float64) @ hits[new]
    return chosen


def simplify(lits, full, mult, hits, tol):
    """Drop literals while the rule's precision on the data stays >= its own minus tol;
    at each step drop the one whose removal gains the most support."""
    lits = list(lits)
    m = full[:, lits].all(1)
    p0 = hits[m].sum() / mult[m].sum()
    while len(lits) > 1:
        best = None
        for j in lits:
            rest = [k for k in lits if k != j]
            m2 = full[:, rest].all(1)
            s2 = mult[m2].sum()
            if hits[m2].sum() / s2 >= p0 - tol and (best is None or s2 > best[1]):
                best = (j, s2)
        if best is None:
            break
        lits.remove(best[0])
    return lits


def unit_rules(W_u, S_u, X, fire, lits, polarity, max_show, simplify_tol=0.0, all_rules=False):
    """Rules of one unit (polarity=True) or of its negation (polarity=False); see module doc."""
    if polarity:
        T, Xl = S_u, X                                   # literals that must be TRUE
    else:                                                # NOT u: FALSE literals weighing > W - S
        T = float(W_u.sum()) - S_u
        T += 1e-9 * max(1.0, abs(T))
        Xl = ~X
    target = fire if polarity else ~fire
    # everything below runs on the distinct literal patterns, not on every node
    cols = np.flatnonzero(W_u > 0)
    rows, inv = patterns(Xl[:, cols])
    mult = np.bincount(inv, minlength=len(rows)).astype(np.float64)
    hits = np.bincount(inv, weights=target, minlength=len(rows))
    full = np.zeros((len(rows), Xl.shape[1]), dtype=bool)
    full[:, cols] = rows
    total = float(count_min_covers(W_u, T).sum())
    if total <= EXHAUSTIVE_LIMIT:
        covers, mode = [c for c, _ in supported_min_covers(W_u, T, full, 1, counts=mult.astype(np.int64))], 'all observed'
    else:
        covers, mode = shortest_per_pattern(W_u, T, full), 'shortest per example'

    masks = [full[:, list(c)].all(1) if c else np.ones(len(rows), dtype=bool) for c in covers]
    sup = np.array([mult[m].sum() for m in masks])
    covered_any = np.zeros(len(rows), dtype=bool)
    for m in masks:
        covered_any |= m
    order = np.argsort(-sup, kind='stable')
    pool = order[:POOL]
    chosen = pool[greedy_cover(np.array([masks[i] for i in pool]).reshape(len(pool), len(rows)), hits)] \
        if len(pool) else []

    def describe(c, m):
        r = {'literals': [int(i) for i in c], 'support': float(mult[m].sum() / len(X)),
             'precision': float(hits[m].sum() / mult[m].sum())}
        if simplify_tol is not None and c:
            s = simplify(c, full, mult, hits, simplify_tol)
            ms = full[:, s].all(1)
            r.update(simple=[int(i) for i in s], simple_support=float(mult[ms].sum() / len(X)),
                     simple_precision=float(hits[ms].sum() / mult[ms].sum()))
        return r

    kept = [describe(covers[i], masks[i]) for i in chosen]
    if simplify_tol is not None:
        # simplified rules can coincide, or contain one another; a rule whose simplified
        # literals include another kept rule's holds only where that one does -> merged
        kept.sort(key=lambda r: (len(r.get('simple', r['literals'])), -r.get('simple_support', r['support'])))
        merged, keys = [], []
        for r in kept:
            k = frozenset(r.get('simple', r['literals']))
            if any(k2 <= k for k2 in keys):
                continue
            keys.append(k)
            merged.append(r)
        kept = merged
    kept.sort(key=lambda r: -r.get('simple_support', r['support']))
    for r in kept[:max_show]:                    # readable forms for the markdown
        r['text'] = [t for t in (lits.text(i, polarity) for i in r['literals']) if t is not None]
        if 'simple' in r:
            r['simple_text'] = [t for t in (lits.text(i, polarity) for i in r['simple']) if t is not None]
    out = {
        'polarity': 'pos' if polarity else 'neg', 'mode': mode,
        'n_rules_formal_est': total, 'n_rules_observed': len(covers), 'n_rules': len(kept),
        'rate': float(target.mean()),
        'coverage': float(hits[covered_any].sum() / max(target.sum(), 1)),
        'rules': kept, 'max_show': max_show,
        # identical observed rule sets = identical function on the data
        'signature': hash(frozenset(frozenset(c) for c in covers)),
    }
    if all_rules:
        out['all_rules'] = [{'literals': [int(i) for i in covers[i]], 'support': float(sup[i] / len(X))}
                            for i in order[:MAX_STORED]]
    return out


# ----------------------------------------------------------------------------- driver

@torch.no_grad()
def extract(model, acts, ds_name, max_show=15, simplify_tol=0.0, all_rules=False):
    L = len(model.convs)
    lls = logical_layers(model)
    proto = type(model).__name__ == 'GINTELLProtoNode'
    out = {'kind': 'proto-node' if proto else 'base', 'layers': {}, 'head': {}, 'prototypes': {}}
    feat = feature_names(ds_name, model.convs[0].nn[0].in_features // 2)
    kw = dict(max_show=max_show, simplify_tol=simplify_tol, all_rules=all_rules)

    # head first: it decides which prototypes / units matter
    hl = ProtoHeadLiterals(model, acts) if proto else ConceptHeadLiterals(model, acts)
    Wh = model.fc.weight.detach().double().cpu().numpy()
    Sh = (-model.fc.b).detach().double().cpu().numpy()
    Xh, Yh = acts[-1]['x_bin'].numpy(), acts[-1]['y_bin'].numpy()
    need_neg, used_protos = set(), set()
    for c in range(Wh.shape[0]):
        r = unit_rules(Wh[c], Sh[c], Xh, Yh[:, c], hl, True, **kw)
        for rule in r['rules']:
            for i in rule['literals']:
                if proto:
                    used_protos.add(hl.proto(i))
                elif not hl.child(i)[2]:
                    need_neg.add(hl.child(i)[:2])
        out['head'][c] = r
    out['head_literals'] = {i: {'true': hl.text(i, True), 'false': hl.text(i, False)} for i in range(2 * hl.n)}

    if proto:
        informative = {l: (lambda r: (r > 0) & (r < 1))(acts[l]['y_bin'].numpy().mean(0)) for l in range(L)}
        P = model.proto.prototypes.detach().cpu().numpy() > 0.5          # [K, L*h]
        care = model.proto.care.detach().cpu().numpy() > 0.5              # all True without a mask
        h = model.hidden_dim
        for k in sorted(used_protos):
            on = [(j // h, j % h) for j in np.flatnonzero(P[k] & care[k]) if informative[j // h][j % h]]
            off = [(j // h, j % h) for j in np.flatnonzero(~P[k] & care[k]) if informative[j // h][j % h]]
            out['prototypes'][k] = {'on': on, 'off': off, 'cared_bits': int(care[k].sum())}
            if model.proto.masked:          # sim = exp(-mismatches / T): a threshold on sim is one on mismatches
                out['prototypes'][k]['temperature'] = float(model.proto.mask_temp)
            need_neg |= set(off)

    # conv layers top-down; a negated unit is expanded only where something above reads it
    for l in reversed(range(L)):
        lits = ConvLiterals(model, l, acts, feat)
        W = lls[l].weight.detach().double().cpu().numpy()
        S = (-lls[l].b).detach().double().cpu().numpy()
        X, Y = acts[l]['x_bin'].numpy(), acts[l]['y_bin'].numpy()
        layer, seen = {}, {}
        for u in range(W.shape[0]):
            entry = {'dead': bool(W[u].sum() < S[u]), 'constant': bool(S[u] <= 0),
                     'fire_rate': float(Y[:, u].mean())}
            if not (entry['dead'] or entry['constant']):
                entry['pos'] = unit_rules(W[u], S[u], X, Y[:, u], lits, True, **kw)
                if (l, u) in need_neg:
                    entry['neg'] = unit_rules(W[u], S[u], X, Y[:, u], lits, False, **kw)
                sig = entry['pos'].pop('signature')
                if 'neg' in entry:
                    entry['neg'].pop('signature')
                if sig in seen and entry['pos']['n_rules_observed']:
                    entry['same_as'] = seen[sig]
                else:
                    seen[sig] = u
                for key in ('pos', 'neg'):
                    for rule in entry.get(key, {}).get('rules', []):
                        for i in set(rule['literals']) | set(rule.get('simple', [])):
                            ch = lits.child(i)
                            # an input from the 1-x half counts nodes where the child is OFF,
                            # whether the rule reads that count as in or out of its set
                            if ch is not None and not ch[2]:
                                need_neg.add(ch[:2])
            layer[u] = entry
        out['layers'][l] = layer
        out.setdefault('literals', {})[l] = {i: {'true': lits.text(i, True), 'false': lits.text(i, False)}
                                             for i in range(2 * lits.n)}
    for r in out['head'].values():
        r.pop('signature', None)
    return out


# ----------------------------------------------------------------------------- formula size

def _shown(rule):
    """The literals of a rule as displayed (simplified when available)."""
    return rule.get('simple', rule['literals'])


def _entry(ex, l, u):
    e = ex['layers'][l][u]
    return ex['layers'][l][e['same_as']] if 'same_as' in e else e


def _head_children(ex, model, i):
    """Conv units a head literal reads: [] for prototype similarities."""
    if ex['kind'] == 'proto-node':
        return []
    n, h = model.fc.in_features // 2, model.convs[0].nn[0].out_features
    l, u = divmod((i % n) % (len(model.convs) * h), h)
    return [(l, u, i < n)]


def formula_size(ex, model, acts):
    """How much one has to read to follow the class rules down to atoms.

    global          distinct units and rules reachable from the class rules, through
                    each unit's kept rules and the lower units their literals count
                    (a NOT unit counts as its own rule set).
    per prediction  for one graph: the class rule that holds, then at every node the
                    first kept rule explaining each needed unit's actual value there.
                    A count literal ("#{nodes n : L2u5(n)} in {2-4}") needs the unit's
                    value at every node of the graph, so all of them are included.
    """
    L = len(model.convs)
    h = {l: (model.convs[l].nn[0].in_features // 2) for l in range(L)}
    start = set()
    for r in ex['head'].values():
        for rule in r['rules']:
            for i in _shown(rule):
                start |= set(_head_children(ex, model, i))
    for p in ex['prototypes'].values():
        start |= {(l, u, True) for l, u in p['on']} | {(l, u, False) for l, u in p['off']}
    seen, frontier, units, rules = set(), list(start), {l: set() for l in range(L)}, {l: 0 for l in range(L)}
    while frontier:
        l, u, pol = frontier.pop()
        if (l, u, pol) in seen:
            continue
        seen.add((l, u, pol))
        r = _entry(ex, l, u).get('pos' if pol else 'neg')
        if r is None:
            continue
        units[l].add(u)
        rules[l] += r['n_rules']
        if l:
            frontier += [(l - 1, i % h[l], i < h[l]) for rule in r['rules'] for i in _shown(rule)]
    head_rules = sum(r['n_rules'] for r in ex['head'].values())
    out = {'global': {'class_rules': head_rules,
                      'units': {f'L{l}': len(units[l]) for l in range(L)},
                      'rules': {f'L{l}': rules[l] for l in range(L)},
                      'total_rules': head_rules + sum(rules.values())}}

    if ex['kind'] != 'base':
        return out
    # per prediction
    def first(rules_, row, pol):
        for k, rule in enumerate(rules_):
            v = row[_shown(rule)]
            if (v.all() if pol else (~v).all()):
                return k
        return None
    batch = acts[0]['batch'].numpy()
    pred = acts[-1]['y'].numpy().argmax(1)
    Xh = acts[-1]['x_bin'].numpy()
    XY = [(acts[l]['x_bin'].numpy(), acts[l]['y_bin'].numpy()) for l in range(L)]
    sizes, n_unexplained = [], 0
    for g in range(len(Xh)):
        head = ex['head'][pred[g]]['rules']
        k = first(head, Xh[g], True)
        if k is None:
            n_unexplained += 1
            continue
        nodes = np.flatnonzero(batch == g)
        needed = {l: set() for l in range(L)}
        for i in _shown(head[k]):
            for l, u, _ in _head_children(ex, model, i):
                needed[l].add(u)
        used = set()
        for l in reversed(range(L)):
            X, Y = XY[l]
            for u in sorted(needed[l]):
                e = _entry(ex, l, u)
                for v in nodes:
                    pol = bool(Y[v, u])
                    r = e.get('pos' if pol else 'neg')
                    kk = first(r['rules'], X[v], pol) if r else None
                    if kk is None:
                        continue
                    used.add((l, u, pol, kk))
                    if l:
                        needed[l - 1] |= {i % h[l] for i in _shown(r['rules'][kk])}
        sizes.append((len(used), sum(len(needed[l]) for l in range(L))))
    s = np.array(sizes) if sizes else np.zeros((1, 2))
    out['per_prediction'] = {'rules_median': float(np.median(s[:, 0])), 'rules_p90': float(np.quantile(s[:, 0], 0.9)),
                             'units_median': float(np.median(s[:, 1])),
                             'graphs_without_class_rule': n_unexplained, 'graphs': int(len(Xh))}
    return out


# ----------------------------------------------------------------------------- markdown

def md_rules(r, indent=''):
    on = 'on' if r['polarity'] == 'pos' else 'off'
    obs = (f'{r["n_rules_observed"]} observed' if r['mode'] == 'all observed'
           else f'{r["n_rules_observed"]} shortest rules of observed examples')
    lines = [f'{indent}{r["n_rules"]} rule(s) cover every case explained by the {obs} '
             f'(~{r["n_rules_formal_est"]:.3g} formally valid); they hold on {r["coverage"]:.1%} '
             f'of the cases where the unit is {on}']
    for rule in r['rules'][:r['max_show']]:
        stat = f'support {rule["support"]:.1%} · precision {rule["precision"]:.0%}'
        if 'simple_text' in rule and len(rule['simple']) < len(rule['literals']):
            body = ' AND '.join(rule['simple_text']) or 'always'
            stat = (f'support {rule["simple_support"]:.1%} · precision {rule["simple_precision"]:.0%} · '
                    f'simplified on data from {len(rule["text"])} literals')
        else:
            body = ' AND '.join(rule['text']) if rule['text'] else 'always (every literal holds on all data)'
        lines.append(f'{indent}- {body}  `{stat}`')
    if r['n_rules'] > r['max_show']:
        lines.append(f'{indent}- ... {r["n_rules"] - r["max_show"]} more in rules.json')
    if r['n_rules'] == 0 and r['rate'] > 0:
        lines.append(f'{indent}- none: no example satisfies a rule, yet the network '
                     f'{"fires" if r["polarity"] == "pos" else "stays off"} on {r["rate"]:.1%} of them. It gets '
                     'there only through soft literal values (e.g. "false" literals at 0.03 with large '
                     'weights), which the logic reading cannot see.')
    return lines


def to_markdown(ex, title, meta, classes):
    L = len(ex['layers'])
    proto = ex['kind'] == 'proto-node'
    chain = ('class -> prototype similarities -> prototypes (patterns over L0..L2 units) -> ' if proto
             else 'class -> mean / max / sum over nodes of L0..L2 units -> ')
    md = [f'# {title}', '', meta, '']
    fs = ex.get('formula_size')
    if fs:
        g = fs['global']
        md += ['## Formula size', '',
               f'To follow every class rule down to atoms: **{g["total_rules"]} rules** '
               f'({g["class_rules"]} class rules, ' + ', '.join(
                   f'{g["rules"][k]} rules over {g["units"][k]} units at {k}' for k in sorted(g['units'], reverse=True))
               + ').']
        if 'per_prediction' in fs:
            p = fs['per_prediction']
            md.append(f'One prediction needs a median of **{p["rules_median"]:.0f} rules** '
                      f'(90th percentile {p["rules_p90"]:.0f}) over {p["units_median"]:.0f} units; '
                      f'{p["graphs_without_class_rule"]} of {p["graphs"]} graphs have no class rule for their '
                      'predicted class (no class unit fires, argmax decides).')
        md.append('')
    md += ['## How to read', '',
          '- `N[v]`: the node and its neighbours. `#{n in N[v] : P(n)} >= 2`: at least 2 of them satisfy P.',
          '- A unit fires iff **at least one** of its rules holds (OR of ANDs). `NOT LxuY` is the unit not firing; '
          'its own rules are listed when something above uses it.',
          '- Only rules that occur on the data are considered, and of those only a covering set: the fewest '
          'rules that together explain every case the observed rules explain.',
          '- `simplified on data`: literals were dropped because, on the data, the rule is as precise without '
          'them (e.g. a degree proxy implied by the other literals). The exact minimal rule is in rules.json.',
          '- Count conditions are evaluated on the counts observed on the validation set; literals true on '
          'every node are left out of the text.',
          '- `support`: share of validation nodes (graphs, for the classes) where the rule holds. '
          '`precision`: share of those where the network\'s unit really fires (y >= 0.5); below 100% means '
          'soft (not exactly 0/1) literal values move the weighted sum.',
          f'- Chain: {chain}L2 units -> L1 units -> L0 units -> atoms.',
          '- rules.json: kept rules as literal indices (`layers.<l>.<unit>.pos|neg.rules[].literals`, '
          'simplified in `.simple`); literal i reads `literals.<l>.<i>.true` (`.false` in NOT rules), '
          '`head_literals` for the classes.', '']
    md += ['## 1. Classes', '']
    for c, r in ex['head'].items():
        md.append(f'### {classes[c] if c < len(classes) else f"class {c}"} '
                  f'(predicted on {r["rate"]:.1%} of graphs)')
        md += md_rules(r) + ['']
    sec = 2
    if proto:
        md += [f'## {sec}. Prototypes used by the class rules', '',
               'Similarity = fraction of the concepts on which a node agrees with the prototype. '
               'Listed: the informative concepts (units that fire on some nodes but not all) the prototype '
               'requires ON / OFF.', '']
        if any('temperature' in p for p in ex['prototypes'].values()):
            md += ['Masked prototypes: sim(n, P) = exp(-m / T), m = number of cared concepts on which node n '
                   'disagrees with P, so sim = 1 iff n satisfies the whole conjunction below and '
                   'sim >= x iff m <= -T ln x.', '']
        for k, p in sorted(ex['prototypes'].items()):
            on = ', '.join(f'L{l}u{u}' for l, u in p['on']) or '-'
            off = ', '.join(f'L{l}u{u}' for l, u in p['off']) or '-'
            if 'temperature' in p:
                conj = ' AND '.join([f'L{l}u{u}' for l, u in p['on']] + [f'NOT L{l}u{u}' for l, u in p['off']]) or 'TRUE'
                md += [f'**P{k}** ({p["cared_bits"]} cared bits, T = {p["temperature"]:.3g}): {conj}', '']
            else:
                md += [f'**P{k}** ON: {on}', '', f'OFF: {off}', '']
        sec += 1
    for l in reversed(range(L)):
        layer = ex['layers'][l]
        dead = [u for u, e in layer.items() if e['dead']]
        const = [u for u, e in layer.items() if e['constant']]
        dup = {u: e['same_as'] for u, e in layer.items() if 'same_as' in e}
        md += [f'## {sec + (L - 1 - l)}. Layer {l} units', '',
               f'Never fire (pruned below threshold): {", ".join(f"L{l}u{u}" for u in dead) or "none"}. '
               f'Always fire: {", ".join(f"L{l}u{u}" for u in const) or "none"}. '
               f'Duplicates: {", ".join(f"L{l}u{u} = L{l}u{v}" for u, v in dup.items()) or "none"}.', '']
        for u, e in layer.items():
            if 'pos' not in e or 'same_as' in e:
                continue
            md.append(f'### L{l}u{u}  (fires on {e["fire_rate"]:.1%} of nodes)')
            md += md_rules(e['pos'])
            if 'neg' in e:
                md += ['', f'**NOT L{l}u{u}** (off on {1 - e["fire_rate"]:.1%} of nodes)']
                md += md_rules(e['neg'])
            md.append('')
    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='directory holding best.pt, args.json, data.pkl')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--max_show', type=int, default=15, help='rules per unit in rules.md')
    ap.add_argument('--simplify_tol', type=float, default=0.0,
                    help='max precision loss allowed when dropping literals on the data; negative disables')
    ap.add_argument('--all_rules', action='store_true',
                    help=f'also store every observed rule (up to {MAX_STORED} per unit) in rules.json')
    ap.add_argument('--out', default=None, help='default: <run_path>/rules')
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    ds_name = dataset_from_path(a.run_path)
    model = torch.load(os.path.join(a.run_path, 'best.pt'), map_location='cpu', weights_only=False).eval()
    if type(model).__name__ not in ('GINTELLProtoNode', 'GINTELL'):
        raise SystemExit(f'{type(model).__name__}: only base LogiX-GIN (GINTELL) and node-level prototype '
                         'models (GINTELLProtoNode) are supported')
    split = pickle.load(open(os.path.join(a.run_path, 'data.pkl'), 'rb'))
    ds = get_dataset(ds_name)[split[f'{a.split}_indices']]
    acts = collect_activations(model, DataLoader(ds, batch_size=256), torch.device('cpu'))
    print(f'{ds_name}: {len(ds)} {a.split} graphs, {acts[0]["x_bin"].shape[0]} nodes')

    tol = None if a.simplify_tol < 0 else a.simplify_tol
    ex = extract(model, acts, ds_name, a.max_show, tol, a.all_rules)
    ex['formula_size'] = formula_size(ex, model, acts)
    out = a.out or os.path.join(a.run_path, 'rules')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'rules.json'), 'w') as f:
        json.dump(ex, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    meta = f'Rules measured on the {a.split} split. Run: `{a.run_path}`'
    sj = os.path.join(a.run_path, 'sparsity.json')
    if os.path.exists(sj):
        m = json.load(open(sj))['after']['metrics']
        meta = f'Test accuracy {m["test_acc"]:.3f} · AUC {m.get("test_auc", float("nan")):.3f} · ' + meta
    classes = ['mutagen', 'nonmutagen'] if ds_name == 'Mutagenicity' else []
    kind = 'prototype' if ex['kind'] == 'proto-node' else 'base'
    with open(os.path.join(out, 'rules.md'), 'w') as f:
        f.write(to_markdown(ex, f'Rules of the {kind} LogiX-GIN model — {ds_name}', meta, classes))

    conv = [e['pos'] for layer in ex['layers'].values() for e in layer.values() if 'pos' in e]
    lens = [len(r['literals']) for u in conv for r in u['rules']]
    slens = [len(r.get('simple', r['literals'])) for u in conv for r in u['rules']]
    fs = ex['formula_size']
    print(f'formula size: {fs["global"]["total_rules"]} rules to read {fs["global"]["units"]}'
          + (f', {fs["per_prediction"]["rules_median"]:.0f} per prediction (median)' if 'per_prediction' in fs else ''))
    print(f'conv units: {sum(u["n_rules_observed"] for u in conv)} observed rules -> {len(lens)} kept '
          f'(median length {np.median(lens) if lens else 0:.0f}, simplified {np.median(slens) if slens else 0:.0f}); '
          f'class rules kept: {[r["n_rules"] for r in ex["head"].values()]} -> {out}/rules.md, rules.json '
          f'({os.path.getsize(os.path.join(out, "rules.json")) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
