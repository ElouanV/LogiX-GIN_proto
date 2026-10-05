"""Extract the logical formula that each latent component of a LogiX-GIN represents.

This is a port of the rule-extraction procedure in ``nbs/LayerWiseRules.ipynb``,
which is the repository's canonical reading of the TELL layers. **The logic is
unchanged**: ``find_logic_rules``, ``extract_rules``, ``find_step_intervals`` and
``find_minimal_sets`` below are the notebook's functions, moved into a module,
documented, and given a driver. Only two things are adapted, both marked ADAPTED:
the readout half of ``forward_with_activations`` (the notebook targets GINTELL's
mean/max/sum readout, we also need the prototype readouts of ``models_proto``),
and the feature map that names head inputs.

------------------------------------------------------------------------------
Why the formula is exact rather than a post-hoc approximation
------------------------------------------------------------------------------

A conv layer is ``GINConv(Sequential(LogicalLayer(...)))`` built with ``eps=0``.
GINConv computes ``nn(aggregate)``, so the LogicalLayer never sees a node
embedding; it sees

    u_i = x_i + sum_{j in N(i)} x_j                        (closed neighbourhood)

where ``x`` has already been widened to ``[x, 1-x]``. Every coordinate of ``u``
is therefore a **count** over the closed neighbourhood: at layer 0 a count of
atoms of one type (or of atoms *not* of that type), at layer l a count of
neighbours satisfying one concept of layer l-1.

The LogicalLayer applies two stages to ``u``:

1. ``phi_in(u) = step(w*u + b)`` with ``w = exp(w_) > 0``. ``step`` is a partial
   Fourier series of a **square wave**, so it is periodic with period ``2*pi``:
   the literal is true when ``w*u + b`` falls in ``(0, pi)`` modulo ``2*pi``.
   In counts that is a **union of intervals**, not a threshold - which is what
   ``find_step_intervals`` recovers numerically. Reading ``t = -b/w`` as a lower
   threshold is wrong: it is only the first interval's lower edge.

2. ``o_j = sigmoid(phi_in(u) @ w_j + b_j)`` with ``w_j >= 0``
   (``weight = sigmoid(weight_sigma) * exp(weight_exp) * prune``). Because the
   weights are non-negative, stage 2 is **monotone** in the literals, so

       unit j fires  <=>  sum of w_j over the true literals  >=  -b_j

   and "which literal sets make unit j fire" is exactly a subset-sum problem.
   ``find_logic_rules`` enumerates those subsets. Negation is available despite
   non-negative weights because the input carries both ``x`` and ``1-x``.

Composing the two stages gives, for each latent component, a DNF over
neighbourhood-count conditions, whose literals are themselves components of the
layer below - so the procedure recurses down to atom counts at layer 0.

------------------------------------------------------------------------------
Support: the part that cannot be skipped
------------------------------------------------------------------------------

A subset whose weights clear the threshold is a *sufficient* condition, but
nothing guarantees any real graph satisfies it. ``find_logic_rules`` therefore
takes the observed literal activations and a ``min_support``, and prunes any
conjunction that fewer than ``min_support`` examples jointly satisfy.

This matters concretely. Extracting rules for this repository's trained node
model *without* support pruning yields conjunctions of 15-18 literals whose
measured joint support is 0.0000% - formally valid, satisfied by nothing, and
worthless as an explanation. Note that the notebook's own driver (cell 50)
passes ``activations`` only for the head and not for the conv layers; this
module passes them everywhere by default, which uses ``find_logic_rules``
exactly as written but at a call site the notebook did not exercise.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------

    # every unit of every conv layer, recursively expanded to atom counts
    python latent_logic.py --run_path results_proto/<...>/<seed> --all

    # one component
    python latent_logic.py --run_path ... --layer 2 --unit 17

    # library
    from latent_logic import collect_activations, explain_component, format_component
    acts = collect_activations(model, loader, device)
    expl = explain_component(model, layer=2, unit=17, acts=acts)
    print(format_component(expl))
"""
import argparse
import json
import os
import pickle

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from models_proto.tell import step
from utils.utils import get_dataset

# data/Mutagenicity/Mutagenicity/raw/Mutagenicity_label_readme.txt
ATOMS = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
CLASSES = ['mutagen', 'nonmutagen']


# =============================================================================
# PART 1 - ported verbatim from nbs/LayerWiseRules.ipynb (logic unchanged)
# =============================================================================

def inverse_sigmoid(x):
    """Computes the inverse of the sigmoid function (logit function)."""
    return torch.log(x / (1 - x))


@torch.no_grad()
def find_logic_rules(w, t_in, t_out, activations=None, max_rule_len=10,
                     max_rules=100, min_support=1):
    """Literal subsets whose weights reach ``t_out``, optionally data-supported.

    Notebook cell 9, unchanged. Because the layer's weights are non-negative and
    ``phi_in`` output is in [0,1], a set of literals is sufficient for the unit to
    fire exactly when their weights sum to at least ``t_out`` - hence a subset-sum
    enumeration, explored in descending weight order.

    Args:
        w:            [in_features] non-negative weights of ONE output unit.
        t_in:         phi_in thresholds. Cloned for fidelity with the notebook but
                      not used by the search - the interval reading of phi_in is
                      done separately by ``find_step_intervals``.
        t_out:        scalar the weight sum must reach (see ``extract_rules``).
        activations:  [n_examples x in_features] bool, the observed truth value of
                      each input literal. When given, a literal must hold on at
                      least ``min_support`` examples, and every partial
                      conjunction must hold jointly on at least that many.
        max_rule_len: give up on conjunctions longer than this.
        max_rules:    stop after this many rules.
        min_support:  minimum number of examples (see ``activations``).

    Returns:
        set of tuples of input-literal indices. Each tuple is a conjunction; the
        set is their disjunction.
    """
    w = w.clone()
    t_in = t_in.clone()
    t_out = t_out.clone()
    t_out = t_out.item()
    ordering_scores = w
    sorted_idxs = torch.argsort(ordering_scores, 0, descending=True)
    mask = w > 1e-5
    if activations is not None:
        mask = mask & (activations.sum(0) >= min_support)
    total_result = set()

    # Filter and sort indices based on the mask
    idxs_to_visit = sorted_idxs[mask[sorted_idxs]]
    if idxs_to_visit.numel() == 0:
        return total_result

    # Sort weights based on the filtered indices
    sorted_weights = w[idxs_to_visit]
    current_combination = []
    result = set()
    # Running joint-satisfaction mask of `current_combination`. The notebook recomputes
    # activations[:, comb + [i]].all(-1) from scratch for every candidate, which is
    # O(N*k) each; carrying the mask down and AND-ing one column is O(N) and returns
    # exactly the same values. Logic unchanged, arithmetic unchanged.
    base_mask = (torch.ones(activations.shape[0], dtype=torch.bool)
                 if activations is not None else None)

    def find_logic_rules_recursive(index, current_sum, mask=base_mask):
        # Stop if the maximum number of rules has been reached
        if len(result) >= max_rules:
            return

        if len(current_combination) > max_rule_len:
            return

        # Check if the current combination satisfies the condition
        if current_sum >= t_out:
            c = idxs_to_visit[current_combination].cpu().detach().tolist()
            c = tuple(sorted(c))
            result.add(c)
            return

        # Prune if remaining weights can't satisfy t_out
        remaining_max_sum = current_sum + sorted_weights[index:].sum()
        if remaining_max_sum < t_out:
            return

        # Explore further combinations
        for i in range(index, idxs_to_visit.shape[0]):
            # Prune based on activations if provided
            new_mask = mask
            if activations is not None:
                new_mask = mask & activations[:, idxs_to_visit[i]]
                if len(current_combination) > 0 and new_mask.sum().item() < min_support:
                    continue

            current_combination.append(i)
            find_logic_rules_recursive(i + 1, current_sum + sorted_weights[i], new_mask)
            current_combination.pop()

    # Start the recursive process
    find_logic_rules_recursive(0, 0)
    return result


def extract_rules(layer, feature=None, activations=None, max_rule_len=float('inf'),
                  max_rules=5, min_support=10, out_threshold=0.5):
    """DNF of one (or every) output unit of a ``LogicalLayer``.

    Notebook cell 9, unchanged except that the layer is passed explicitly instead
    of being monkey-patched onto the class.

    ``t_out = -b + logit(out_threshold)`` is the weight sum needed for the unit's
    sigmoid to exceed ``out_threshold``; at the default 0.5 the logit term is 0.

    Returns:
        list with one rule set per requested feature (see ``find_logic_rules``).
    """
    ws = layer.weight
    t_in = layer.phi_in.t
    t_out = -layer.b + inverse_sigmoid(torch.tensor(out_threshold))

    rules = []
    features = range(layer.out_features) if feature is None else [feature]
    for i in features:
        w = ws[i].to('cpu')
        ti = t_in.to('cpu')
        to = t_out[i].to('cpu')
        rules.append(find_logic_rules(w, ti, to, activations, max_rule_len,
                                      max_rules, min_support))
    return rules


def find_step_intervals(w, b, xmin, xmax, tau=5, resolution=1000):
    """Intervals of the input on which ``step(w*x + b) > 0.5``, scanned on [xmin, xmax].

    Notebook cell 48, unchanged. This is what turns a literal into a statement
    about counts. ``step`` is periodic, so the result is in general a union of
    intervals; scanning only the observed range keeps the answer to counts the
    data can actually produce.
    """
    # Sample x values
    xs = torch.linspace(xmin, xmax, resolution)
    wxb = w * xs + b
    ys = step(wxb, tau)

    intervals = []
    above = ys[0] > 0.5
    start = xs[0].item() if above else None

    for i in range(1, len(xs)):
        curr = ys[i] > 0.5
        if curr and not above:
            # Rising edge
            start = xs[i - 1].item()
        elif not curr and above:
            # Falling edge
            end = xs[i].item()
            intervals.append((start, end))
            start = None
        above = curr

    if above and start is not None:
        intervals.append((start, xs[-1].item()))

    return intervals


def find_minimal_sets(list_of_sets):
    """Drop any rule that another rule is a strict subset of. Notebook cell 36."""
    minimal_sets = []
    for i, s in enumerate(list_of_sets):
        if not any((set(other) < set(s)) or (s == other and i != j)
                   for j, other in enumerate(list_of_sets)):
            minimal_sets.append(s)
    return minimal_sets


# =============================================================================
# PART 2 - activations
# =============================================================================

def scatter_sum(x, edge_index):
    """Closed-neighbourhood sum: messages from neighbours, plus the node itself.

    Notebook cell 12, with torch_scatter replaced by index_add_ so the module has
    no dependency beyond torch (torch_scatter is not installed in the logix-gin
    environment). Identical arithmetic: sum over incoming edges, then ``+ x``,
    which is what GINConv computes for eps=0.
    """
    out = torch.zeros_like(x)
    out.index_add_(0, edge_index[1], x[edge_index[0]])
    return out + x


@torch.no_grad()
def forward_with_activations(model, x, edge_index, batch):
    """Run the model, recording for each layer what the rule extraction needs.

    Notebook cell 12. Per conv layer:
        x      the widened input [x, 1-x]
        x_sum  its closed-neighbourhood sum - the COUNT the LogicalLayer sees
        x_bin  phi_in(x_sum) >= 0.5 - the truth value of each input literal
        y      the layer's output concepts
        y_bin  y >= 0.5

    ADAPTED: the final entry is the head. The notebook hard-codes GINTELL's
    ``hstack([mean, max, sum])`` readout; here it goes through ``model.readout``
    so the prototype readouts of ``models_proto`` work too. The conv loop, which
    is what this module is about, is unchanged.
    """
    returns = []
    xs = []
    for conv in model.convs:
        ret = {}
        ret['x'] = torch.hstack([x, 1 - x])
        ret['x_sum'] = scatter_sum(ret['x'], edge_index)
        ret['x_bin'] = conv.nn[0].phi_in(ret['x_sum']) >= 0.5
        x = conv(torch.hstack([x, 1 - x]), edge_index)
        xs.append(x)
        ret['y'] = x
        ret['y_bin'] = x >= 0.5
        ret['batch'] = batch
        returns.append(ret)

    ret = {}
    if hasattr(model, 'readout'):            # ADAPTED: models_proto prototype readout
        s = model.readout(xs, batch)
    else:                                    # plain GINTELL (models/model.py)
        h = torch.hstack(xs)
        s = torch.hstack([global_mean_pool(h, batch), global_max_pool(h, batch),
                          global_add_pool(h, batch)])
    ret['x'] = model.head_input(s) if hasattr(model, 'head_input') else torch.hstack([s, 1 - s])
    ret['x_sum'] = ret['x']                  # no aggregation at the head
    ret['x_bin'] = model.fc.phi_in(ret['x']) >= 0.5
    out = model.fc(ret['x'])
    ret['y'] = out
    ret['y_bin'] = out >= 0.5
    ret['batch'] = batch
    returns.append(ret)
    return out, returns


@torch.no_grad()
def collect_activations(model, loader, device):
    """Concatenate ``forward_with_activations`` over a loader.

    Notebook cell 28, with the batch-offset bookkeeping kept. Returns a list with
    one dict per conv layer plus one for the head; tensors are on CPU because the
    support tests below index them heavily.
    """
    acc = None
    for data in loader:
        data = data.to(device)
        xf = data.x.float() if data.x is not None else torch.ones(
            (data.num_nodes, model.num_features), device=device)
        _, rets = forward_with_activations(model, xf, data.edge_index, data.batch)
        rets = [{k: v.cpu() for k, v in r.items()} for r in rets]
        if acc is None:
            acc = rets
            continue
        for l in range(len(rets)):
            for k in rets[l]:
                if k == 'batch':
                    rets[l][k] = rets[l][k] + acc[l][k].max() + 1
                    acc[l][k] = torch.cat([acc[l][k], rets[l][k]])
                else:
                    acc[l][k] = torch.vstack([acc[l][k], rets[l][k]])
    return acc


# =============================================================================
# PART 3 - naming the literals
# =============================================================================

def _fmt_intervals(intervals):
    return ' U '.join(f'[{a:.2f}, {b:.2f}]' for a, b in intervals) or 'empty'


def literal_name(model, layer, i, intervals=None):
    """Human-readable name of input literal ``i`` of conv ``layer``.

    The first half of a layer's inputs count a feature, the second half count its
    negation. At layer 0 the features are atom types; deeper they are the
    previous layer's units.
    """
    n = model.convs[layer].nn[0].in_features // 2
    base, neg = i % n, i >= n
    if layer == 0:
        sym = ATOMS[base] if base < len(ATOMS) else f'f{base}'
    else:
        sym = f'L{layer - 1}u{base}'
    name = f'#{"NOT " if neg else ""}{sym}'
    if intervals is not None:
        name += f' in {_fmt_intervals(intervals)}'
    return name


def head_feat_map(model, level, num_layers, hidden_dim):
    """Name each head input.

    ADAPTED from notebook cell 31. There the head consumes the GINTELL readout,
    so an input is ``(sign, readout_op, conv_layer, hidden_dim)``. A prototype
    head instead consumes per-prototype similarities, so an input is
    ``(sign, pool_op, prototype)``: a prototype is a pattern over ALL
    ``num_layers * hidden_dim`` concepts, not a single one, so it is explained by
    its own components rather than by one (layer, dim) pair.
    """
    if level is None:                                   # plain GINTELL: mean|max|sum
        base = [(op, l, d) for op in ('mean', 'max', 'sum')
                for l in range(num_layers) for d in range(hidden_dim)]
        return [(sign, *t) for sign in ('pos', 'neg') for t in base]
    if level == 'node':
        ops, K = model.pool_ops, model.num_prototypes
        base = [(op, k) for op in ops for k in range(K)]
    elif level == 'graph':
        base = [('pooled', k) for k in range(model.num_prototypes)]
    else:                                               # 'both'
        base = [(op, k) for op in model.node_pool_ops for k in range(model.num_prototypes)]
        base += [('pooled', k) for k in range(model.num_graph_prototypes)]
    return [(sign, *t) for sign in ('pos', 'neg') for t in base]


# =============================================================================
# PART 4 - driver: the formula of one latent component, expanded down the stack
# =============================================================================

def explain_head(model, level, acts, max_rule_len=3, max_rules=5, min_support=2):
    """The head's rule per class, over its own inputs. Notebook cells 30-33.

    For a prototype model an input is a (sign, pooling op, prototype) triple, so
    this says which prototypes each class reads - not which latent component. The
    prototypes themselves are patterns over all concepts; `explain_proto.py`
    relates a prototype to the components it keys on, and those components are
    what `explain_component` turns into formulas.
    """
    a = acts[-1]
    names = head_feat_map(model, level, model.num_layers, model.hidden_dim)
    rules = extract_rules(model.fc, activations=a['x_bin'], max_rule_len=max_rule_len,
                          max_rules=max_rules, min_support=min_support)
    out = {}
    for c, rs in enumerate(rules):
        refined = []
        for conj in rs:
            # drop literals true on every observed graph (notebook cell 32)
            term = tuple(i for i in conj if not a['x_bin'][:, i].all().item())
            refined.append(term if term else (True,))
        entries = []
        for term in find_minimal_sets(refined):
            if term == (True,):
                entries.append({'literals': [], 'support': 1.0, 'always_true': True})
                continue
            m = a['x_bin'][:, list(term)].all(-1)
            def _nm(i):
                t = names[i]
                neg = 'NOT ' if t[0] == 'neg' else ''
                if len(t) == 4:          # plain GINTELL: (sign, op, layer, dim)
                    return f'{neg}{t[1]}(L{t[2]}u{t[3]})'
                return f'{neg}p{t[2]}.{t[1]}'          # prototype: (sign, op, k)
            entries.append({
                'literals': [_nm(i) for i in term],
                'support': float(m.float().mean()),
                'always_true': False})
        out[CLASSES[c] if c < len(CLASSES) else f'class{c}'] = entries
    return out


def _unit_rules(model, layer, unit, acts, max_rule_len, max_rules, min_support,
                use_support):
    """Rules for one unit, with each literal named by its count intervals.

    Follows notebook cell 50: extract, drop literals that hold on every observed
    example (they say nothing), attach each surviving literal's intervals from the
    observed range of ``x_sum``, then keep the minimal rules.
    """
    ll = model.convs[layer].nn[0]
    a = acts[layer]
    phi = ll.phi_in
    tau = phi.tau if phi.tau is not None else 10
    rules = extract_rules(ll, feature=unit,
                          activations=a['x_bin'] if use_support else None,
                          max_rule_len=max_rule_len, max_rules=max_rules,
                          min_support=min_support)[0]

    refined, children = [], set()
    for conj in rules:
        term = []
        for i in conj:
            if a['x_bin'][:, i].all().item():
                continue                     # always true on the data: uninformative
            xs = a['x_sum'][:, i]
            iv = find_step_intervals(phi.w[i].cpu(), phi.b[i].cpu(),
                                     xs.min().cpu(), xs.max().cpu(), tau=tau)
            term.append((int(i), tuple(iv)))
            if layer != 0:
                children.add((layer - 1, int(i)))
        refined.append(tuple(term) if term else (True,))
    minimal = find_minimal_sets(refined)

    # empirical check: how often is each conjunction actually satisfied, and does
    # satisfying it really make the unit fire? Verification only - no logic changed.
    checked = []
    for term in minimal:
        if term == (True,):
            checked.append({'literals': [], 'support': 1.0, 'precision': None,
                            'always_true': True})
            continue
        idx = [i for i, _ in term]
        m = a['x_bin'][:, idx].all(-1)
        sup = float(m.float().mean())
        prec = float((a['y_bin'][m, unit]).float().mean()) if m.any() else None
        checked.append({'literals': [(i, iv, literal_name(model, layer, i, iv))
                                     for i, iv in term],
                        'support': sup, 'precision': prec, 'always_true': False})
    return checked, children


def explain_component(model, layer, unit, acts, max_rule_len=4, max_rules=5,
                      min_support=2, use_support=True, recursive=True,
                      max_depth=None):
    """The logical formula of latent component ``(layer, unit)``.

    Returns a dict::

        {'layer':l, 'unit':u, 'rules':[{literals, support, precision}, ...],
         'children': {(layer,unit): <same structure>, ...}}

    ``rules`` is a disjunction; each entry's ``literals`` is a conjunction of
    count conditions on the layer below. With ``recursive``, every referenced
    lower-layer unit is expanded too, so the formula bottoms out at atom counts.
    """
    out, seen = {}, set()
    queue = [(layer, unit, 0)]
    while queue:
        l, u, depth = queue.pop(0)
        n_out = model.convs[l].nn[0].out_features
        key = (l, u % n_out)
        if key in seen:
            continue
        seen.add(key)
        rules, children = _unit_rules(model, l, u % n_out, acts, max_rule_len,
                                      max_rules, min_support, use_support)
        out[key] = {'layer': l, 'unit': u % n_out, 'rules': rules}
        if recursive and (max_depth is None or depth < max_depth):
            for cl, ci in sorted(children):
                queue.append((cl, ci, depth + 1))
    root = (layer, unit)
    return {'root': root, 'components': out}


def format_component(expl, model=None):
    """Render ``explain_component`` output as indented text."""
    lines = []
    root = expl['root']
    for (l, u), c in sorted(expl['components'].items(),
                            key=lambda kv: (kv[0][0] != root[0], -kv[0][0], kv[0][1])):
        tag = f'L{l}u{u}' + ('   <- requested' if (l, u) == root else '')
        lines.append(f'{tag}')
        if not c['rules']:
            lines.append('    (no rule with the requested support and length)')
        for r in c['rules']:
            if r['always_true']:
                lines.append('    ALWAYS TRUE on the observed data')
                continue
            body = ' AND '.join(nm for _, _, nm in r['literals'])
            prec = '-' if r['precision'] is None else f"{r['precision']:.0%}"
            lines.append(f"    {body}")
            lines.append(f"      support {r['support']:.3%} · precision {prec}")
        lines.append('')
    return '\n'.join(lines)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='a seed directory holding best.pt')
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--dataset', default='Mutagenicity')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                    help="split the activations are measured on (the notebook uses val)")
    ap.add_argument('--layer', type=int, default=None)
    ap.add_argument('--unit', type=int, default=None)
    ap.add_argument('--all', action='store_true', help='every unit of every conv layer')
    ap.add_argument('--head', action='store_true',
                    help="also extract the head's rule per class")
    ap.add_argument('--max_rule_len', type=int, default=4)
    ap.add_argument('--max_rules', type=int, default=5)
    ap.add_argument('--min_support', type=int, default=2,
                    help='a rule must be jointly satisfied by at least this many examples. '
                         'Measured: at layer 0, dropping it leaves 45%% of rules satisfied '
                         'by nothing; 2 removes all of them and keeps more rules than 10.')
    ap.add_argument('--no_support', action='store_true',
                    help='extract without the joint-support prune (notebook cell 50 '
                         'does this for conv layers; expect unsatisfiable rules)')
    ap.add_argument('--no_recursive', action='store_true')
    ap.add_argument('--max_depth', type=int, default=None)
    ap.add_argument('--out', default=None, help='write JSON here (default: run_path/latent_logic.json)')
    a = ap.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = json.load(open(os.path.join(a.run_path, 'args.json')))
    model = torch.load(os.path.join(a.run_path, a.ckpt), map_location=device,
                       weights_only=False).eval()
    level = args.get('proto_level')

    dataset = get_dataset(a.dataset)
    split = pickle.load(open(os.path.join(a.run_path, 'data.pkl'), 'rb'))
    ds = dataset[split[f'{a.split}_indices']]
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    print(f'== {a.run_path}')
    print(f'== level={level} · {len(ds)} graphs from the {a.split} split · '
          f'{len(model.convs)} conv layers x {model.convs[0].nn[0].out_features} units')
    print('== literals are counts over the closed neighbourhood (self + neighbours); '
          'phi_in accepts a union of intervals, not a threshold\n')

    acts = collect_activations(model, loader, device)
    print(f'activations collected: {acts[0]["x_bin"].shape[0]} nodes\n')

    if a.head:
        print('--- head: which prototypes each class reads ---')
        for cls, entries in explain_head(model, level, acts, max_rule_len=a.max_rule_len,
                                         max_rules=a.max_rules,
                                         min_support=a.min_support).items():
            print(f'  {cls}:')
            if not entries:
                print('    (no rule with the requested support and length)')
            for e in entries:
                if e['always_true']:
                    print('    ALWAYS TRUE (default class)')
                else:
                    print(f"    {' AND '.join(e['literals'])}"
                          f"   [support {e['support']:.2%}]")
        print()

    targets = []
    if a.all:
        for l in range(len(model.convs)):
            targets += [(l, u) for u in range(model.convs[l].nn[0].out_features)]
    elif a.layer is not None and a.unit is not None:
        targets = [(a.layer, a.unit)]
    elif not a.head:
        ap.error('pass --all, --head, or both --layer and --unit')

    results = {}
    for l, u in targets:
        expl = explain_component(model, l, u, acts,
                                 max_rule_len=a.max_rule_len, max_rules=a.max_rules,
                                 min_support=a.min_support,
                                 use_support=not a.no_support,
                                 recursive=not a.no_recursive,
                                 max_depth=a.max_depth)
        results[f'L{l}u{u}'] = {f'L{k[0]}u{k[1]}': {
            'rules': [{'literals': [nm for _, _, nm in r['literals']],
                       'support': r['support'], 'precision': r['precision'],
                       'always_true': r['always_true']} for r in v['rules']]}
            for k, v in expl['components'].items()}
        if a.all:
            # print as we go: a full sweep at max_rule_len 10 can take hours, and a
            # summary held to the end would leave it invisible
            own = expl['components'][(l, u)]['rules']
            own = [r for r in own if not r['always_true']]
            name = f'L{l}u{u}'
            if not own:
                alw = any(r['always_true'] for r in expl['components'][(l, u)]['rules'])
                print(f'{name:<10} {"ALWAYS TRUE" if alw else "no supported rule"}',
                      flush=True)
            else:
                r = own[0]
                prec = 0 if r['precision'] is None else r['precision']
                print(f'{name:<10} {len(r["literals"]):2d} lit · support '
                      f'{r["support"]:.3%} · precision {prec:.0%}  '
                      f'{" AND ".join(nm for _, _, nm in r["literals"])}', flush=True)
        else:
            print(format_component(expl, model))

    if a.head:
        results['head'] = explain_head(model, level, acts, max_rule_len=a.max_rule_len,
                                       max_rules=a.max_rules, min_support=a.min_support)

    out = a.out or os.path.join(a.run_path, 'latent_logic.json')
    with open(out, 'w') as f:
        json.dump({'level': level, 'split': a.split,
                   'min_support': a.min_support, 'max_rule_len': a.max_rule_len,
                   'components': results}, f, indent=1)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
