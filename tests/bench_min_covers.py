"""Benchmark min_covers.py: speed, scaling, parallelism, and fidelity on a trained model.

    ~/miniconda3/envs/logix-gin/bin/python tests/bench_min_covers.py            # everything
    ~/miniconda3/envs/logix-gin/bin/python tests/bench_min_covers.py --quick    # ~1 min
    ... --sections synthetic parallel     # skip the trained-model section
    ... --run_path results_proto/<config>/<inner>/<seed>  --out bench.json

Sections
    synthetic  enumeration time vs n against brute force and latent_logic.find_logic_rules,
               covers/s, worst gap between two outputs (the delay), count-DP accuracy.
    parallel   extract_layer wall time for 1..8 processes, and what the results cost to ship.
    real       a trained LogiX-GIN: how many minimal covers each unit has, legacy vs new on
               the production calls, and how faithful the rules are to the network.

"Fidelity" separates the two ways a rule set can disagree with the network:
    binarization  the complete DNF equals the threshold on *binarized* literals
                  (x_bin = phi_in >= 0.5); the network sums the soft phi_in values.
    truncation    a length- or support-capped rule set covers only part of the DNF.
"""
import argparse
import glob
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from min_covers import (count_min_covers, extract_layer, iter_min_covers,  # noqa: E402
                        supported_min_covers, unique_rows)


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


def brute(w, S):
    n, out = len(w), set()
    for k in range(n + 1):
        for c in itertools.combinations(range(n), k):
            s = w[list(c)].sum()
            if s >= S and all(s - w[i] < S for i in c):
                out.add(frozenset(c))
    return out


def max_delay(w, S, eps=0.0):
    """Longest wall-clock gap between consecutive outputs (includes the first)."""
    last, worst, k = time.perf_counter(), 0.0, 0
    for _ in iter_min_covers(w, S, eps=eps):
        now = time.perf_counter()
        worst, last, k = max(worst, now - last), now, k + 1
    return worst, k


def legacy():
    try:
        import torch
        from latent_logic import find_logic_rules
        return torch, find_logic_rules
    except Exception as e:
        print(f'  (legacy find_logic_rules unavailable: {e})')
        return None, None


# ----------------------------------------------------------------------------- synthetic

def bench_synthetic(quick):
    torch, old = legacy()
    ns = [12, 16, 20, 24] if quick else [12, 16, 20, 24, 28, 32]
    rows = []
    print(f'\n{"regime":<9}{"n":>4}{"covers":>10}{"count est":>11}{"count ms":>10}{"new ms":>10}'
          f'{"covers/s":>11}{"max gap us":>11}{"legacy ms":>11}{"speedup":>9}  check')
    for regime in ('uniform', 'trained'):
        rng = np.random.default_rng(0)
        for n in ns:
            w = rng.random(n) if regime == 'uniform' else rng.exponential(1.0, n) ** 2
            S = 0.5 * w.sum()
            est, t_cnt = timed(count_min_covers, w, S)
            if est.sum() > 5e6:                           # would not fit in memory as tuples
                print(f'{regime:<9}{n:>4}{"-":>10}{est.sum():>11.2e}{t_cnt * 1e3:>10.1f}'
                      f'   enumeration skipped (~{est.sum() / 6e5:.0f} s at 6e5 covers/s, too big for memory)')
                rows.append(dict(regime=regime, n=n, count_est=float(est.sum()), count_s=t_cnt))
                continue
            new, t_new = timed(lambda: list(iter_min_covers(w, S)))
            gap, _ = max_delay(w, S)
            check, t_old = '', None
            if n <= 16:
                check = 'brute ok' if set(map(frozenset, new)) == brute(w, S) else 'BRUTE MISMATCH'
            if old is not None and len(new) <= 20000:
                res, t_old = timed(old, torch.tensor(w), torch.zeros(n), torch.tensor(S),
                                   None, float('inf'), float('inf'))
                same = {frozenset(c) for c in res} == set(map(frozenset, new))
                check += (' ' if check else '') + ('legacy ok' if same else 'LEGACY MISMATCH')
            r = dict(regime=regime, n=n, covers=len(new), count_est=float(est.sum()),
                     count_s=t_cnt, new_s=t_new, max_gap_s=gap, legacy_s=t_old, check=check)
            rows.append(r)
            sp = f'{t_old / t_new:8.0f}x' if t_old else f'{"-":>9}'
            lg = f'{t_old * 1e3:11.1f}' if t_old else f'{"skip":>11}'
            print(f'{regime:<9}{n:>4}{len(new):>10}{est.sum():>11.0f}{t_cnt * 1e3:>10.1f}'
                  f'{t_new * 1e3:>10.1f}{len(new) / max(t_new, 1e-9):>11.2e}{gap * 1e6:>11.0f}'
                  f'{lg}{sp}  {check}')
    return rows


# ----------------------------------------------------------------------------- parallel

def bench_parallel(quick):
    rng = np.random.default_rng(1)
    units, n = (16, 24) if quick else (32, 30)
    W = rng.exponential(1.0, (units, n)) ** 2
    S = 0.5 * W.sum(1)
    total = sum(count_min_covers(W[u], S[u]).sum() for u in range(units))
    print(f'\nsynthetic layer: {units} units x {n} inputs, ~{total:.2e} covers in total')
    rows, base, serial = [], None, None
    for jobs in (1, 2, 4, 8):
        res, t = timed(extract_layer, W, S, n_jobs=jobs)
        base, serial = base or t, serial or res
        rows.append(dict(n_jobs=jobs, seconds=t, speedup=base / t))
        print(f'  n_jobs={jobs}: {t:6.2f}s  speedup {base / t:4.2f}x')
    blob, t_pk = timed(pickle.dumps, serial)
    per = [r['seconds'] for r in serial]              # uncontended, from the 1-process run
    print(f'  results pickle: {len(blob) / 1e6:.1f} MB in {t_pk:.2f}s (paid again on unpickle); '
          f'slowest unit {max(per):.2f}s of {sum(per):.2f}s total -> best possible speedup '
          f'{sum(per) / max(per):.1f}x')
    return dict(units=units, n=n, total_covers=float(total), runs=rows,
                pickle_mb=len(blob) / 1e6, pickle_s=t_pk, slowest_unit_s=max(per),
                sum_unit_s=sum(per))


# ----------------------------------------------------------------------------- real model

def load_run(run_path, split):
    import torch
    from torch_geometric.loader import DataLoader
    from latent_logic import collect_activations
    from utils.utils import get_dataset
    torch.set_grad_enabled(False)
    if run_path is None:
        hits = sorted(glob.glob('results_proto/Mutagenicity/*proto_level=node*/*/0/best.pt'))
        if not hits:
            return None, None
        run_path = os.path.dirname(hits[0])
    model = torch.load(os.path.join(run_path, 'best.pt'), map_location='cpu',
                       weights_only=False).eval()
    ds = get_dataset(json.load(open(os.path.join(run_path, 'args.json'))).get('dataset', 'Mutagenicity'))
    idx = pickle.load(open(os.path.join(run_path, 'data.pkl'), 'rb'))[f'{split}_indices']
    acts = collect_activations(model, DataLoader(ds[idx], batch_size=256), torch.device('cpu'))
    print(f'\nrun: {run_path}\n{split} split: {acts[0]["x_bin"].shape[0]} nodes')
    return model, acts


def greedy_shortest(w, S, true_mask):
    """Fewest literals of this example that already reach S (heaviest first); 0 if none."""
    ws = np.sort(w[true_mask])[::-1]
    c = np.cumsum(ws)
    return int(np.searchsorted(c, S) + 1) if len(c) and c[-1] >= S else 0


def bench_real(run_path, split, quick):
    model, acts = load_run(run_path, split)
    if model is None:
        print('\nno trained run found; pass --run_path')
        return None
    torch, old = legacy()
    layers = [c.nn[0] for c in model.convs] + [model.fc]
    out = {'layers': []}

    print(f'\n{"layer":<6}{"n":>5}{"eps":>7}{"covers/unit: min":>18}{"median":>10}{"max":>10}'
          f'{"count s":>9}{"thr=net":>9}{"shortest/node med":>19}')
    for li, ll in enumerate(layers):
        W = ll.weight.detach().double().numpy()
        S = (-ll.b).detach().double().numpy()
        X = acts[li]['x_bin'].numpy()
        net = acts[li]['y_bin'].numpy()
        thr = X.astype(np.float64) @ W.T >= S
        agree = float((thr == net).mean())               # complete DNF vs network
        fired = np.argwhere(thr)
        sub = fired[np.random.default_rng(0).choice(len(fired), min(len(fired), 3000), replace=False)]
        short = [greedy_shortest(W[u], S[u], X[i]) for i, u in sub] if len(sub) else [0]
        for eps in ((0.0,) if quick else (0.0, 1e-3)):
            cnt, t = timed(lambda: np.array([count_min_covers(W[u], S[u], eps=eps).sum()
                                             for u in range(W.shape[0])]))
            q = np.quantile(cnt, [0, 0.5, 1])
            print(f'{("L" + str(li)) if li < len(layers) - 1 else "head":<6}{W.shape[1]:>5}{eps:>7g}'
                  f'{q[0]:>18.2e}{q[1]:>10.2e}{q[2]:>10.2e}{t:>9.2f}{agree:>9.4f}'
                  f'{np.median(short):>19.0f}')
            out['layers'].append(dict(layer=li, n=W.shape[1], eps=eps, covers_min=q[0],
                                      covers_median=q[1], covers_max=q[2], count_s=t,
                                      threshold_vs_network=agree,
                                      shortest_per_node_median=float(np.median(short))))

    # layer 0: the only conv layer whose full DNF is small enough to enumerate
    ll = layers[0]
    W = ll.weight.detach().double().numpy()
    S = (-ll.b).detach().double().numpy()
    X = acts[0]['x_bin'].numpy()
    net = acts[0]['y_bin'].numpy()
    print('\nlayer 0, all units')
    new_pure, t_new = timed(lambda: [list(iter_min_covers(W[u], S[u], eps=1e-5)) for u in range(len(S))])
    line = f'  pure weights     new {t_new:7.3f}s  ({sum(map(len, new_pure))} covers)'
    res = dict(pure_new_s=t_new, pure_covers=sum(map(len, new_pure)))
    if old is not None:
        o, t_old = timed(lambda: [old(ll.weight[u].detach().double(), ll.phi_in.t, -ll.b[u].detach().double(),
                                      None, float('inf'), float('inf')) for u in range(len(S))])
        same = all({frozenset(c) for c in a} == set(map(frozenset, b)) for a, b in zip(o, new_pure))
        line += f'   legacy {t_old:7.3f}s   x{t_old / t_new:.0f}   same={same}'
        res.update(pure_legacy_s=t_old, pure_same=same)
    print(line)

    k, L = 2, 4                                          # latent_logic defaults

    def run_new():
        rows, mult = unique_rows(X)                      # shared by every unit of the layer
        return [supported_min_covers(W[u], S[u], rows, k, L, eps=1e-5, counts=mult)
                for u in range(len(S))]
    new_sup, t_new = timed(run_new)
    line = f'  support>={k}, len<={L}  new {t_new:7.3f}s  ({sum(map(len, new_sup))} rules)'
    res.update(sup_new_s=t_new, sup_rules=sum(map(len, new_sup)))
    if old is not None:
        Xt = acts[0]['x_bin']
        o, t_old = timed(lambda: [old(ll.weight[u].detach().double(), ll.phi_in.t, -ll.b[u].detach().double(),
                                      Xt, L, float('inf'), k) for u in range(len(S))])
        same = all({frozenset(c) for c in a} == {frozenset(c) for c, _ in b} for a, b in zip(o, new_sup))
        line += f'   legacy {t_old:7.3f}s   x{t_old / t_new:.0f}   same={same}'
        res.update(sup_legacy_s=t_old, sup_same=same)
    print(line)

    # fidelity of the supported, length-capped rule set against the network
    hold = np.zeros_like(net)
    for u, rules in enumerate(new_sup):
        for c, _ in rules:
            hold[:, u] |= X[:, list(c)].all(1)
    thr = X.astype(float) @ W.T >= S
    tp = (hold & net).sum()
    res.update(rules_precision=float(tp / max(hold.sum(), 1)),
               rules_recall=float(tp / max(net.sum(), 1)),
               rules_recall_vs_threshold=float((hold & thr).sum() / max(thr.sum(), 1)),
               threshold_vs_network=float((thr == net).mean()))
    print(f'  rule set vs network:   precision {res["rules_precision"]:.4f}  recall {res["rules_recall"]:.4f}')
    print(f'  truncation only:       recall vs complete DNF {res["rules_recall_vs_threshold"]:.4f}'
          f'   (rules never fire outside it)')
    print(f'  binarization only:     complete DNF vs network agreement {res["threshold_vs_network"]:.4f}')
    out['layer0'] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sections', nargs='+', default=['synthetic', 'parallel', 'real'],
                    choices=['synthetic', 'parallel', 'real'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--run_path', default=None)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--out', default=None, help='write the numbers here as JSON')
    a = ap.parse_args()
    report = {}
    if 'synthetic' in a.sections:
        report['synthetic'] = bench_synthetic(a.quick)
    if 'parallel' in a.sections:
        report['parallel'] = bench_parallel(a.quick)
    if 'real' in a.sections:
        report['real'] = bench_real(a.run_path, a.split, a.quick)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1, default=float)
        print(f'\nwritten: {a.out}')


if __name__ == '__main__':
    main()
