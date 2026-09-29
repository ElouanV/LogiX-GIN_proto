"""Rule statistics of a trained LogiX-GIN: how many rules each unit has and how faithful they are.

Used by sparsify_proto.py (before/after pruning) and train_proto.py (final evaluation,
logged to MLflow), so the size of the extracted explanation is tracked next to the
accuracy of every run.
"""
import numpy as np
import torch

from latent_logic import collect_activations
from min_covers import count_min_covers


def logical_layers(model):
    """The conv LogicalLayers in order, then the head."""
    return [c.nn[0] for c in model.convs] + [model.fc]


@torch.no_grad()
def rule_stats(model, loader, device, max_nodes=3000):
    """Per LogicalLayer: how many literals and rules a unit has, and how faithful they are.

    nonzero/unit      weights > 1e-3 per output unit (median)
    covers/unit       number of minimal covers = rules per unit (median, max), counted
                      by the DP of min_covers.count_min_covers (exact once pruned)
    shortest/node     fewest true literals that already reach the threshold, over the
                      (node, unit) pairs that fire (median) - the shortest explanation
    dead / constant   units that can never fire / always fire
    thr=net           agreement of the binarized-literal threshold (the complete rule
                      set) with the network's own output >= 0.5
    """
    model.eval()
    acts = collect_activations(model, loader, device)
    out = []
    rng = np.random.default_rng(0)
    for li, ll in enumerate(logical_layers(model)):
        W = ll.weight.detach().double().cpu().numpy()
        S = (-ll.b).detach().double().cpu().numpy()
        X = acts[li]['x_bin'].numpy()
        net = acts[li]['y_bin'].numpy()
        thr = X.astype(np.float64) @ W.T >= S
        covers = np.array([count_min_covers(W[u], S[u]).sum() for u in range(len(S))])
        fired = np.argwhere(thr)
        fired = fired[rng.choice(len(fired), min(len(fired), max_nodes), replace=False)]
        short = []
        for i, u in fired:
            c = np.cumsum(np.sort(W[u][X[i]])[::-1])
            short.append(int(np.searchsorted(c, S[u]) + 1))
        out.append({
            'layer': 'head' if li == len(model.convs) else f'L{li}',
            'n_in': int(W.shape[1]),
            'nonzero_per_unit_median': float(np.median((W > 1e-3).sum(1))),
            'covers_per_unit_median': float(np.median(covers)),
            'covers_per_unit_max': float(covers.max()),
            'shortest_per_node_median': float(np.median(short)) if short else None,
            'dead_units': int((W.sum(1) < S).sum()),
            'constant_units': int((S <= 0).sum()),
            'threshold_vs_network': float((thr == net).mean()),
        })
    return out


def rule_metrics(stats, prefix='rules/'):
    """Flatten rule_stats for metric logging: ``rules/L1/covers_per_unit_median`` etc.

    Rule counts span 1e1..1e14, so they are also logged as log10 to plot on one axis.
    """
    m = {}
    for s in stats:
        for k, v in s.items():
            if k in ('layer', 'n_in') or v is None:
                continue
            m[f'{prefix}{s["layer"]}/{k}'] = v
            if k.startswith('covers'):
                m[f'{prefix}{s["layer"]}/log10_{k}'] = float(np.log10(max(v, 1.0)))
    return m


def print_stats(title, metrics, stats):
    head = '  '.join(f'{k} {v:.4f}' for k, v in metrics.items() if isinstance(v, float))
    print(f'\n{title}: {head}')
    print(f'  {"layer":<6}{"n_in":>5}{"nonzero/unit":>14}{"rules/unit med":>16}{"max":>10}'
          f'{"shortest/node":>15}{"dead":>6}{"thr=net":>9}')
    for s in stats:
        sh = '-' if s['shortest_per_node_median'] is None else f'{s["shortest_per_node_median"]:.0f}'
        print(f'  {s["layer"]:<6}{s["n_in"]:>5}{s["nonzero_per_unit_median"]:>14.0f}'
              f'{s["covers_per_unit_median"]:>16.2e}{s["covers_per_unit_max"]:>10.2e}'
              f'{sh:>15}{s["dead_units"]:>6}{s["threshold_vs_network"]:>9.4f}')
