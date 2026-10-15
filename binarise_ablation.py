"""Which binarisation costs accuracy when a prototype LogiX-GIN is read as rules?

The rules read every value at 0.5: the literals of every conv unit, the conv unit outputs
(the node states one layer passes to the next and to the readout) and the head literals.
This binarises one part at a time, everything else soft, and reports the accuracy of each
condition and its agreement with the all-soft network (docs/QAT.md, "Binarising the trunk,
not the head, costs accuracy"). Evaluation only; it uses the QAT hooks (quant_rate = 1 in
eval mode binarises, models_proto/tell.quant_noise).

    soft         nothing binarised: the network
    head         head literals only (the prototype similarity step stays as computed)
    trunk        every conv literal and conv unit output
    trunk_lits   conv literals only
    trunk_outs   conv unit outputs only
    L<l>         conv layer l only (its literals and outputs)
    rule         everything: the rule model (= interp_metrics logic_acc)

Predictions are the argmax of the head margins W.lit + b (the graded head output can
saturate to ties). Also reported per conv layer: the share of soft unit outputs and of soft
literals in (0.1, 0.9), i.e. far from any bit.

    python binarise_ablation.py --run_path <seed dir> [<seed dir> ...] [--ckpt best.pt] --split test --out table.json

With several run paths, it prints mean ± std per condition and the paired difference from
the soft network (runs worse, Wilcoxon signed-rank p).
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch_geometric.loader import DataLoader

from interp_metrics import aggregate


def conditions(num_layers):
    """{name: (literals hard per conv layer, outputs hard per conv layer, head literals hard)}."""
    none, every = [0] * num_layers, [1] * num_layers
    out = {'soft': (none, none, 0), 'head': (none, none, 1), 'trunk': (every, every, 0),
           'trunk_lits': (every, none, 0), 'trunk_outs': (none, every, 0)}
    for l in range(num_layers):
        one = [int(i == l) for i in range(num_layers)]
        out[f'L{l}'] = (one, one, 0)
    out['rule'] = (every, every, 1)
    return out


@torch.no_grad()
def predict(model, loader, device, lits, outs, head):
    """Margin argmax and labels with the given parts binarised (QAT hooks, eval mode)."""
    model.eval()
    for i, conv in enumerate(model.convs):
        conv.nn[0].quant_rate = float(lits[i])
    model.fc.quant_rate = float(head)
    model.quant_rate = 0.0
    preds, ys = [], []
    try:
        for d in loader:
            d = d.to(device)
            x, xs = d.x.float(), []
            for i, conv in enumerate(model.convs):
                x = conv(torch.hstack([x, 1 - x]), d.edge_index)
                if outs[i]:
                    x = (x >= 0.5).float()
                xs.append(x)
            preds.append(model.head(model.readout(xs, d.batch), margin=True).argmax(1).cpu())
            ys.append(d.y.view(-1).cpu())
    finally:
        for conv in model.convs:
            conv.nn[0].quant_rate = 0.0
        model.fc.quant_rate = 0.0
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def graded_shares(model, loader, device, low=0.1, high=0.9):
    """Per conv layer, the share of soft unit outputs and of soft literals in (low, high)."""
    model.eval()
    L = len(model.convs)
    outs, lits, n = np.zeros(L), np.zeros(L), 0
    for d in loader:
        d = d.to(device)
        x = d.x.float()
        for i, conv in enumerate(model.convs):
            inp = torch.hstack([x, 1 - x])
            lit = conv.nn[0].phi_in(aggregate(conv, inp, d.edge_index))
            x = conv(inp, d.edge_index)
            lits[i] += ((lit > low) & (lit < high)).float().mean().item()
            outs[i] += ((x > low) & (x < high)).float().mean().item()
        n += 1
    return (outs / n).tolist(), (lits / n).tolist()


def binarise_ablation(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    res, soft = {}, None
    for name, (lits, outs, head) in conditions(len(model.convs)).items():
        p, y = predict(model, loader, device, lits, outs, head)
        soft = p if name == 'soft' else soft
        res[name] = {'acc': float((p == y).mean()), 'agree': float((p == soft).mean())}
    res['graded_outputs'], res['graded_literals'] = graded_shares(model, loader, device)
    return res


def summarise(rows):
    """Mean ± std per condition over runs, and the paired difference from 'soft'."""
    names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    soft = np.array([r['soft']['acc'] for r in rows])
    lines = []
    for k in names:
        a = np.array([r[k]['acc'] for r in rows])
        ag = np.array([r[k]['agree'] for r in rows])
        d = a - soft
        p = wilcoxon(a, soft).pvalue if k != 'soft' and len(rows) > 5 and (d != 0).any() else float('nan')
        lines.append(f'{k:11s} acc {a.mean():.3f} ± {a.std():.3f}   Δ {d.mean():+.3f}   worse {int((d < 0).sum())}/{len(rows)}'
                     f'   p {p:.3f}   agree {ag.mean():.3f}')
    go = np.array([r['graded_outputs'] for r in rows]).mean(0)
    gl = np.array([r['graded_literals'] for r in rows]).mean(0)
    lines.append(f'in (0.1, 0.9) per conv layer: outputs {np.round(go, 3).tolist()}  literals {np.round(gl, 3).tolist()}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', nargs='+', required=True, help='directories holding the checkpoint and data.pkl')
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out', default=None, help='write {run_path: result} as JSON here')
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for p in a.run_path:
        model = torch.load(os.path.join(p, a.ckpt), map_location=device, weights_only=False)
        dataset = pickle.load(open(os.path.join(p, 'data.pkl'), 'rb'))[f'{a.split}_dataset']
        results[p] = binarise_ablation(model, dataset, device)
        print(p, {k: round(v['acc'], 3) for k, v in results[p].items() if isinstance(v, dict)}, flush=True)
    print(summarise(list(results.values())))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
