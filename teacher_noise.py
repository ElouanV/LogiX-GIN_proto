"""How random the GIN teacher's binary node states are, and what it does to its accuracy and
to the students distilled from it (docs/GUMBEL_TEACHER.md).

The teacher binarises every conv unit with models.model.gumbel_sigmoid(hard=True): one
Gumbel(0, 1) draw is added to the logit and the sum thresholded at 0. The draw happens at
evaluation too (tau only rescales logit + noise, it does not change its sign), so each
evaluation of the same checkpoint is a new sample. Per k-fold teacher (test split):

    reported        test accuracy in the teacher's total_results.csv (one noisy evaluation)
    noisy_*         mean / std / min / max over --draws evaluations
    deterministic   accuracy with the noise off (deterministic=True: bit = [logit > 0])
    majority        accuracy with every bit at its most likely value: P(logit + g > 0) =
                    1 - exp(-e^logit) for g ~ Gumbel(0, 1), >= 1/2 iff logit >= ln ln 2 (-0.367)
    p1              P(bit = 1) of each (node, unit) over the draws
    uncertain       share of node bits with 0.05 < p1 < 0.95
    ceiling         mean max(p1, 1 - p1): best bit accuracy a deterministic student can
                    reach against the sampled targets it is distilled on
    agree2          mean p1^2 + (1 - p1)^2: agreement of two independent draws

With --student <final run dir> ... the student's symbolic trunk states (interp_metrics.trunk_states)
are compared to the teacher's majority bit (p1 >= 0.5) and to its deterministic bit, and its own
soft unit outputs to p1 on the bits the teacher is unsure about (0.1 < p1 < 0.9): their
correlation, and the share of graded outputs (in (0.1, 0.9)) there and on the other bits.

    python teacher_noise.py --datasets Mutagenicity PROTEINS --draws 20 --out noise.csv
    python teacher_noise.py --student results_final/<exp>/<ds>/<model>/fold0/run/... --draws 20
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from utils.datasets import get_dataset
from utils.runs import EVAL_BATCH_SIZE, RunDir, load_teacher, make_loaders, num_features, split_datasets
from utils.hps import TEACHER_CFG


@torch.no_grad()
def teacher_draws(teacher, loader, draws):
    """Per draw: predicted classes [G]; and P(bit = 1) [N × L·h] over the draws, plus the
    deterministic bits and predictions."""
    preds, bits = [], []
    for _ in range(draws):
        p, b = [], []
        for d in loader:
            xs, ys = teacher.forward_e(d.x.float(), d.edge_index, d.batch)
            b.append(torch.hstack(ys[:-1]))
            p.append(ys[-1].argmax(-1))
        preds.append(torch.cat(p))
        bits.append(torch.cat(b))
    det_p, det_b, maj_p = [], [], []
    for d in loader:
        p, ys = forward_fixed_bits(teacher, d, 0.0)
        det_p.append(p)
        det_b.append(torch.hstack(ys))
        maj_p.append(forward_fixed_bits(teacher, d, MODE)[0])
    return torch.stack(preds), torch.stack(bits).mean(0), torch.cat(det_p), torch.cat(det_b), torch.cat(maj_p)


MODE = math.log(math.log(2))        # P(logit + Gumbel(0, 1) > 0) >= 1/2 iff logit >= ln ln 2


def forward_fixed_bits(teacher, d, threshold):
    """The teacher's prediction and node states with every bit [logit > threshold]: 0 is
    deterministic=True of models.model.gumbel_sigmoid, MODE the most likely bit."""
    x, xs = d.x.float(), []
    for conv in teacher.convs:
        x = conv(x, d.edge_index)
        if not teacher.nogumbel:
            x = (x > threshold).float()
        xs.append(x)
    h = torch.hstack(xs)
    z = torch.hstack([global_mean_pool(h, d.batch), global_max_pool(h, d.batch), global_add_pool(h, d.batch)])
    out = torch.sigmoid(teacher.fc2(torch.relu(teacher.fc1(z))))
    return out.argmax(-1), xs


def teacher_row(dataset, fold, tdir, draws):
    ds = get_dataset(dataset)
    import pickle
    data = pickle.load(open(os.path.join(tdir, 'data.pkl'), 'rb'))
    test = make_loaders(split_datasets(ds, data), EVAL_BATCH_SIZE)['test']
    y = torch.cat([d.y.reshape(-1) for d in test])
    t = load_teacher(tdir, num_features(ds), ds.num_classes, 'cpu')
    preds, p1, det_pred, _, maj_pred = teacher_draws(t, test, draws)
    acc = (preds == y).float().mean(1).numpy()
    rep = pd.read_csv(os.path.join(os.path.dirname(tdir), 'total_results.csv'))
    rep = float(rep.loc[rep['seed'] == fold, 'test_acc'].iloc[0])
    return {'dataset': dataset, 'fold': fold, 'reported': rep, 'noisy_mean': acc.mean(), 'noisy_std': acc.std(ddof=1),
            'noisy_min': acc.min(), 'noisy_max': acc.max(), 'deterministic': float((det_pred == y).float().mean()),
            'majority': float((maj_pred == y).float().mean()),
            'uncertain': float(((p1 > 0.05) & (p1 < 0.95)).float().mean()),
            'ceiling': float(torch.maximum(p1, 1 - p1).mean()), 'agree2': float((p1 ** 2 + (1 - p1) ** 2).mean())}


@torch.no_grad()
def student_row(run_path, draws):
    from interp_metrics import trunk_states
    run = RunDir(run_path)
    ds = get_dataset(run.dataset)
    test = make_loaders(split_datasets(ds, run.data), EVAL_BATCH_SIZE)['test']
    t = run.teacher(num_features(ds), ds.num_classes, 'cpu')
    m = run.model('cpu').eval()
    _, p1, _, det_b, _ = teacher_draws(t, test, draws)
    s = torch.cat([torch.hstack(trunk_states(m, d.x.float(), d.edge_index)) for d in test])
    soft = torch.cat([torch.hstack(trunk_states(m, d.x.float(), d.edge_index, symbolic=False)) for d in test])
    graded = (soft > 0.1) & (soft < 0.9)
    unsure10 = (p1 > 0.1) & (p1 < 0.9)
    unsure = (p1 > 0.05) & (p1 < 0.95)
    return {'run': run_path, 'dataset': run.dataset, 'seed': run.seed,
            'vs_majority': float((s == (p1 >= 0.5).float()).float().mean()),
            'vs_deterministic': float((s == det_b).float().mean()),
            'vs_one_draw': float((s * p1 + (1 - s) * (1 - p1)).mean()),
            'ceiling': float(torch.maximum(p1, 1 - p1).mean()),
            'student_bit_on_uncertain_is_majority': float((s[unsure] == (p1[unsure] >= 0.5).float()).float().mean())
            if unsure.any() else None,
            'teacher_unsure': float(unsure10.float().mean()),
            'corr_soft_p1_unsure': float(np.corrcoef(soft[unsure10].numpy(), p1[unsure10].numpy())[0, 1])
            if unsure10.sum() > 1 else None,
            'graded_on_unsure': float(graded[unsure10].float().mean()) if unsure10.any() else None,
            'graded_on_sure': float(graded[~unsure10].float().mean()) if (~unsure10).any() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='*', default=[])
    ap.add_argument('--student', nargs='*', default=[], help='final student run directories (best.pt)')
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--out', default=None, help='CSV of the teacher rows')
    a = ap.parse_args()
    torch.manual_seed(0)
    rows = []
    for dataset in a.datasets:
        for tdir in sorted(glob.glob(os.path.join('results', dataset, TEACHER_CFG, '[0-9]'))):
            rows.append(teacher_row(dataset, int(os.path.basename(tdir)), tdir, a.draws))
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    if rows:
        df = pd.DataFrame(rows)
        print(df.groupby('dataset')[[c for c in df.columns if c not in ('dataset', 'fold')]].mean().round(4).to_string())
        if a.out:
            df.to_csv(a.out, index=False)
    for p in a.student:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in student_row(p, a.draws).items()}, flush=True)


if __name__ == '__main__':
    main()
