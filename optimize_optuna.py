"""Optuna hyper-parameter search of the LogiX-GIN students (search spaces, protocol and
outputs: utils/hps.py).

One process is one worker of one study (dataset, model, fold); start several on the same
study to run its trials in parallel (scripts/run_hps.sh does).

    python optimize_optuna.py --dataset MUTAG --model node_mask_push --fold 0 --n_trials 30
    python optimize_optuna.py --export            # results_hps/best_params.csv + per-study files
"""
import argparse

from utils import hps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset')
    ap.add_argument('--model', choices=sorted(hps.MODELS))
    ap.add_argument('--fold', type=int, default=0, help='k-fold split whose train/val the study tunes on')
    ap.add_argument('--n_trials', type=int, default=30, help='finished trials the study stops at (all workers)')
    ap.add_argument('--sampler_seed', type=int, default=0)
    ap.add_argument('--report_every', type=int, default=50, help='epochs between pruning checks after warmup')
    ap.add_argument('--root', default=hps.ROOT, help='a new root for a new search space')
    ap.add_argument('--export', action='store_true', help='only export every study under --root')
    a = ap.parse_args()
    if a.export:
        print(hps.export_all(a.root).to_string())
        return
    if not (a.dataset and a.model):
        ap.error('--dataset and --model are required unless --export')
    study = hps.run_worker(a.dataset, a.model, a.fold, a.n_trials, a.root, a.sampler_seed, a.report_every)
    print(f'{study.study_name}: {len(study.trials)} trials')


if __name__ == '__main__':
    main()
