"""Run an experiment described by a YAML config (configs/experiments/*.yaml); the
pipeline, units and outputs are documented in utils/experiment.py.

    python run_experiment.py configs/experiments/sum_ablation.yaml run --datasets MUTAG PROTEINS
    python run_experiment.py configs/experiments/sum_ablation.yaml run --datasets AIDS --stages teacher teacher_summary
    python run_experiment.py configs/experiments/sum_ablation.yaml status
    python run_experiment.py configs/experiments/sum_ablation.yaml plan        # list the work left in the progress table
    python run_experiment.py configs/experiments/sum_ablation.yaml summary --datasets MUTAG
    python run_experiment.py configs/experiments/sum_ablation.yaml pack --datasets AIDS   # results -> bundles/
    python run_experiment.py configs/experiments/sum_ablation.yaml unpack bundles/<archive>.tar.gz

The protocol every machine follows is docs/EXPERIMENT_PROTOCOL.md.

One machine should own a dataset: its teachers, studies and final runs then live in one
place and nothing has to be copied.
"""
import argparse
import os

from utils.experiment import Experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the units of some datasets')
    r.add_argument('--datasets', nargs='+', help='default: every dataset of the config')
    r.add_argument('--stages', nargs='+', choices=['teacher', 'teacher_summary', 'hps', 'final', 'summary'])
    r.add_argument('--jobs', type=int, help='default: the config\'s jobs')
    r.add_argument('--dry', action='store_true', help='only list the units to run')
    r.add_argument('--allow_dirty', action='store_true', help='skip the preflight refusal (tests only)')
    pk = sub.add_parser('pack', help='archive the results of some datasets for the main machine')
    pk.add_argument('--datasets', nargs='+', required=True)
    pk.add_argument('--out', help='default: bundles/<experiment>_<datasets>_<host>.tar.gz')
    up = sub.add_parser('unpack', help='extract a pack archive on the main machine')
    up.add_argument('archive')
    up.add_argument('--overwrite', action='store_true')
    for name in ('status', 'plan', 'summary'):
        p = sub.add_parser(name)
        p.add_argument('--datasets', nargs='+')
    u = sub.add_parser('unit', help='(internal) run one unit')
    u.add_argument('--stage', required=True)
    u.add_argument('--dataset', required=True)
    u.add_argument('--model')
    u.add_argument('--fold', type=int)
    a = ap.parse_args()

    exp = Experiment(a.config)
    datasets = getattr(a, 'datasets', None) or exp.cfg['datasets']
    if a.cmd == 'run':
        raise SystemExit(0 if exp.run(datasets, a.stages, a.jobs, a.dry, a.allow_dirty) else 1)
    if a.cmd == 'pack':
        import socket
        os.makedirs('bundles', exist_ok=True)
        exp.pack(a.datasets, a.out or os.path.join(
            'bundles', f"{exp.name}_{'-'.join(a.datasets)}_{socket.gethostname()}.tar.gz"))
    elif a.cmd == 'unpack':
        exp.unpack(a.archive, a.overwrite)
    if a.cmd == 'unit':
        exp.run_unit(a.stage, a.dataset, a.model, a.fold)
    elif a.cmd == 'status':
        print(f"{'dataset':14s} {'stage':16s} done/total")
        for (ds, stage), (d, n) in exp.status(datasets).items():
            print(f'{ds:14s} {stage:16s} {d}/{n}')
    elif a.cmd == 'plan':
        print(f'{exp.plan(datasets)} units reported as queued')
    elif a.cmd == 'summary':
        for ds in datasets:
            df = exp.summary(ds)
            print(ds, 'no final run yet' if df is None else '\n' + df.round(3).to_string())


if __name__ == '__main__':
    main()
