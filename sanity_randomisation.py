"""Model-randomisation sanity check of a trained model's explanations (utils/sanity.py).

Usage:
    python sanity_randomisation.py --run_path <run dir | results_final fold dir> [...] \
        [--split test] [--mode permute|init] [--order independent|cascade] [--draws 3] [--seed 0]
Writes <run dir>/sanity_<order>_<mode>_<split>.json (one row per stage and draw) and prints a table
of the stage means.
"""
import argparse
import json
import os

import pandas as pd

from utils.runs import RunDir, get_device
from utils.sanity import MODES, ORDERS, RandomisationTest


def resolve(path):
    """A results_final fold dir points at its run through final.json; any other path is a run."""
    final = os.path.join(path, 'final.json')
    return json.load(open(final))['run_dir'] if os.path.exists(final) else path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', nargs='+', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--mode', default='permute', choices=MODES)
    ap.add_argument('--order', default='independent', choices=ORDERS)
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    device = get_device()
    for path in a.run_path:
        run = RunDir(resolve(path))
        test = RandomisationTest(run.model(device, a.ckpt), run.pickled(a.split), run.dataset, device)
        rows = test.run(a.mode, a.order, a.draws, a.seed, log=lambda s: None)
        out = run.file(f'sanity_{a.order}_{a.mode}_{a.split}.json')
        with open(out, 'w') as f:
            json.dump({'split': a.split, 'mode': a.mode, 'order': a.order, 'draws': a.draws, 'seed': a.seed, 'rows': rows}, f, indent=1)
        df = pd.DataFrame(rows).drop(columns=['mode', 'order', 'draw'])
        order = list(dict.fromkeys(df['stage']))
        with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.precision', 3):
            print(path)
            print(df.groupby('stage', sort=False).mean(numeric_only=True).loc[order].to_string())
        print(f'-> {out}')


if __name__ == '__main__':
    main()
