"""Figures of the prototype interpretability results, from the interp_<split>.json files
that interp_metrics.py writes next to each checkpoint (no training, no GPU).

    python plot_interp.py --out figures/interp
    python plot_interp.py --out figures/interp --csv figures/interp/runs.csv   # + the table

Writes tradeoff (accuracy and fidelity vs explanation size), vocab_sweep, trunk_grid,
levers and collapse (AIDS/BBBP) as PNG and PDF.
"""
import argparse

from utils.viz import (collect_interp, plot_collapse, plot_levers, plot_tradeoff, plot_trunk_grid,
                       plot_vocab_sweep, save, summarize)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default='results_proto')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--dataset', default='Mutagenicity')
    ap.add_argument('--out', default='figures/interp')
    ap.add_argument('--csv', default=None, help='also write the per-run table here')
    a = ap.parse_args()

    df = collect_interp(a.root, a.split)
    if df.empty:
        raise SystemExit(f'no interp_{a.split}.json under {a.root}: run interp_metrics.py first')
    if a.csv:
        df.drop(columns=['path']).to_csv(a.csv, index=False)
    figs = {'tradeoff': plot_tradeoff(df, a.dataset), 'vocab_sweep': plot_vocab_sweep(df, a.dataset),
            'trunk_grid': plot_trunk_grid(df, a.dataset), 'levers': plot_levers(df, a.dataset),
            'collapse': plot_collapse(df)}
    for name, fig in figs.items():
        print(*save(fig, a.out, name))
    print(summarize(df[df.ds == a.dataset], by=['method', 'trunk']).round(3).to_string())


if __name__ == '__main__':
    main()
