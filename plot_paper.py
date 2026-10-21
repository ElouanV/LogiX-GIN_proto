"""Figures of the paper, from the final folds named in configs/paper_figures.yaml
(utils/paper_figures.py). CPU only, reads checkpoints and metrics, changes no result.

    python plot_paper.py                         # every figure -> paper/figures/
    python plot_paper.py pareto agreement --out /tmp/preview
    python plot_paper.py --config configs/paper_figures.yaml --png

Writes paper/figures/fig_<name>.pdf (fig_agreement_<dataset>.pdf and
fig_prototypes_<dataset>.pdf per dataset), the files paper/figures/fig_<name>.tex include.
A figure whose runs are missing is skipped with a warning; the others are still drawn.
"""
import argparse
import traceback

from utils.paper_figures import CONFIG, Figures, load_config, write


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('figures', nargs='*', default=list(Figures.NAMES), help=f'any of {", ".join(Figures.NAMES)}')
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--out', default='paper/figures')
    ap.add_argument('--png', action='store_true', help='also write PNG previews')
    a = ap.parse_args()

    figs = Figures(load_config(a.config))
    formats = ('pdf', 'png') if a.png else ('pdf',)
    failed = []
    for name in a.figures:
        try:
            print(*write(figs.draw(name), a.out, formats), sep='\n')
        except Exception:
            traceback.print_exc()
            failed.append(name)
    if failed:
        raise SystemExit(f'not drawn: {", ".join(failed)}')


if __name__ == '__main__':
    main()
