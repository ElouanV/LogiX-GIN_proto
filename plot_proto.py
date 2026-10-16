"""Figures that open trained node-level prototype runs: unit vocabulary, prototype cards and
worked examples (utils/viz_proto.py). CPU only, reads checkpoints, changes no result.

    python plot_proto.py --run_path <seed dir> [--compare <seed dir>] --names NMP "NMP + vocab 0.1" \
        --out figures/proto

vocabulary.png compares every run given; prototypes_<i>.png and examples_<i>.png are drawn
per run; with --inside N, also trunk_<i>_<j>.png (trunk states against the first prototype
the explanation cites) and unfold_<i>_<j>.png (the prediction unfolded down to atom counts)
for the first N worked examples.
"""
import argparse

from utils.runs import RunDir
from utils.viz_proto import (citations, explain_graphs, head_literal_map, literal_kind, pick_examples, plot_examples,
                             plot_prototypes, plot_trunk_match, plot_unfolding, plot_vocabulary, save)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', nargs='+', required=True, help='seed directories (node-level prototype runs)')
    ap.add_argument('--names', nargs='+', default=None, help='display names, one per run')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--dataset', default=None, help='default: read from the run path')
    ap.add_argument('--top', type=int, default=6, help='prototype cards per run')
    ap.add_argument('--inside', type=int, default=0, help='trunk + unfolding figures for this many examples')
    ap.add_argument('--out', default='figures/proto')
    a = ap.parse_args()
    names = a.names or [f'run {i}' for i in range(len(a.run_path))]

    models, cited = {}, {}
    for i, (path, name) in enumerate(zip(a.run_path, names)):
        run = RunDir(path, a.dataset)
        ds, model, data = run.dataset, run.model().eval(), run.data
        recs = explain_graphs(model, data[f'{a.split}_dataset'])
        models[name], cited[name] = model, set(citations(model, recs)[0])
        print(*save(plot_prototypes(model, recs, data['train_dataset'], ds, top=a.top), a.out, f'prototypes_{i}'))
        print(*save(plot_examples(model, recs, ds), a.out, f'examples_{i}'))
        lit_map = head_literal_map(model)
        for j, (_, r) in enumerate(pick_examples(recs, ds)[:a.inside]):
            ks = [lit_map[c][1] for c in r['expl'] or [] if literal_kind(model, c) == 'present']
            if ks:                                 # needs a cited prototype presence
                print(*save(plot_trunk_match(model, r, ks[0], ds), a.out, f'trunk_{i}_{j}'))
                print(*save(plot_unfolding(model, r, ds, k=ks[0]), a.out, f'unfold_{i}_{j}'))
    if all(m.proto.masked for m in models.values()):
        print(*save(plot_vocabulary(models, cited), a.out, 'vocabulary'))


if __name__ == '__main__':
    main()
