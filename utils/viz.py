"""Figures of the prototype interpretability results (interp_metrics.py outputs).

`collect_interp` gathers every ``interp_<split>.json`` under ``results_proto/`` into one
table (one row per run and seed), labelled by method, trunk and post-hoc fine-tune;
the ``plot_*`` functions draw the comparison figures from that table. Plotting only
reads metrics already computed, so it needs no GPU and changes no result.

Colour encodes the prototype readout (no mask / care mask / care mask + shared
vocabulary), three categorical slots that stay distinguishable under colour-vision
deficiency; marker shape encodes the level (node / graph) and a hollow marker means no
push. Means carry direct labels, so identity never rests on colour alone.
"""
import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical slots 1-3 (validated all-pairs for scatter forms), text inks
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
READOUT_COLOR = {'dense': BLUE, 'mask': ORANGE, 'vocab': AQUA}
LEVEL_MARKER = {'node': 'o', 'graph': 's'}

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight',
})


def _arg(cfg, key, default=None):
    m = re.search(rf'(?:^|\|){key}=([^|]+)', cfg)
    return m.group(1) if m else default


def _sparse_tag(cfg):
    """Short tag of a sparsify_proto.py fine-tune from its config dir name."""
    tags = []
    if _arg(cfg, 'hard') == 'True':
        tags.append('hard')
    if float(_arg(cfg, 'hoyer_reg', 0)) or float(_arg(cfg, 'hoyer_fc', 0)):
        tags.append('hoyer')
    if float(_arg(cfg, 'unit_hoyer', 0) or 0):
        tags.append('unit')
    return ' + '.join(tags) or 'finetune'


def collect_interp(root='results_proto', split='test'):
    """One row per scored run: metrics of interp_<split>.json plus its configuration.

    Columns added: ds, level, push, mask, vocab_reg, layer_cost, anneal (mask annealed
    with --mask_anneal_frac), hidden, layers, trunk ('64x3'), teacher_lr, seed, sparse
    (post-hoc fine-tune tag or ''), method (short label) and path.
    """
    rows = []
    for f in glob.glob(os.path.join(root, '*', '*', '*', '*', '**', f'interp_{split}.json'), recursive=True):
        rel = os.path.relpath(f, root).split(os.sep)
        ds, cfgdir, teacher, seed = rel[:4]
        if not seed.isdigit():
            continue
        cfg_file = os.path.join(root, ds, cfgdir, 'config.txt')
        cfg = open(cfg_file).read().strip() if os.path.exists(cfg_file) else cfgdir
        sparse = _sparse_tag(rel[5]) if len(rel) > 6 and rel[4] == 'sparse' else ''
        r = json.load(open(f))
        r.update(ds=ds, level=_arg(cfg, 'proto_level', cfgdir.split('-')[0]),
                 push=_arg(cfg, 'push_every', '0') != '0', mask=_arg(cfg, 'proto_mask') == 'True',
                 vocab_reg=float(_arg(cfg, 'vocab_reg', 0)), layer_cost=_arg(cfg, 'mask_layer_cost', ''),
                 anneal=_arg(cfg, 'mask_anneal_frac') is not None,
                 hidden=int(_arg(teacher, 'hidden_dim')), layers=int(_arg(teacher, 'num_layers')),
                 teacher_lr=float(_arg(teacher, 'lr')), seed=int(seed), sparse=sparse, path=os.path.dirname(f))
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['trunk'] = df.hidden.astype(str) + 'x' + df.layers.astype(str)
    df['method'] = df.apply(method_label, axis=1)
    return df


def method_label(r):
    """'node + mask + push', '+ vocab 0.1', '+ layer cost 0.1,1,1', '+ hard' ..."""
    parts = [r['level']] + (['mask'] if r['mask'] else []) + (['push'] if r['push'] else [])
    name = ' + '.join(parts) if len(parts) > 1 else f'{r["level"]} dense'
    if name == 'node + mask + push':
        name = 'NMP'
    if r['vocab_reg']:
        name += f' + vocab {r["vocab_reg"]:g}'
    if r['layer_cost']:
        name += f' + layer cost {r["layer_cost"]}'
    if r['sparse']:
        name += f' + {r["sparse"]}'
    return name


def readout_kind(r):
    return 'vocab' if r['vocab_reg'] else 'mask' if r['mask'] else 'dense'


def main_runs(df, ds='Mutagenicity'):
    """The comparable set: one dataset, 64x3 trunk, teacher lr 1e-3, annealed masks."""
    return df[(df.ds == ds) & (df.trunk == '64x3') & (df.teacher_lr == 1e-3) & (~df['mask'] | df.anneal)]


def summarize(df, by='method', cols=('acc', 'auc', 'balanced_acc', 'logic_acc', 'logic_fidelity', 'rule_backed',
                                      'expl_bits_mean', 'protos_cited', 'units_cited', 'gt_precision')):
    """Mean, std and seed list per group."""
    cols = [c for c in cols if c in df]
    g = df.groupby(by)
    out = g[cols].agg(['mean', 'std'])
    out.columns = [f'{a}_{b}' for a, b in out.columns]
    out['seeds'] = g.seed.apply(lambda s: ','.join(map(str, sorted(s))))
    out['n'] = g.seed.size()
    return out


def _scatter_runs(ax, df, x, y, label_offsets=None):
    """Seeds as small marks, the mean ± std as a large labelled mark, per method."""
    label_offsets = label_offsets or {}
    for name, d in df.groupby('method'):
        r = d.iloc[0]
        color, marker = READOUT_COLOR[readout_kind(r)], LEVEL_MARKER.get(r['level'], 'o')
        face = color if r['push'] else 'white'
        ax.scatter(d[x], d[y], s=14, marker=marker, facecolors=face, edgecolors=color, alpha=0.35, linewidths=0.8)
        mx, my = d[x].mean(), d[y].mean()
        ax.errorbar(mx, my, xerr=d[x].std() if len(d) > 1 else None, yerr=d[y].std() if len(d) > 1 else None,
                    fmt='none', ecolor=color, elinewidth=1, alpha=0.7)
        ax.scatter([mx], [my], s=70, marker=marker, facecolors=face, edgecolors=color, linewidths=2,
                   zorder=3)
        dx, dy, ha = label_offsets.get(name, (6, 4, 'left'))
        ax.annotate(f'{name} ({len(d)})', (mx, my), xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.5, color=INK, ha=ha, va='center')


def _encoding_legend(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='', marker='o', mfc=c, mec=c, label=lab) for lab, c in
         (('no mask', BLUE), ('care mask', ORANGE), ('care mask + shared vocabulary', AQUA))]
    h += [Line2D([], [], ls='', marker='o', mfc=INK2, mec=INK2, label='node prototypes'),
          Line2D([], [], ls='', marker='s', mfc=INK2, mec=INK2, label='graph prototypes'),
          Line2D([], [], ls='', marker='o', mfc='white', mec=INK2, label='hollow: no push')]
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7.5)


# hand-placed labels of the Mutagenicity means (points per x, y): (dx, dy, alignment)
TRADEOFF_LABELS = {
    'acc': {'NMP + vocab 0.1': (0, 15, 'center'), 'graph + mask': (8, 7, 'left'), 'node + mask': (-6, -12, 'right'),
            'NMP': (9, -5, 'left'), 'graph + mask + push': (9, -4, 'left'), 'node + push': (0, 14, 'center'),
            'node dense': (0, -14, 'center'), 'graph dense': (9, 3, 'left'), 'graph + push': (9, -6, 'left')},
    'logic_fidelity': {'node + push': (9, 0, 'left'), 'node + mask': (9, 2, 'left'),
                       'NMP + vocab 0.1': (0, -15, 'center'), 'NMP': (9, -3, 'left'),
                       'node dense': (-9, -6, 'right'), 'graph + push': (9, 5, 'left'),
                       'graph dense': (9, -6, 'left'), 'graph + mask': (9, -2, 'left'),
                       'graph + mask + push': (9, 0, 'left')},
}
LEVER_LABELS = {'NMP + hard + hoyer': (-7, 7, 'right'), 'NMP + hard': (7, 7, 'left'),
                'node dense + hard': (0, -12, 'center'), 'NMP + vocab 1': (-7, 4, 'right'),
                'NMP + vocab 0.1': (7, 7, 'left'), 'NMP + vocab 0.3': (-7, -9, 'right'),
                'NMP + hoyer + unit': (7, 3, 'left'), 'NMP + hoyer': (7, -7, 'left'),
                'NMP + layer cost 0.25,1,1': (7, 3, 'left'), 'NMP + layer cost 0.1,1,1': (7, -6, 'left'),
                'NMP': (9, 5, 'left')}


def plot_tradeoff(df, ds='Mutagenicity', label_offsets=None):
    """Accuracy and fidelity against explanation size (bits per prediction, log scale)."""
    label_offsets = label_offsets or (TRADEOFF_LABELS if ds == 'Mutagenicity' else {})
    d = main_runs(df, ds)
    d = d[(d['sparse'] == '') & (d.layer_cost == '') & d.vocab_reg.isin([0, 0.1])].dropna(subset=['expl_bits_mean'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, gridspec_kw={'wspace': 0.3})
    for ax, y, lab in zip(axes, ('acc', 'logic_fidelity'), ('test accuracy (network)', 'fidelity (rules = network)')):
        _scatter_runs(ax, d, 'expl_bits_mean', y, label_offsets.get(y))
        ax.set_xscale('log')
        ax.set_xlim(2.5, 4000)
        ax.set_xlabel('explanation size: bits per prediction (log)')
        ax.set_ylabel(lab)
    axes[0].set_title(f'{ds}: same accuracy, explanations 20-100x shorter', loc='left', fontsize=10)
    axes[1].set_title('fidelity does not follow explanation size', loc='left', fontsize=10)
    _encoding_legend(axes[1])
    fig.text(0, -0.03, 'Test split. Small marks: seeds; large marks: mean ± std; (n) seeds.', fontsize=7.5, color=INK2)
    return fig


def plot_vocab_sweep(df, ds='Mutagenicity', seeds=(0, 1, 2)):
    """NMP metrics against the shared-vocabulary weight, seed-matched."""
    d = main_runs(df, ds)
    d = d[(d.method.str.startswith('NMP')) & (d['sparse'] == '') & (d.layer_cost == '') & d.seed.isin(seeds)]
    panels = [('units_cited', 'trunk units used (all explanations)'), ('expl_bits_mean', 'bits per prediction'),
              ('acc', 'test accuracy'), ('logic_fidelity', 'fidelity'), ('rule_backed', 'rule-backed rate'),
              ('gt_precision', 'NO2/NH2 precision')]
    regs = sorted(d.vocab_reg.unique())
    xs = np.arange(len(regs))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    for ax, (col, lab) in zip(axes.flat, panels):
        g = d.groupby('vocab_reg')[col]
        for i, v in enumerate(regs):
            vals = d[d.vocab_reg == v][col]
            ax.scatter(np.full(len(vals), i), vals, s=12, color=AQUA if v else ORANGE, alpha=0.4)
        ax.errorbar(xs, g.mean().reindex(regs), yerr=g.std().reindex(regs), color=INK2, lw=1.2, marker='o',
                    ms=5, mfc=INK, mec=INK, capsize=2)
        ax.set_title(lab, loc='left', fontsize=9)
        if col == 'gt_precision' and 'gt_chance' in d:
            ax.axhline(d.gt_chance.mean(), color=INK2, ls=':', lw=1)
            ax.annotate('chance', (xs[-1], d.gt_chance.mean()), xytext=(0, 3), textcoords='offset points',
                        fontsize=7, color=INK2, ha='right')
    for ax in axes[1]:
        ax.set_xticks(xs, [f'{v:g}' for v in regs])
        ax.set_xlabel('vocab_reg')
    fig.suptitle(f'{ds}, NMP: shared-vocabulary penalty (seeds {",".join(map(str, seeds))}, test split)',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    return fig


def plot_trunk_grid(df, ds='Mutagenicity', seeds=(0, 1, 2)):
    """NMP over trunk width x depth: one heatmap per metric, mean over seeds in each cell."""
    d = df[(df.ds == ds) & (df.method == 'NMP') & df.seed.isin(seeds) & (df.teacher_lr == 1e-3)]
    panels = [('acc', 'test accuracy', '{:.3f}'), ('logic_fidelity', 'fidelity', '{:.2f}'),
              ('units_cited', 'units used', '{:.0f}'), ('gt_precision', 'NO2/NH2 precision', '{:.2f}')]
    widths, depths = sorted(d.hidden.unique()), sorted(d.layers.unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 2.6), gridspec_kw={'wspace': 0.08})
    for ax, (col, lab, fmt) in zip(axes, panels):
        m = d.groupby(['hidden', 'layers'])[col].mean().unstack().reindex(index=widths, columns=depths)
        ax.imshow(m.values, cmap='Blues', aspect='auto')
        for i in range(len(widths)):
            for j in range(len(depths)):
                v = m.values[i, j]
                if not np.isnan(v):
                    dark = v > np.nanmean(m.values)
                    ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8.5,
                            color='white' if dark else INK)
        ax.set_xticks(range(len(depths)), [f'{l} layers' for l in depths])
        ax.set_yticks(range(len(widths)), [f'width {w}' for w in widths] if ax is axes[0] else [])
        ax.set_title(lab, loc='left', fontsize=9)
        ax.grid(False)
    fig.suptitle(f'{ds}, NMP over trunk sizes (mean of seeds {",".join(map(str, seeds))}, test split; darker = higher)',
                 x=0.01, ha='left', fontsize=10, y=1.05)
    return fig


def plot_collapse(df, datasets=('AIDS', 'BBBP')):
    """Balanced accuracy per seed, dense vs NMP node prototypes: 0.5 = majority-class collapse."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.2), sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        d = df[(df.ds == ds) & (df['sparse'] == '') & df.method.isin(['node dense', 'NMP'])]
        for i, (name, color) in enumerate((('node dense', BLUE), ('NMP', ORANGE))):
            v = d[d.method == name].sort_values('seed')
            if v.empty:
                continue
            jitter = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), v.balanced_acc, s=28, color=color, edgecolors='white', linewidths=1,
                       zorder=3)
            n_coll = int((v.balanced_acc < 0.55).sum())
            ax.annotate(f'{n_coll}/{len(v)} collapsed', (i, 0.47), ha='center', fontsize=7.5, color=INK2)
        ax.axhline(0.5, color=INK2, ls=':', lw=1)
        ax.set_xticks([0, 1], ['node dense', 'NMP'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0.42, 1.0)
        ax.set_title(ds, loc='left', fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel('balanced accuracy (test)')
    fig.suptitle('Majority-class collapse (balanced acc 0.5) of dense prototypes; one mark per seed',
                 x=0.01, ha='left', fontsize=10, y=1.03)
    return fig


def plot_levers(df, ds='Mutagenicity', seeds=(0, 1, 2)):
    """Every lever tried on NMP against NMP itself, seed-matched: fidelity vs bits."""
    d = main_runs(df, ds)
    d = d[(d.method.str.startswith('NMP') | (d.method == 'node dense + hard')) & d.seed.isin(seeds)]
    d = d.dropna(subset=['expl_bits_mean'])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, v in d.groupby('method'):
        mx, my = v.expl_bits_mean.mean(), v.logic_fidelity.mean()
        ref = name == 'NMP'
        ax.errorbar(mx, my, xerr=v.expl_bits_mean.std(), yerr=v.logic_fidelity.std(), fmt='o',
                    color=ORANGE if ref else INK2, ms=8 if ref else 5, elinewidth=0.8, capsize=2)
        dx, dy, ha = LEVER_LABELS.get(name, (6, 3, 'left'))
        ax.annotate(f'{name}  (acc {v.acc.mean():.3f})', (mx, my), xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.5, color=INK, ha=ha, va='center')
    ax.set_xscale('log')
    ax.set_xlim(1, 2000)
    ax.set_ylim(None, 1.04)
    ax.set_xlabel('bits per prediction (log)')
    ax.set_ylabel('fidelity')
    ax.set_title(f'{ds}: levers on NMP (seeds {",".join(map(str, seeds))}, mean ± std, test split)',
                 loc='left', fontsize=10)
    return fig


def save(fig, out_dir, name, formats=('png', 'pdf')):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        p = os.path.join(out_dir, f'{name}.{fmt}')
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths
