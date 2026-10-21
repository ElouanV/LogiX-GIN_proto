"""The paper's figures (plot_paper.py), drawn from the final folds of the experiments.

Which runs each figure reads is configs/paper_figures.yaml: a model is a key of its
``models`` block (experiment, model name, per-dataset exceptions), its folds are
results_final/<experiment>/<dataset>/<model>/fold<k>/ (final.json + interp.json, read by
utils.experiment.final_record). Moving the figures to new runs means editing that file.

Every figure is built from functions that already exist: the interpretability metrics of
interp.json, the QAT / hardening histories of the fine-tunes (qat.json, harden.json), the
rule statistics of the explorer (explorer.reader.rule_stats: shortest rule and number of
minimal covers per unit), the agreement scores of utils/agreement.py and the prototype
figures of utils/viz_proto.py. Plotting reads checkpoints and metrics, needs no GPU and
changes no result.

Figures that show one trained model take a *representative fold*, fixed in advance by
``fold_rule`` (median test balanced accuracy), never the best-looking one.

Colour follows the model (fixed slot order of utils.viz plus three more slots of the same
validated palette); every model also has its own marker and a direct label, so identity
never rests on colour alone.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import yaml

from utils.experiment import final_record
from utils.runs import RunDir
from utils.viz import AQUA, BLUE, GRID, INK, INK2, ORANGE, plt, save

CONFIG = 'configs/paper_figures.yaml'
# validated categorical slots 1-6 (dataviz reference palette, light mode): adjacent pairs
# pass the CVD check; slots 2-5 are under 3:1 against the surface, hence direct labels
SLOTS = [BLUE, ORANGE, AQUA, '#eda100', '#e87ba4', '#008300']
MARKERS = ['o', 's', 'D', '^', 'v', 'P']


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


class Figures:
    """One configuration of the paper's figures; ``draw(name)`` returns [(file stem, figure)]."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.get('root', 'results_final')
        self.split = cfg.get('split', 'test')
        keys = list(cfg['models'])
        self.style = {k: (SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]) for i, k in enumerate(keys)}

    # ------------------------------------------------------------------ sources

    def label(self, key):
        return self.cfg['models'][key]['label']

    def source(self, key, ds):
        m = self.cfg['models'][key]
        return m.get('by_dataset', {}).get(ds, m['experiment']), m['model']

    def folds(self, key, ds):
        """The model's final folds on ``ds`` as a DataFrame (empty if it was not run there)."""
        exp, model = self.source(key, ds)
        d = os.path.join(self.root, exp, ds, model)
        dirs = sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if f.startswith('fold'))
        rows = [final_record(os.path.join(d, f)) for f in dirs if os.path.exists(os.path.join(d, f, 'final.json'))]
        if not rows:
            warnings.warn(f'no final folds for {key} on {ds} ({d})')
        return pd.DataFrame(rows)

    def representative(self, key, ds):
        """The fold chosen by ``fold_rule``: lower median of the test balanced accuracy."""
        rule = self.cfg.get('fold_rule', 'median_balanced_acc')
        if rule != 'median_balanced_acc':
            raise ValueError(f'unknown fold_rule {rule!r}')
        df = self.folds(key, ds)
        if df.empty:
            raise ValueError(f'{key} has no final fold on {ds}')
        df = df.sort_values([f'{self.split}_balanced_acc', 'fold']).reset_index(drop=True)
        return df.iloc[(len(df) - 1) // 2]

    def run(self, key, ds):
        r = self.representative(key, ds)
        run = RunDir(r['run_dir'], ds)
        return r, run, run.model().eval()

    @staticmethod
    def teacher_acc(run_dir, split='test'):
        """Mean test accuracy of the k-fold teachers a student was distilled from."""
        rd = RunDir(run_dir)
        f = os.path.join(rd.teacher_path, '..', 'total_results.csv')
        return float(pd.read_csv(f)[f'{split}_acc'].mean()) if os.path.exists(f) else None

    def finish(self, fig):
        if self.cfg.get('preliminary', True):
            fig.text(0.995, 0.005, 'PRELIMINARY', ha='right', va='bottom', fontsize=7, color=INK2, alpha=0.8)
        return fig

    def _legend(self, fig, keys, extra=(), ncol=4):
        """One shared legend above the panels: marker + colour + name per model."""
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=self.style[k][0], marker=self.style[k][1], ls='', markersize=6,
                          markeredgecolor='white') for k in keys] + [h for h, _ in extra]
        labels = [self.label(k) for k in keys] + [lab for _, lab in extra]
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize=7,
                   handletextpad=0.3, columnspacing=1.2)

    def _mean_point(self, ax, key, x, y, label=False):
        """Folds as small marks, the mean ± std as a large labelled mark."""
        color, marker = self.style[key]
        ax.scatter(x, y, s=12, marker=marker, color=color, alpha=0.35, linewidths=0)
        mx, my = float(np.mean(x)), float(np.mean(y))
        if len(x) > 1:
            ax.errorbar(mx, my, xerr=np.std(x, ddof=1), yerr=np.std(y, ddof=1), fmt='none', ecolor=color,
                        elinewidth=1, alpha=0.7)
        ax.scatter([mx], [my], s=64, marker=marker, color=color, edgecolors='white', linewidths=1.5, zorder=3)
        if label:
            ax.annotate(f'{self.label(key)} ({len(x)})', (mx, my), xytext=(6, 4), textcoords='offset points',
                        fontsize=6.5, color=INK, va='center')

    # ------------------------------------------------------------------ figures

    def pareto(self, spec):
        """Rule-model accuracy against explanation size (trunk bits), one panel per dataset."""
        s = self.split
        dss = spec['datasets']
        fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.0), squeeze=False)
        shown = []
        for ax, ds in zip(axes[0], dss):
            teacher = None
            for key in spec['models']:
                df = self.folds(key, ds)
                col = f'{s}_interp_expl_bits{"_evidence" if spec.get("size", "evidence") == "evidence" else ""}_mean'
                if df.empty or col not in df:
                    continue
                df = df.dropna(subset=[col, f'{s}_interp_logic_acc'])
                self._mean_point(ax, key, df[col].clip(lower=1), 100 * df[f'{s}_interp_logic_acc'])
                shown += [key] if key not in shown else []
                teacher = teacher or self.teacher_acc(df['run_dir'].iloc[0], s)
            if teacher:
                ax.axhline(100 * teacher, color=INK2, lw=1, ls='--')
            ax.set_xscale('log')
            ax.set_title(ds, fontsize=9, loc='left')
            ax.set_xlabel('explanation size (trunk bits per prediction'
                          + (', evidence)' if spec.get('size', 'evidence') == 'evidence' else ')'))
        axes[0][0].set_ylabel(f'rule-model accuracy, {s} (%)')
        from matplotlib.lines import Line2D
        fig.tight_layout()
        self._legend(fig, [k for k in spec['models'] if k in shown],
                     [(Line2D([], [], color=INK2, lw=1, ls='--'), 'GIN teacher (network)')])
        return [('fig_pareto', self.finish(fig))]

    @staticmethod
    def _history(run_dir):
        """Per-epoch fine-tune history: qat.json (soft and rule val accuracy, rate) or
        harden.json (soft val accuracy only); None for a run without a fine-tune."""
        for name in ('qat.json', 'harden.json'):
            f = os.path.join(run_dir, name)
            if os.path.exists(f):
                h = pd.DataFrame(json.load(open(f))['history'])
                return h.rename(columns={'val_acc': 'val_acc_soft'})
        return None

    def binarisation(self, spec):
        """Left: validation accuracy of the soft network and of its rules during each
        fine-tune (mean over folds), with the binarised share below. Right: final test
        network vs rule-model accuracy; on the diagonal the network computes its rules."""
        ds, s = spec['dataset'], self.split
        fig = plt.figure(figsize=(7.2, 3.0))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.3, 1], hspace=0.08)
        ax, ax_rate, ax_end = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
        for key in spec['finetunes']:
            color, _ = self.style[key]
            hs = [h for h in (self._history(r) for r in self.folds(key, ds)['run_dir']) if h is not None]
            if not hs:
                continue
            h = pd.concat(hs).groupby('epoch').mean(numeric_only=True)
            ax.plot(h.index, 100 * h['val_acc_soft'], color=color, lw=1.5, ls='--')
            if 'val_acc_rules' in h:
                ax.plot(h.index, 100 * h['val_acc_rules'], color=color, lw=2)
            for col in ('rate', 'sharpness'):
                if col in h:
                    ax_rate.plot(h.index, h[col], color=color, lw=1.5)
                    ax_rate.set_ylabel(col, fontsize=7)
        ax.set_ylabel('validation accuracy (%)')
        ax.tick_params(labelbottom=False)
        ax.plot([], [], color=INK2, lw=2, label='rules')
        ax.plot([], [], color=INK2, lw=1.5, ls='--', label='network')
        ax.legend(fontsize=6.5, loc='lower right')
        ax_rate.set_xlabel('fine-tune epoch')
        for key in [spec['start']] + spec['finetunes']:
            df = self.folds(key, ds)
            if df.empty or f'{s}_interp_logic_acc' not in df:
                continue
            self._mean_point(ax_end, key, 100 * df[f'{s}_acc'], 100 * df[f'{s}_interp_logic_acc'])
        lo, hi = ax_end.get_xlim()
        lo, hi = min(lo, ax_end.get_ylim()[0]), max(hi, ax_end.get_ylim()[1])
        ax_end.plot([lo, hi], [lo, hi], color=GRID, lw=1, zorder=0)
        ax_end.set_xlabel(f'network accuracy, {s} (%)')
        ax_end.set_ylabel(f'rule-model accuracy, {s} (%)')
        fig.tight_layout()
        keys = [k for k in [spec['start']] + spec['finetunes'] if not self.folds(k, ds).empty]
        self._legend(fig, keys)
        fig.suptitle(ds, fontsize=9, x=0.01, ha='left', y=1.0)
        return [('fig_binarisation', self.finish(fig))]

    @staticmethod
    def unit_stats(model, eps=1e-6):
        """Per conv unit: layer, shortest rule, number of minimal covers, dead (no weight > eps)."""
        from explorer.reader import rule_stats
        rows = []
        for l, conv in enumerate(model.convs):
            ll = conv.nn[0]
            W, S = ll.weight.detach().cpu().numpy(), (-ll.b).detach().cpu().numpy()
            dead = (W > eps).sum(1) == 0
            mins, covers = rule_stats(W, S)
            rows += [{'layer': l, 'min_rule': m, 'covers': c, 'dead': bool(d)} for m, c, d in zip(mins, covers, dead)]
        return pd.DataFrame(rows)

    def rule_complexity(self, spec):
        """Shortest rule and number of minimal covers per conv unit, per layer, dense vs
        layer-wise Hoyer (every fold that has the sparse fine-tune)."""
        ds, key = spec['dataset'], spec['model']
        rows = []
        for run_dir in self.folds(key, ds)['run_dir']:
            sparse = os.path.join(run_dir, 'sparse', spec['sparse'])
            if not os.path.exists(os.path.join(sparse, 'best.pt')):
                continue
            for variant, path in (('dense', run_dir), ('Hoyer', sparse)):
                st = self.unit_stats(RunDir(path, ds).model())
                rows.append(st.assign(variant=variant))
        if not rows:
            raise ValueError(f'no {key} fold on {ds} has the sparse fine-tune {spec["sparse"]}')
        df = pd.concat(rows, ignore_index=True)
        n_folds = len(rows) // 2
        layers = sorted(df.layer.unique())
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        colors = {'dense': BLUE, 'Hoyer': ORANGE}
        for ax, col, ylabel, log in ((axes[0], 'min_rule', 'shortest rule (literals)', False),
                                     (axes[1], 'covers', 'minimal covers per unit', True)):
            for vi, variant in enumerate(('dense', 'Hoyer')):
                d = df[(df.variant == variant) & ~df.dead]
                data = [d[d.layer == l][col].dropna().astype(float).values for l in layers]
                pos = [l + (vi - 0.5) * 0.36 for l in layers]
                bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False)
                for b in bp['boxes']:
                    b.set(facecolor=colors[variant], alpha=0.45, edgecolor=colors[variant])
                for part in ('medians', 'whiskers', 'caps'):
                    for line in bp[part]:
                        line.set(color=colors[variant])
            ax.set_xticks(layers, [f'L{l}' for l in layers])
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale('log')
        for l in layers:                                 # dead units of the sparse trunk
            dead = df[(df.variant == 'Hoyer') & (df.layer == l)].dead.mean()
            if dead > 0:
                axes[0].annotate(f'{100 * dead:.0f}% dead', (l + 0.18, 1), xycoords=('data', 'axes fraction'),
                                 xytext=(0, -2), textcoords='offset points', ha='center', va='top', fontsize=6.5,
                                 color=INK2)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[v], alpha=0.6) for v in colors]
        axes[1].legend(handles, list(colors), fontsize=7, loc='upper left')
        fig.suptitle(f'{ds}, {self.label(key)}, {n_folds} folds (dead units left out)', fontsize=8, x=0.01,
                     ha='left')
        fig.tight_layout()
        return [('fig_rule_complexity', self.finish(fig))]

    def agreement(self, spec):
        """Agreement histograms of the prototypes the evidence cites (with >= 1 cared bit),
        by decreasing class separation, on the training graphs of the representative fold."""
        from torch_geometric.loader import DataLoader
        from utils.agreement import agreement_scores, plot_agreement, separation
        from utils.datasets import spec as ds_spec
        from utils.runs import EVAL_BATCH_SIZE
        from utils.viz_proto import citations, explain_graphs
        out = []
        for ds in spec['datasets']:
            r, run, model = self.run(spec['model'], ds)
            data = run.data
            scores, y = agreement_scores(model, DataLoader(data['train_dataset'], batch_size=EVAL_BATCH_SIZE), 'cpu')
            a = scores[0]
            n_care = model.proto_layers[0].care.sum(1).detach().cpu().numpy()
            cited = citations(model, explain_graphs(model, data[f'{self.split}_dataset'], evidence=True))[0]
            keep = [k for k in cited if n_care[k] > 0]
            sep = separation(a, y)
            keep = sorted(keep, key=lambda k: -sep[k])[:spec.get('top', 8)]
            names = {c: ds_spec(ds).class_name(c) for c in np.unique(y)}
            fig = plot_agreement(a[:, keep], y, names, f'{ds}, {self.label(spec["model"])}, fold {r["fold"]}',
                                 n_care=n_care[keep], ids=keep)
            out.append((f'fig_agreement_{ds}', self.finish(fig)))
        return out

    def prototypes(self, spec):
        """Prototype cards of the representative fold (utils.viz_proto.plot_prototypes)."""
        from utils.viz_proto import explain_graphs, plot_prototypes
        out = []
        for ds in spec['datasets']:
            r, run, model = self.run(spec['model'], ds)
            data = run.data
            recs = explain_graphs(model, data[f'{self.split}_dataset'], evidence=True)
            fig = plot_prototypes(model, recs, data['train_dataset'], ds, top=spec.get('top', 6))
            out.append((f'fig_prototypes_{ds}', self.finish(fig)))
        return out

    def explanation(self, spec):
        """One prediction unfolded down to atom counts: the first correctly predicted test
        graph of a ground-truth class that contains its region (viz_proto.pick_examples)."""
        from utils.viz_proto import explain_graphs, head_literal_map, literal_kind, pick_examples, plot_unfolding
        ds = spec['dataset']
        r, run, model = self.run(spec['model'], ds)
        recs = explain_graphs(model, run.data[f'{self.split}_dataset'], evidence=True)
        lit_map = head_literal_map(model)
        for _, rec in pick_examples(recs, ds):
            ks = [lit_map[c][1] for c in rec['expl'] or [] if literal_kind(model, c) == 'present']
            if ks:
                return [('fig_explanation', self.finish(plot_unfolding(model, rec, ds, k=ks[0])))]
        raise ValueError(f'no worked example of {ds} cites a prototype presence')

    NAMES = ('pareto', 'binarisation', 'rule_complexity', 'agreement', 'prototypes', 'explanation')

    def draw(self, name):
        if name not in self.NAMES:
            raise ValueError(f'unknown figure {name!r} (one of {self.NAMES})')
        return getattr(self, name)(self.cfg['figures'][name])


def write(figures, out_dir, formats=('pdf',)):
    """Save [(stem, figure)] to out_dir; returns the paths."""
    return [p for stem, fig in figures for p in save(fig, out_dir, stem, formats)]
