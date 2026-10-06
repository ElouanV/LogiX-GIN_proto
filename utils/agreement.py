"""Agreement distributions: how well each prototype separates the classes.

For every graph and every prototype, the *agreement* is the fraction of the bits the
prototype cares about that the graph's input matches (Hamming agreement; all bits without
a care mask): 1 means the input equals the prototype on every cared bit. The input is the
symbolic trunk (binary rule outputs, as interp_metrics.py reads the model):

    node-level prototypes   the graph's best-matching node (max over its nodes), i.e.
                            "does the graph contain this pattern"
    graph-level prototypes  the pooled graph vector (mean / max of the binary node states,
                            a sum block thresholded by phi_sum)

A prototype separates the classes when the agreement distributions of the classes do not
overlap. ``separation`` = max over classes c of |2·AUC(c vs rest) - 1|, in [0, 1]
(0: same distribution for every class, 1: a threshold on the agreement isolates a class).

``log_agreement`` runs at the end of every prototype training run (train_proto.py, so
every Optuna trial and every final fold): it writes agreement.{png,json,npz} next to the
checkpoint and logs them, plus the summary metrics, to MLflow. A failure here never stops
training.
"""
import json
import os
import warnings

import numpy as np
import torch
from torch_geometric.nn import global_max_pool

SPLITS = ('train', 'val', 'test')
PLOT_SPLIT = 'test'


@torch.no_grad()
def agreement_scores(model, loader, device):
    """Per prototype layer (model.proto_layers order): agreement [G × K] and labels [G]."""
    from interp_metrics import trunk_states
    model.eval()
    layers = model.proto_layers
    scores, labels = [[] for _ in layers], []
    for data in loader:
        data = data.to(device)
        x = data.x.float() if data.x is not None else torch.ones((data.num_nodes, model.num_features), device=device)
        xs = trunk_states(model, x, data.edge_index, symbolic=True)
        for i, (layer, inp) in enumerate(zip(layers, _symbolic_inputs(model, xs, data.batch))):
            a = layer_agreement(layer, inp)
            if a.shape[0] != data.num_graphs:                # node-level: best-matching node
                a = global_max_pool(a, data.batch)
            scores[i].append(a.cpu())
        labels.append(data.y.view(-1).cpu())
    return [torch.cat(s).numpy() for s in scores], torch.cat(labels).numpy()


def _symbolic_inputs(model, xs, batch):
    """prototype_inputs with the graph-level sum block binarised, as in the symbolic reading."""
    h = torch.hstack(xs)
    if hasattr(model, 'proto_node'):                             # both
        return [h, model.graph_input(h, batch, symbolic=True)]
    if hasattr(model, 'graph_input'):                            # graph
        return [model.graph_input(h, batch, symbolic=True)]
    return [h]                                                   # node


def layer_agreement(layer, x):
    """Fraction of each prototype's cared bits that x matches: [n × K] in [0, 1]."""
    p, c = layer.prototypes, layer.care
    agree = x @ (c * p).t() + (1 - x) @ (c * (1 - p)).t()
    n_care = c.sum(1)
    return torch.where(n_care > 0, agree / n_care.clamp(min=1), torch.ones_like(agree))


def separation(a, y):
    """Per prototype (column of a [G × K]): max over classes of |2·AUC(class vs rest) - 1|."""
    from sklearn.metrics import roc_auc_score
    classes = np.unique(y)
    out = np.zeros(a.shape[1])
    if len(classes) < 2:
        return out
    for k in range(a.shape[1]):
        out[k] = max(abs(2 * roc_auc_score(y == c, a[:, k]) - 1) for c in classes)
    return out


def summarize(scores, y, split):
    """Metrics of one split: mean / max separation over the prototypes of every layer."""
    m, per = {}, {}
    for i, a in enumerate(scores):
        sep = separation(a, y)
        tag = '' if len(scores) == 1 else f'L{i}_'
        m[f'agreement/{split}_{tag}separation_mean'] = float(sep.mean())
        m[f'agreement/{split}_{tag}separation_max'] = float(sep.max())
        per[f'layer{i}'] = {'separation': sep.tolist(),
                            'mean_by_class': {str(c): a[y == c].mean(0).tolist() for c in np.unique(y)}}
    return m, per


def plot_agreement(a, y, class_names=None, title=None, bins=30, n_care=None):
    """One panel per prototype: agreement histogram of each class (one colour per class).
    ``n_care``: bits each prototype cares about, shown in its title (0 = always true)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.viz import AQUA, BLUE, INK2, ORANGE
    classes = np.unique(y)
    palette = [BLUE, ORANGE, AQUA] + list(plt.get_cmap('tab10').colors)
    k = a.shape[1]
    cols = min(k, 8)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 1.5 * rows + 0.6), sharex=True, squeeze=False)
    # bins over the observed range (shared by the panels): agreements often sit in a narrow
    # band (e.g. 0.65-0.9), which [0, 1] bins would blur into one or two bars
    lo, hi = float(a.min()), float(a.max())
    pad = max(0.02, 0.05 * (hi - lo))
    edges = np.linspace(max(0.0, lo - pad), min(1.0, hi + pad), bins + 1)
    sep = separation(a, y)
    for j, ax in enumerate(axes.flat):
        if j >= k:
            ax.axis('off')
            continue
        for ci, c in enumerate(classes):
            v = a[y == c, j]
            # density per class, so the class sizes don't hide the shapes
            ax.hist(v, bins=edges, density=True, histtype='stepfilled', alpha=0.35, color=palette[ci % len(palette)])
            ax.hist(v, bins=edges, density=True, histtype='step', lw=1.0, color=palette[ci % len(palette)])
        bits = '' if n_care is None else f'  {int(n_care[j])} bits'
        ax.set_title(f'P{j}  sep {sep[j]:.2f}{bits}', fontsize=7, color=INK2)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    names = class_names or {}
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[ci % len(palette)], alpha=0.6) for ci in range(len(classes))]
    fig.legend(handles, [names.get(int(c), f'class {int(c)}') for c in classes], loc='upper right',
               ncol=len(classes), fontsize=7, frameon=False)
    fig.supxlabel('agreement with the prototype (fraction of cared bits matched)', fontsize=7)
    if title:
        fig.suptitle(title, fontsize=8, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def log_agreement(model, loaders, device, path, class_names=None, title=None):
    """Compute, save next to the checkpoint and log to MLflow; never raises."""
    from utils import tracking
    try:
        metrics, record, arrays = {}, {}, {}
        n_care = [l.care.sum(1).detach().cpu().numpy() for l in model.proto_layers]
        for split, loader in loaders.items():
            scores, y = agreement_scores(model, loader, device)
            m, per = summarize(scores, y, split)
            metrics.update(m)
            record[split] = per
            arrays[f'{split}_y'] = y
            for i, a in enumerate(scores):
                arrays[f'{split}_layer{i}'] = a
                record[split][f'layer{i}']['n_care'] = n_care[i].tolist()
        with open(os.path.join(path, 'agreement.json'), 'w') as f:
            json.dump({'definition': __doc__.split('\n\n')[1], 'metrics': metrics, 'splits': record}, f, indent=1)
        np.savez_compressed(os.path.join(path, 'agreement.npz'), **arrays)
        files = ['agreement.json', 'agreement.npz']
        split = PLOT_SPLIT if PLOT_SPLIT in loaders else next(iter(loaders))
        n_layers = sum(1 for k in arrays if k.startswith(f'{split}_layer'))
        for i in range(n_layers):
            name = 'agreement.png' if n_layers == 1 else f'agreement_layer{i}.png'
            fig = plot_agreement(arrays[f'{split}_layer{i}'], arrays[f'{split}_y'], class_names,
                                 f'{title or ""} {split} set'.strip(), n_care=n_care[i])
            fig.savefig(os.path.join(path, name), dpi=130)
            import matplotlib.pyplot as plt
            plt.close(fig)
            files.append(name)
        tracking.log_metrics(metrics)
        for f in files:
            tracking.log_artifact(os.path.join(path, f), artifact_path='agreement')
        return metrics
    except Exception as e:
        warnings.warn(f'agreement distribution failed, the run continues: {e!r}')
        return None
