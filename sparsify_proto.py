"""Post-training sparsification of a trained LogiX-GIN run (base or prototype): the layer-by-layer Hoyer fine-tune.

Works on both kinds of runs: ``results_logic/...`` (base LogiX-GIN, train_logic.py) and
``results_proto/...`` (prototype variants, train_proto.py).

Why: a conv unit of layer 1-2 needs ~13 literals to reach its threshold because its
weight mass is spread over ~35 mid-sized weights, which gives it ~1e10 minimal covers
(rules). Dropping small weights at extraction time barely helps (see min_covers.py);
the mass has to be *moved* onto few weights, which needs training.

The procedure is the one of classic LogiX-GIN (nbs/LayerWiseRules.ipynb cells 11, 19 and
23), ported unchanged to any run: the head first, then the conv layers from the last to
the first, each step training only that layer's weights (``weight_sigma``,
``weight_exp``; prototypes, masks and thresholds stay fixed) with Adam on the notebook's
loss (task BCE + NLL, sqrt penalty on every conv, notebook Hoyer + sqrt penalty on the
head; its ``train_sparsity_epoch``). After every epoch the model is kept if (val acc,
-nonzero weights of the layer) beats the kept one, or if it has fewer nonzero weights
(> 1e-4) at no less than ``--lw_tol`` x the kept val acc; a step ends after ``patience``
epochs without a keep (at most ``--lw_max_epochs``) and the next one starts from the kept
model. Head: lr 0.01, patience 100, conv_reg 0.1, fc_reg 0.1; convs: lr 0.005, patience
50, conv_reg 0.1, fc_reg 0.01 (the notebook's). ``--head_lr``, ``--conv_lr`` (one value or
one per conv layer) and ``--{head,conv}_patience`` override them; on Mutagenicity NMP the
notebook's values never keep a head, L1 or L0 step. The tolerance is relative to the last
kept model, so val accuracy can drift down by up to 1% per keep (notebook behaviour,
kept; the step history in sparsity.json shows it). No pruning afterwards, as in the
notebook. Deviation: an exception in a batch is raised instead of printed and skipped.

The joint variant (all layers at once with a row-wise Hoyer penalty, then pruning and
recovery, plus fan-in schedules, a unit penalty, ``--task_only`` and ``--hard``) did worse
on prototype runs (Mutagenicity NMP: fidelity 0.69 vs 0.80) and was removed; it is in the
history up to commit ca49a45 (``git show ca49a45:sparsify_proto.py``), and its output
directories (``sparse/epochs=...|hoyer_reg=...``) stay readable by utils/viz.py.

Output: ``<run_path>/sparse/<config>/`` with ``best.pt``, ``args.json``, a link to the
run's ``data.pkl``, and ``sparsity.json`` (metrics and rule statistics before and
after, and the per-step history). The directory is a regular run directory, so
``latent_logic.py --run_path`` and ``tests/bench_min_covers.py --run_path`` work on it
unchanged. The run is also logged to MLflow (experiment ``sparsify/<dataset>``, see
utils/tracking.py) and the result is registered as ``logix-proto-sparse-<dataset>-<level>``.

    python sparsify_proto.py --run_path "results_proto/Mutagenicity/<cfg>/<baseline cfg>/0"
    python sparsify_proto.py --run_path ... --stats_only      # rule statistics, no training
"""
import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F

from rule_eval import logical_layers, print_stats, rule_metrics, rule_stats
from train_proto import test_epoch
from utils import tracking
from utils.evaluation import full_eval, held_out
from utils.runs import load_run
from utils.utils import set_seed, zero_nan_gradients


def unit_groups(model):
    """(LogicalLayer, lower-unit index of every input column) for the layers above L0.

    The lower units of a conv layer are the previous layer's units (both polarities). The
    head of a base run reads conv units through mean|max|sum pooling; the head of a
    prototype run reads prototypes, one group per prototype over its pooling ops
    (``readout_groups``), not conv units.
    """
    L, h = len(model.convs), model.convs[0].nn[0].out_features
    out = []
    for c in model.convs[1:]:
        n = c.nn[0].in_features // 2
        out.append((c.nn[0], torch.arange(2 * n) % n))
    n = model.fc.in_features // 2
    head = model.readout_groups() if hasattr(model, 'readout_groups') else torch.arange(n) % (L * h)
    out.append((model.fc, head.repeat(2)))
    return out


def used_units(model, eps=0.0):
    """Per layer above L0 (and head): how many lower units are read by some nonzero weight."""
    return [int(torch.zeros(int(g.max()) + 1).index_add_(0, g, (ll.weight.detach().cpu() > eps).float().sum(0)).gt(0).sum())
            for ll, g in unit_groups(model)]


# ---------------------------------------------------------------------------
# layer-by-layer sparsification (nbs/LayerWiseRules.ipynb cells 11, 19, 23)
# ---------------------------------------------------------------------------

def notebook_hoyer_loss(weights, lambda_=1.0, epsilon=1e-12):
    """Hoyer loss of the notebook (cell 11), verbatim: per row, but normalised by the
    number of entries of the whole matrix."""
    l1_norm = torch.sum(torch.abs(weights), -1)
    l2_norm = torch.sqrt(torch.sum(weights**2, -1) + epsilon)
    hoyer = (torch.sqrt(torch.tensor(weights.numel())) - l1_norm / l2_norm) / \
            (torch.sqrt(torch.tensor(weights.numel())) - 1 + epsilon)
    loss = lambda_ * (1 - hoyer)
    return loss.mean()


def notebook_sparsity_epoch(model, loader, device, optimizer, num_classes, conv_reg=1, fc_reg=1):
    """train_sparsity_epoch of the notebook (cell 11): task loss + sqrt penalties + head Hoyer.
    optimizer=None: the same loss on loader without training (eval mode, no gradient, RNG
    untouched, utils.evaluation.held_out), i.e. the val loss."""
    if optimizer is None:
        with held_out():
            return _sparsity_epoch(model.eval(), loader, device, None, num_classes, conv_reg, fc_reg)
    return _sparsity_epoch(model.train(), loader, device, optimizer, num_classes, conv_reg, fc_reg)


def _sparsity_epoch(model, loader, device, optimizer, num_classes, conv_reg, fc_reg):
    train = optimizer is not None
    total_loss = total_correct = 0
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        if data.y.numel() == 0 or data.x.isnan().any() or data.y.isnan().any():
            continue
        y = data.y.reshape(-1).to(device).long()
        if train:
            optimizer.zero_grad()
        model.fc.phi_in.tau = 10
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))
        pred = out.argmax(-1)
        loss = F.binary_cross_entropy(out.reshape(-1), F.one_hot(y, num_classes=num_classes).float().reshape(-1)) \
            + F.nll_loss(F.log_softmax(out, dim=-1), y)
        for conv in model.convs:
            loss = loss + conv_reg * (torch.sqrt(torch.clamp(conv.nn[0].weight, min=1e-5)).sum(-1).mean()
                                      + conv.nn[0].phi_in.entropy)
        fc = model.fc
        loss = loss + (notebook_hoyer_loss(torch.clamp(fc.weight, min=1e-5)) + fc.reg_loss + fc.phi_in.entropy) \
            + fc_reg * (torch.sqrt(torch.clamp(fc.weight, min=1e-5)).sum(-1).mean() + fc.phi_in.entropy)
        if train:
            loss.backward()
            zero_nan_gradients(model)
            optimizer.step()
        total_loss += loss.item() * data.num_graphs / len(loader.dataset)
        total_correct += pred.eq(y).sum().item() / len(loader.dataset)
    return total_loss, total_correct


# (lr, patience, conv_reg, fc_reg) of the head step (cell 19) and of each conv step (cell 23)
LAYERWISE_HEAD = (0.01, 100, 0.1, 0.1)
LAYERWISE_CONV = (0.005, 50, 0.1, 0.01)


def layerwise_steps(model, head_lr=None, head_patience=None, conv_lr=None, conv_patience=None):
    """[(name, 'head' or conv index, lr, patience, conv_reg, fc_reg)] in the notebook's order.
    None keeps the notebook's value; conv_lr is one lr for every conv layer or one per layer
    (L0 first)."""
    L = len(model.convs)
    lr0, pat0, *regs = LAYERWISE_HEAD
    head = ('head', 'head', head_lr or lr0, head_patience or pat0, *regs)
    lr0, pat0, *regs = LAYERWISE_CONV
    lrs = [lr0] * L if not conv_lr else list(conv_lr) * L if len(conv_lr) == 1 else list(conv_lr)
    if len(lrs) != L:
        raise ValueError(f'conv_lr: one value or one per conv layer ({L}), got {len(conv_lr)}')
    return [head] + [(f'L{l}', l, lrs[l], conv_patience or pat0, *regs) for l in reversed(range(L))]


def _layer(model, which):
    return model.fc if which == 'head' else model.convs[which].nn[0]


def nonzero_weights(layer, eps=1e-4):
    return int((layer.weight > eps).sum().item())


def layerwise_sparsify(model, train_loader, val_loader, device, num_classes, max_epochs=1000, tol=0.99,
                       log=lambda metrics, step: None, **step_args):
    """The notebook's layer-by-layer Hoyer fine-tune (step_args: layerwise_steps' lr and
    patience overrides). Returns (model, per-step history)."""
    history, step_no = [], 0
    for name, which, lr, patience, conv_reg, fc_reg in layerwise_steps(model, **step_args):
        layer = _layer(model, which)
        optimizer = torch.optim.Adam([layer.weight_sigma, layer.weight_exp], lr=lr)
        val = test_epoch(model, val_loader, device)
        best = start = (val, -nonzero_weights(layer))
        best_model, left, keeps = copy.deepcopy(model), patience, 0
        t0 = time.time()
        for epoch in range(max_epochs):
            loss, _ = notebook_sparsity_epoch(model, train_loader, device, optimizer, num_classes, conv_reg, fc_reg)
            val_loss, _ = notebook_sparsity_epoch(model, val_loader, device, None, num_classes, conv_reg, fc_reg)
            val, n = test_epoch(model, val_loader, device), nonzero_weights(layer)
            if (val, -n) > best or (-n > best[1] and val >= tol * best[0]):
                best_model, left, best, keeps = copy.deepcopy(model), patience, (val, -n), keeps + 1
            left -= 1
            log({f'layerwise/{name}/val_acc': val, f'layerwise/{name}/nonzero': n,
                 f'layerwise/{name}/train_loss': loss, f'layerwise/{name}/val_loss': val_loss,
                 'layerwise/train_loss': loss, 'layerwise/val_loss': val_loss, 'layerwise/val_acc': val}, step_no)
            step_no += 1
            if epoch % 25 == 0:
                print(f'{name} epoch {epoch:4d}  loss {loss:.4f}  val {val:.4f}  nonzero {n}  kept {best}  '
                      f'patience {left}  ({time.time() - t0:.0f}s)', flush=True)
            if left == 0:
                break
        model = copy.deepcopy(best_model)
        rec = {'step': name, 'lr': lr, 'patience': patience, 'epochs': epoch + 1, 'keeps': keeps, 'start_val_acc': start[0],
               'start_nonzero': -start[1], 'val_acc': best[0], 'nonzero': -best[1],
               'n_weights': _layer(model, which).weight.numel()}
        print(f'{name}: val {start[0]:.4f} -> {best[0]:.4f}, nonzero {-start[1]} -> {-best[1]} of '
              f'{rec["n_weights"]} after {epoch + 1} epochs ({keeps} keeps)', flush=True)
        history.append(rec)
    return model, history


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='a trained seed directory holding best.pt')
    ap.add_argument('--dataset', default=None, help='default: read from run_path')
    ap.add_argument('--lw_max_epochs', type=int, default=1000, help='max epochs per layer')
    ap.add_argument('--lw_tol', type=float, default=0.99, help='keep sparser models down to this x val acc')
    ap.add_argument('--head_lr', type=float, default=None, help=f'head step lr (default {LAYERWISE_HEAD[0]})')
    ap.add_argument('--head_patience', type=int, default=None, help=f'default {LAYERWISE_HEAD[1]}')
    ap.add_argument('--conv_lr', type=float, nargs='+', default=None,
                    help=f'conv step lr: one value, or one per conv layer, L0 first (default {LAYERWISE_CONV[0]})')
    ap.add_argument('--conv_patience', type=int, default=None, help=f'default {LAYERWISE_CONV[1]}')
    ap.add_argument('--batch_size', type=int, default=None, help='default: the run\'s batch size')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None, help='default: <run_path>/sparse/<config>')
    ap.add_argument('--stats_only', action='store_true', help='print the run\'s rule statistics and exit')
    a = ap.parse_args()

    set_seed(a.seed)
    loaded = load_run(a.run_path, a.dataset, batch_size=a.batch_size)
    run, model_proto, device, num_classes = loaded.run, loaded.model, loaded.device, loaded.num_classes
    stage, ds_name, run_seed, run_args = run.stage, run.dataset, run.seed, run.args
    train_loader, val_loader, test_loader = (loaded.loaders[s] for s in ('train', 'val', 'test'))
    loaders = {'val': val_loader, 'test': test_loader}
    before = {'metrics': full_eval(model_proto, loaders, device),
              'rules': rule_stats(model_proto, val_loader, device)}
    print_stats('before', before['metrics'], before['rules'])
    if a.stats_only:
        return

    step_args = {'head_lr': a.head_lr, 'head_patience': a.head_patience,
                 'conv_lr': a.conv_lr, 'conv_patience': a.conv_patience}
    sp_args = {'layerwise': True, 'lw_max_epochs': a.lw_max_epochs, 'lw_tol': a.lw_tol,
               'batch_size': a.batch_size, 'seed': a.seed, **step_args}
    # overrides only appear in the name when set, so default runs keep their directory
    cfg = '|'.join([f'{k}={sp_args[k]}' for k in ('layerwise', 'lw_max_epochs', 'lw_tol')] +
                   [f'{k}={",".join(map(str, v)) if isinstance(v, list) else v}' for k, v in step_args.items() if v is not None])
    out_dir = a.out_dir or os.path.join(a.run_path, 'sparse', cfg)
    os.makedirs(out_dir, exist_ok=True)
    level = run_args.get('proto_level', 'logic')
    registered = f'logix-proto-sparse-{ds_name}-{level}' if stage == 'proto' else f'logix-gin-sparse-{ds_name}'
    code_dirs = ('models_proto',) if stage == 'proto' else ('models',)

    with tracking.run(f'sparsify/{ds_name}', run_name=f'{level}|{cfg}/seed{run_seed}',
                      params={**sp_args, **{f'source/{k}': v for k, v in run_args.items()}},
                      tags={'config': cfg, 'source_config': f'{run.config}/{run.teacher_config}', 'seed': run_seed,
                            'source_stage': stage, 'kind': 'seed', 'dataset': ds_name, 'proto_level': level,
                            'source_run_path': a.run_path, 'out_dir': out_dir},
                      stage='finetune/hoyer', dataset=ds_name, parent_run_id=tracking.fine_tune_parent(a.run_path)):
        tracking.log_metrics({f'before/{k}': v for k, v in before['metrics'].items()})
        tracking.log_metrics(rule_metrics(before['rules'], prefix='before/rules/'))

        model_proto, history = layerwise_sparsify(model_proto, train_loader, val_loader, device, num_classes,
                                                  a.lw_max_epochs, a.lw_tol, log=tracking.log_metrics,
                                                  **step_args)
        torch.save(model_proto, os.path.join(out_dir, 'best.pt'))
        tracking.write_pipeline(out_dir)
        n = sum(ll.weight.numel() for ll in logical_layers(model_proto))
        frac = sum(int((ll.weight <= 1e-4).sum()) for ll in logical_layers(model_proto)) / n
        after = {'metrics': full_eval(model_proto, loaders, device),
                 'rules': rule_stats(model_proto, val_loader, device),
                 'pruned_fraction': frac, 'layerwise_steps': history}
        print_stats('before', before['metrics'], before['rules'])
        print_stats('after', after['metrics'], after['rules'])
        after['used_units'] = used_units(model_proto)
        print(f'lower units read by L1.., head: {after["used_units"]}')

        run.write_finetune(out_dir, 'sparsify', sp_args, {'before': before, 'after': after}, record_name='sparsity')

        tracking.log_metrics({f'after/{k}': v for k, v in after['metrics'].items()})
        tracking.log_metrics(rule_metrics(after['rules'], prefix='after/rules/'))
        tracking.log_artifact(os.path.join(out_dir, 'sparsity.json'))
        tracking.log_artifact(os.path.join(out_dir, 'args.json'))
        tracking.log_model(model_proto, registered, code_dirs=code_dirs,
                           tags={'seed': run_seed, 'config': cfg, 'val_acc': after['metrics']['val_acc'],
                                 'test_acc': after['metrics']['test_acc'], 'path': out_dir})
        print(f'\nwritten: {out_dir}')


if __name__ == '__main__':
    main()
