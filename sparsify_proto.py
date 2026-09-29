"""Post-training sparsification of a trained LogiX-GIN run (base or prototype): Hoyer fine-tuning, then pruning.

Works on both kinds of runs: ``results_logic/...`` (base LogiX-GIN, train_logic.py) and
``results_proto/...`` (prototype variants, train_proto.py); each is fine-tuned with the
objective of the script that trained it.

Why: a conv unit of layer 1-2 needs ~13 literals to reach its threshold because its
weight mass is spread over ~35 mid-sized weights, which gives it ~1e10 minimal covers
(rules). Dropping small weights at extraction time barely helps (see min_covers.py);
the mass has to be *moved* onto few weights, which needs training.

Three phases, starting from ``<run_path>/best.pt``:

1. Hoyer fine-tuning. The training objective of train_proto.py (layer-wise
   distillation against the frozen teacher + task loss, the ``train_full`` regime)
   plus ``hoyer_reg * sum_conv (1 - Hoyer(W))`` and ``hoyer_fc * (1 - Hoyer(W_fc))``.
   Hoyer(w) = (sqrt(n) - |w|_1/|w|_2) / (sqrt(n) - 1) is 1 for a one-hot row and 0 for
   a uniform one, and scale-invariant, so it concentrates mass without shrinking it.
   It is applied per output unit (row), the level at which rules are counted. The
   notebook version (nbs/LayerWiseRules.ipynb cell 11) normalises by the total number
   of entries instead of the row length and only penalises the head.
2. Pruning. Every weight ``<= prune_eps`` is fixed to 0 through LogicalLayer's prune
   mask (``weight = sigmoid(ws) * exp(we) * prune``), so the sparsity is exact. With
   ``--max_fanin k`` only the k largest weights of each unit survive as well: a rule
   uses only a unit's nonzero weights, so no rule can then exceed k literals.
   With ``--fanin_schedule 12,8,6,4`` the fan-in is instead capped gradually: each step
   keeps the k largest weights of every unit, then trains ``--step_epochs`` epochs
   (masks frozen) so the remaining weights take over before the next cut. A unit with k
   inputs has at most C(k, k/2) minimal rules (they form an antichain), each of at most
   k literals, and reads at most k lower units. ``--fc_fanin`` sets a different final
   cap for the head.
3. Recovery. Same objective without the Hoyer term, masks frozen; the epoch with the
   best validation accuracy is kept.

``--run_path`` may itself be a sparse run (``.../sparse/<cfg>``); with ``--epochs 0``
the schedule then starts from its already Hoyer-sparsified weights.

Output: ``<run_path>/sparse/<config>/`` with ``best.pt``, ``args.json``, a link to the
run's ``data.pkl``, and ``sparsity.json`` (metrics and rule statistics before and
after). The directory is a regular run directory, so ``latent_logic.py --run_path`` and
``tests/bench_min_covers.py --run_path`` work on it unchanged. The run is also logged
to MLflow (experiment ``sparsify/<dataset>``, see utils/tracking.py) and the result is
registered as ``logix-proto-sparse-<dataset>-<level>``.

    python sparsify_proto.py --run_path "results_proto/Mutagenicity/<cfg>/<baseline cfg>/0"
    python sparsify_proto.py --run_path ... --stats_only      # rule statistics, no training
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models_proto.model import GIN
from rule_eval import logical_layers, print_stats, rule_metrics, rule_stats
from train_logic import train_epoch as train_epoch_logic
from train_proto import test_epoch, train_epoch as train_epoch_proto
from utils import tracking
from utils.evaluation import evaluate
from utils.utils import get_dataset, set_seed


def hoyer(W, eps=1e-12):
    """Row-wise Hoyer sparsity of a [out, in] matrix: 1 = one weight per row, 0 = uniform."""
    n = W.shape[-1]
    ratio = W.abs().sum(-1) / torch.sqrt((W ** 2).sum(-1) + eps)
    return (n ** 0.5 - ratio) / (n ** 0.5 - 1)


def hoyer_penalty(model, hoyer_reg, hoyer_fc):
    def f(m):
        loss = 0
        if hoyer_reg:
            loss = loss + hoyer_reg * sum((1 - hoyer(c.nn[0].weight)).mean() for c in m.convs)
        if hoyer_fc:
            loss = loss + hoyer_fc * (1 - hoyer(m.fc.weight)).mean()
        return loss
    return f


@torch.no_grad()
def prune(model, prune_eps, max_fanin=None, fc_fanin=None):
    """Fix every weight <= prune_eps to 0 via the prune mask, and with ``max_fanin`` every
    weight outside the unit's ``max_fanin`` largest (``fc_fanin`` for the head, default
    ``max_fanin``). Returns the fraction pruned."""
    pruned, total = 0, 0
    lls = logical_layers(model)
    for li, ll in enumerate(lls):
        k = (fc_fanin or max_fanin) if li == len(lls) - 1 else max_fanin
        keep = (ll.weight > prune_eps).float()
        if k is not None and k < ll.weight.shape[1]:
            top = torch.zeros_like(keep).scatter_(1, ll.weight.topk(k, dim=1).indices, 1.0)
            keep = keep * top
        ll.set_prune(ll.prune * keep)
        pruned += int((keep == 0).sum())
        total += keep.numel()
    return pruned / total


def split_run_path(run_path):
    """results_{proto,logic}/<ds>/<cfg>/<baseline cfg>/<seed> -> (stage, ds, cfg, baseline cfg, seed)."""
    parts = os.path.normpath(run_path).split(os.sep)
    for stage in ('results_proto', 'results_logic'):
        if stage in parts:
            i = parts.index(stage)
            return stage.split('_')[1], parts[i + 1], parts[i + 2], parts[i + 3], parts[i + 4]
    raise ValueError(f'{run_path} is neither under results_proto/ nor results_logic/')


def teacher_path(run_path):
    """The frozen teacher of a run: results/<ds>/<baseline cfg>/<seed>."""
    _, ds, _, base, seed = split_run_path(run_path)
    return os.path.join('results', ds, base, seed)


def full_eval(model, loaders, device):
    m = {}
    for name, loader in loaders.items():
        m.update(evaluate(model, loader, device, prefix=f'{name}_'))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='a trained seed directory holding best.pt')
    ap.add_argument('--dataset', default=None, help='default: read from run_path')
    ap.add_argument('--hoyer_reg', type=float, default=1.0, help='Hoyer weight on the conv layers')
    ap.add_argument('--hoyer_fc', type=float, default=1.0, help='Hoyer weight on the head')
    ap.add_argument('--epochs', type=int, default=300, help='Hoyer fine-tuning epochs')
    ap.add_argument('--prune_eps', type=float, default=1e-2, help='weights <= this are pruned to 0')
    ap.add_argument('--max_fanin', type=int, default=None,
                    help='keep at most this many weights per unit (caps every rule at this many literals)')
    ap.add_argument('--fanin_schedule', default=None,
                    help='comma-separated decreasing fan-in caps applied one after the other, e.g. 12,8,6,4')
    ap.add_argument('--step_epochs', type=int, default=50, help='training epochs after each schedule step')
    ap.add_argument('--fc_fanin', type=int, default=None,
                    help='final fan-in cap of the head (default: same schedule as the conv layers)')
    ap.add_argument('--recover_epochs', type=int, default=100, help='fine-tuning after pruning')
    ap.add_argument('--lr', type=float, default=None, help='default: the run\'s lr')
    ap.add_argument('--batch_size', type=int, default=None, help='default: the run\'s batch size')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None, help='default: <run_path>/sparse/<config>')
    ap.add_argument('--stats_only', action='store_true', help='print the run\'s rule statistics and exit')
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    stage, ds_name, proto_cfg, base_cfg, run_seed = split_run_path(a.run_path)
    ds_name = a.dataset or ds_name
    run_args = json.load(open(os.path.join(a.run_path, 'args.json')))
    split = pickle.load(open(os.path.join(a.run_path, 'data.pkl'), 'rb'))
    dataset = get_dataset(ds_name)
    num_classes, num_features = dataset.num_classes, dataset.num_features or 10
    bs = a.batch_size or run_args['batch_size']
    train_loader = DataLoader(dataset[split['train_indices']], batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset[split['val_indices']], batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset[split['test_indices']], batch_size=64, shuffle=False)
    loaders = {'val': val_loader, 'test': test_loader}

    model_proto = torch.load(os.path.join(a.run_path, 'best.pt'), map_location=device,
                             weights_only=False)
    before = {'metrics': full_eval(model_proto, loaders, device),
              'rules': rule_stats(model_proto, val_loader, device)}
    print_stats('before', before['metrics'], before['rules'])
    if a.stats_only:
        return

    sp_args = {k: v for k, v in vars(a).items() if k not in ('stats_only', 'out_dir', 'run_path', 'dataset')}
    schedule = [int(k) for k in a.fanin_schedule.split(',')] if a.fanin_schedule else []
    if a.fanin_schedule:
        sp_args['fanin_schedule'] = '-'.join(map(str, schedule))
    keys = ('epochs', 'hoyer_fc', 'hoyer_reg') + (('max_fanin',) if a.max_fanin else ()) \
        + (('fanin_schedule', 'step_epochs') if schedule else ()) + (('fc_fanin',) if a.fc_fanin else ()) \
        + ('prune_eps', 'recover_epochs')
    cfg = '|'.join(f'{k}={sp_args[k]}' for k in keys)
    out_dir = a.out_dir or os.path.join(a.run_path, 'sparse', cfg)
    os.makedirs(out_dir, exist_ok=True)
    level = run_args.get('proto_level', 'logic')
    registered = f'logix-proto-sparse-{ds_name}-{level}' if stage == 'proto' else f'logix-gin-sparse-{ds_name}'
    code_dirs = ('models_proto',) if stage == 'proto' else ('models',)

    with tracking.run(f'sparsify/{ds_name}', run_name=f'{level}|{cfg}/seed{run_seed}',
                      params={**sp_args, **{f'source/{k}': v for k, v in run_args.items()}},
                      tags={'config': cfg, 'source_config': f'{proto_cfg}/{base_cfg}', 'seed': run_seed, 'stage': stage,
                            'kind': 'seed', 'dataset': ds_name, 'proto_level': level,
                            'source_run_path': a.run_path, 'out_dir': out_dir}):
        tracking.log_metrics({f'before/{k}': v for k, v in before['metrics'].items()})
        tracking.log_metrics(rule_metrics(before['rules'], prefix='before/rules/'))

        tpath = teacher_path(a.run_path)
        targs = json.load(open(os.path.join(tpath, 'args.json')))
        teacher = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=targs['hidden_dim'],
                      num_layers=targs['num_layers'], nogumbel=targs['nogumbel']).to(device)
        teacher.load_state_dict(torch.load(os.path.join(tpath, 'best.pt'), map_location='cpu'))
        teacher.eval()

        common = dict(train_full=True, conv_reg=run_args['conv_reg'], fc_reg=run_args['fc_reg'])
        if stage == 'proto':
            common.update(proto_div_reg=run_args['proto_div_reg'], proto_ent_reg=run_args['proto_ent_reg'])
        train_epoch = train_epoch_proto if stage == 'proto' else train_epoch_logic
        lr = a.lr or run_args['lr']

        # 1. Hoyer fine-tuning
        opt = torch.optim.AdamW(model_proto.parameters(), lr=lr, weight_decay=run_args['l2'])
        penalty = hoyer_penalty(model_proto, a.hoyer_reg, a.hoyer_fc)
        t0 = time.time()
        for epoch in range(a.epochs):
            loss, _ = train_epoch(teacher, model_proto, train_loader, device, opt, num_classes,
                                  extra_loss=penalty, **common)
            v = test_epoch(model_proto, val_loader, device)
            with torch.no_grad():
                h = [float(hoyer(ll.weight).mean()) for ll in logical_layers(model_proto)]
            names = [f'L{i}' for i in range(len(h) - 1)] + ['head']
            tracking.log_metrics({'hoyer_phase/val_acc': v, 'hoyer_phase/train_loss': loss[-1],
                                  **{f'hoyer_phase/hoyer_{n}': x for n, x in zip(names, h)}}, step=epoch)
            if epoch % 25 == 0 or epoch == a.epochs - 1:
                print(f'hoyer epoch {epoch:4d}  val {v:.4f}  mean Hoyer per layer '
                      f'{np.round(h, 3).tolist()}  ({time.time() - t0:.0f}s)', flush=True)

        # 2. pruning, in one cut or along the fan-in schedule
        frac = prune(model_proto, a.prune_eps, a.max_fanin, a.fc_fanin if not schedule else None)
        after_prune = test_epoch(model_proto, val_loader, device)
        tracking.log_metrics({'pruned_fraction': frac, 'val_acc_after_prune': after_prune})
        print(f'\npruned {frac:.1%} of the weights at eps {a.prune_eps}: val {after_prune:.4f}')
        step = a.epochs
        for si, k in enumerate(schedule):
            fc_k = max(k, a.fc_fanin) if a.fc_fanin else None
            frac = prune(model_proto, a.prune_eps, k, fc_k)
            cut = test_epoch(model_proto, val_loader, device)
            last = si == len(schedule) - 1
            opt = torch.optim.AdamW(model_proto.parameters(), lr=lr, weight_decay=run_args['l2'])
            for epoch in range(0 if last else a.step_epochs):      # the last step trains in phase 3
                loss, _ = train_epoch(teacher, model_proto, train_loader, device, opt, num_classes, **common)
                v = test_epoch(model_proto, val_loader, device)
                tracking.log_metrics({'schedule_phase/val_acc': v, 'schedule_phase/fanin': k,
                                      'schedule_phase/train_loss': loss[-1]}, step=step)
                step += 1
            print(f'fan-in {k:3d}: pruned {frac:.1%}, val right after the cut {cut:.4f}'
                  + ('' if last else f', after {a.step_epochs} epochs {v:.4f}'), flush=True)
            tracking.log_metrics({f'schedule/k{k}_val_after_cut': cut})
        if schedule:
            after_prune = test_epoch(model_proto, val_loader, device)
            tracking.log_metrics({'pruned_fraction': frac, 'val_acc_after_prune': after_prune})

        # 3. recovery, masks frozen, keep the best validation epoch
        step = max(step, a.epochs)
        opt = torch.optim.AdamW(model_proto.parameters(), lr=lr, weight_decay=run_args['l2'])
        best = after_prune
        torch.save(model_proto, os.path.join(out_dir, 'best.pt'))
        for epoch in range(a.recover_epochs):
            loss, _ = train_epoch(teacher, model_proto, train_loader, device, opt, num_classes, **common)
            v = test_epoch(model_proto, val_loader, device)
            if v >= best:
                best = v
                torch.save(model_proto, os.path.join(out_dir, 'best.pt'))
            tracking.log_metrics({'recover_phase/val_acc': v, 'recover_phase/best_val_acc': best,
                                  'recover_phase/train_loss': loss[-1]}, step=step + epoch)
            if epoch % 25 == 0 or epoch == a.recover_epochs - 1:
                print(f'recover epoch {epoch:4d}  val {v:.4f}  best {best:.4f}', flush=True)

        model_proto = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device, weights_only=False)
        after = {'metrics': full_eval(model_proto, loaders, device),
                 'rules': rule_stats(model_proto, val_loader, device),
                 'pruned_fraction': frac, 'val_after_prune_before_recovery': after_prune}
        print_stats('before', before['metrics'], before['rules'])
        print_stats('after', after['metrics'], after['rules'])

        with open(os.path.join(out_dir, 'args.json'), 'w') as f:
            json.dump({**run_args, 'sparsify': sp_args}, f)
        link = os.path.join(out_dir, 'data.pkl')
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(a.run_path, 'data.pkl')), link)
        with open(os.path.join(out_dir, 'sparsity.json'), 'w') as f:
            json.dump({'before': before, 'after': after}, f, indent=1)

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
