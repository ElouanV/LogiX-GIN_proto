"""Gradual-hardening fine-tune of a trained prototype LogiX-GIN run, before sparsification.

The literals and conv units of a trained run are sharpened towards {0, 1}
(GINTELLProtoBase.set_sharpness, tell.sharpen; sharpness beta = 1 is the trained model)
while it keeps training on the objective that trained it (train_proto.train_epoch in the
``train_full`` regime: chained layer-wise distillation against the frozen teacher + task
loss, with the run's regularisers). beta grows geometrically from 1 to ``--harden_to``
over the first ``--ramp_frac`` of ``--epochs``, then is held; the epoch with the best
validation accuracy at beta = harden_to is kept, so the saved network is close to the rules
it is read as. No push during the fine-tune. This is the pipeline of the W41 meeting:
train -> harden_proto.py -> sparsify_proto.py (layer-wise Hoyer), in place of
``train_proto.py --harden_to``, which hardens during training (Mutagenicity NMP: fidelity
0.80 -> 0.88 but test acc -4.3 at beta 30).

Output: ``<run_path>/harden/<config>/`` with ``best.pt``, ``args.json``, a link to the run's
``data.pkl`` and ``harden.json`` (metrics before / after and the per-epoch history). It is a
regular run directory, so ``interp_metrics.py --run_path`` and ``sparsify_proto.py
--run_path`` work on it unchanged. Logged to MLflow (experiment ``harden/<dataset>``).

    python harden_proto.py --run_path "results_proto/Mutagenicity/<cfg>/<baseline cfg>/0" --harden_to 20
"""
import argparse
import os
import time

import torch

from train_proto import loss_weights, train_epoch
from utils import tracking
from utils.evaluation import full_eval
from utils.optim import build_optimizer
from utils.runs import RunDir, load_run
from utils.training import finetune_run, test_epoch
from utils.utils import set_seed


def ramp(epoch, epochs, harden_to, ramp_frac):
    """Sharpness at ``epoch``: 1 at epoch 0, geometric up to harden_to at ramp_frac * epochs, then held."""
    e1 = max(1.0, ramp_frac * epochs)
    return float(harden_to ** min(1.0, epoch / e1))


def harden(teacher, model, run_args, train_loader, val_loader, device, num_classes, harden_to, epochs, ramp_frac,
           optimizer, out_path, log=lambda metrics, step: None):
    """Fine-tune ``model`` along the sharpness ramp; saves the best val epoch at beta = harden_to
    to ``out_path`` and returns the per-epoch history."""
    common = dict(train_full=True, **loss_weights(run_args))
    history, best, t0 = [], -1.0, time.time()
    for epoch in range(epochs):
        beta = ramp(epoch, epochs, harden_to, ramp_frac)
        model.set_sharpness(beta)
        loss, _ = train_epoch(teacher, model, train_loader, device, optimizer, num_classes, **common)
        val_loss, _ = train_epoch(teacher, model, val_loader, device, None, num_classes, **common)
        val = test_epoch(model, val_loader, device)
        if beta >= harden_to - 1e-9 and val >= best:
            best = val
            torch.save(model, out_path)
        history.append({'epoch': epoch, 'sharpness': beta, 'val_acc': val, 'train_loss': loss[-1], 'val_loss': val_loss[-1]})
        log({'harden/val_acc': val, 'harden/sharpness': beta, 'harden/train_loss': loss[-1], 'harden/val_loss': val_loss[-1],
             'harden/best_val_acc': max(best, 0.0)}, epoch)
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f'epoch {epoch:4d}  beta {beta:6.2f}  loss {loss[-1]:.4f}  val {val:.4f}  best@B {best:.4f}  '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='a trained prototype seed directory holding best.pt')
    ap.add_argument('--dataset', default=None, help='default: read from run_path')
    ap.add_argument('--harden_to', type=float, default=20.0, help='final sharpness B (1 = the trained model)')
    ap.add_argument('--epochs', type=int, default=300, help='fine-tuning epochs')
    ap.add_argument('--ramp_frac', type=float, default=2 / 3, help='fraction of the epochs over which beta goes 1 -> B')
    ap.add_argument('--lr', type=float, default=None, help="default: the run's lr")
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'], help='see utils/optim.py')
    ap.add_argument('--muon_lr', type=float, default=0.02, help='with --optimizer muon')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None, help='default: <run_path>/harden/<config>')
    a = ap.parse_args()
    if a.harden_to < 1:
        ap.error('--harden_to must be >= 1')

    set_seed(a.seed)
    if RunDir(a.run_path).stage != 'proto':
        raise SystemExit('needs a prototype run')
    loaded = load_run(a.run_path, a.dataset, teacher=True)
    run, model, device, loaders = loaded.run, loaded.model, loaded.device, loaded.loaders
    run_args = run.args
    eval_loaders = {'val': loaders['val'], 'test': loaders['test']}
    before = full_eval(model, eval_loaders, device)

    h_args = {'harden_to': a.harden_to, 'epochs': a.epochs, 'ramp_frac': round(a.ramp_frac, 4),
              'lr': a.lr or run_args['lr'], 'optimizer': a.optimizer, 'seed': a.seed}
    keys = ['harden_to', 'epochs', 'ramp_frac'] + (['lr'] if a.lr else []) + (['optimizer'] if a.optimizer != 'adamw' else [])
    if a.optimizer == 'muon':
        h_args['muon_lr'] = a.muon_lr
        keys.append('muon_lr')
    cfg = '|'.join(f'{k}={h_args[k]}' for k in keys)
    out_dir = a.out_dir or os.path.join(a.run_path, 'harden', cfg)
    os.makedirs(out_dir, exist_ok=True)
    optimizer = build_optimizer(model, a.optimizer, h_args['lr'], run_args['l2'], muon_lr=a.muon_lr)

    with finetune_run(run, 'harden', 'finetune/harden', cfg, h_args, out_dir, run_args['proto_level']):
        tracking.log_metrics({f'before/{k}': v for k, v in before.items()})
        history = harden(loaded.teacher, model, run_args, loaders['train'], loaders['val'], device, loaded.num_classes,
                         a.harden_to, a.epochs, a.ramp_frac, optimizer, os.path.join(out_dir, 'best.pt'),
                         log=tracking.log_metrics)
        model = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device, weights_only=False)
        tracking.write_pipeline(out_dir)
        after = full_eval(model, eval_loaders, device)
        print('before', {k: round(v, 4) for k, v in before.items() if 'acc' in k or 'auc' in k})
        print('after ', {k: round(v, 4) for k, v in after.items() if 'acc' in k or 'auc' in k})
        path = run.write_finetune(out_dir, 'harden', h_args, {'before': before, 'after': after, 'history': history})
        tracking.log_metrics({f'after/{k}': v for k, v in after.items()})
        tracking.log_artifact(path)
        print(f'\nwritten: {out_dir}')


if __name__ == '__main__':
    main()
