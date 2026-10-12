"""Quantization-aware fine-tune (QAT) of a trained prototype LogiX-GIN run, by Quant-Noise.

The model is read as rules by binarising, at 0.5, every literal (phi_in of every conv unit
and of the head) and every conv unit output (interp_metrics.trunk_states). The gap between
the network and that rule model is the infidelity; the fidelity breakdown (2026-10-12)
puts most of it in the trunk. Two earlier ways to close it cost ~11 points of accuracy on
Mutagenicity NMP: binarising everything at once with straight-through gradients (the
``--hard`` fine-tune, fidelity 1, acc 0.70) and sharpening every sigmoid gradually
(harden_proto.py, fidelity 0.88, acc 0.70), whose gradients vanish as the sigmoids get steep.

Here the binarisation is applied to a random share ``rate`` of the bits at every forward
pass (Quant-Noise, Fan et al., ICLR 2021, arXiv:2004.07320): the binarised bits pass
straight-through gradients (Bengio et al. 2013, arXiv:1308.3432), the others keep their
exact soft gradients, and ``rate`` grows linearly from 0 to ``--rate_to`` over
``--ramp_frac`` of the epochs (a progressive schedule, as in Incremental Network
Quantization, Zhou et al., ICLR 2017, arXiv:1702.03044), then is held. Unit outputs are
binarised where the next layer reads them, never inside a layer's own distillation loss
(train_proto.train_epoch ``quant``), as QAT quantises activations on the forward path while
the loss stays on real values. The objective is the run's own (train_full distillation
against the frozen teacher + task loss + its regularisers). The epoch with the best
validation accuracy of the rule model (every bit binarised) at rate = rate_to is kept;
saved with rate_to = 1, the network computes exactly its rules. See docs/QAT.md.

Output: ``<run_path>/qat/<config>/`` (best.pt, args.json with 'qat', a link to data.pkl,
qat.json with before / after / per-epoch history), a regular run directory for
interp_metrics.py / sparsify_proto.py. MLflow experiment ``qat/<dataset>``, per epoch:
qat/rate, qat/train_loss, qat/val_acc_soft (every bit soft), qat/val_acc_rules (every bit
binarised = the rule model), qat/val_fidelity (soft network vs rule model),
qat/val_flip_L<l> (share of conv-l unit states that differ between the two),
qat/grad_L<l>, qat/grad_head (gradient norm of the weight logits, last batch).

    python qat_proto.py --run_path "results_proto/Mutagenicity/<cfg>/<baseline cfg>/0"
"""
import argparse
import os
import time

import torch

from harden_proto import load_run, write_run
from interp_metrics import symbolic_head
from sparsify_proto import full_eval
from train_proto import train_epoch
from utils import tracking
from utils.optim import build_optimizer
from utils.utils import set_seed


def ramp_rate(epoch, epochs, rate_to, ramp_frac):
    """Quant-Noise rate at ``epoch``: 0 at epoch 0, linear up to rate_to at ramp_frac * epochs, then held."""
    e1 = max(1.0, ramp_frac * epochs)
    return float(rate_to * min(1.0, epoch / e1))


@torch.no_grad()
def _trunk(model, data, rate):
    """Conv unit states and class scores with every bit soft (rate 0: the network's output) or
    binarised (rate 1: the rule margins W.lit + b, as interp_metrics.symbolic_head; the graded
    head output can saturate to ties)."""
    model.set_quant_noise(rate)
    x, xs = data.x.float(), []
    for conv in model.convs:
        x = model.quant_out(conv(torch.hstack([x, 1 - x]), data.edge_index))
        xs.append(x)
    if rate >= 1:
        return xs, symbolic_head(model, xs, data.batch)[0]
    return xs, model.head(model.readout(xs, data.batch))


@torch.no_grad()
def diagnostics(model, loader, device):
    """Soft network vs rule model on a split: accuracies, agreement, per-layer unit-state flips."""
    was_training, rate = model.training, getattr(model, 'quant_rate', 0.0)
    model.eval()
    n = n_soft = n_rules = n_agree = 0
    flips, cells = [0.0] * len(model.convs), [0] * len(model.convs)
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features), device=device)
        y = data.y.reshape(-1).long()
        xs_s, out_s = _trunk(model, data, 0.0)
        xs_r, out_r = _trunk(model, data, 1.0)
        p_s, p_r = out_s.argmax(-1), out_r.argmax(-1)
        n += len(y)
        n_soft += (p_s == y).sum().item()
        n_rules += (p_r == y).sum().item()
        n_agree += (p_s == p_r).sum().item()
        for i, (a, b) in enumerate(zip(xs_s, xs_r)):
            flips[i] += ((a >= 0.5).float() != b).sum().item()
            cells[i] += a.numel()
    model.set_quant_noise(rate)
    model.train(was_training)
    out = {'val_acc_soft': n_soft / n, 'val_acc_rules': n_rules / n, 'val_fidelity': n_agree / n}
    out.update({f'val_flip_L{i}': f / max(c, 1) for i, (f, c) in enumerate(zip(flips, cells))})
    return out


def grad_norms(model):
    """Gradient norm of each weight-logit matrix after the last batch (vanishing gradients
    under binarisation show up here)."""
    g = {f'grad_L{i}': c.nn[0].weight_sigma.grad for i, c in enumerate(model.convs)}
    g['grad_head'] = model.fc.weight_sigma.grad
    return {k: float(v.norm()) if v is not None else 0.0 for k, v in g.items()}


def qat(teacher, model, run_args, train_loader, val_loader, device, num_classes, rate_to, epochs, ramp_frac,
        optimizer, out_path, log=lambda metrics, step: None):
    """Fine-tune ``model`` along the Quant-Noise ramp; saves the best rule-model val epoch at
    rate = rate_to to ``out_path`` and returns the per-epoch history."""
    common = dict(train_full=True, conv_reg=run_args['conv_reg'], fc_reg=run_args['fc_reg'],
                  proto_div_reg=run_args['proto_div_reg'], proto_ent_reg=run_args['proto_ent_reg'],
                  mask_reg=run_args.get('mask_reg', 0.0), vocab_reg=run_args.get('vocab_reg') or 0.0)
    history, best, t0 = [], -1.0, time.time()
    for epoch in range(epochs):
        rate = ramp_rate(epoch, epochs, rate_to, ramp_frac)
        model.set_quant_noise(rate)
        loss, _ = train_epoch(teacher, model, train_loader, device, optimizer, num_classes, quant=model.quant_out,
                              **common)
        rec = {'epoch': epoch, 'rate': rate, 'train_loss': loss[-1], **grad_norms(model),
               **diagnostics(model, val_loader, device)}
        if rate >= rate_to - 1e-9 and rec['val_acc_rules'] >= best:
            best = rec['val_acc_rules']
            torch.save(model, out_path)
        elif epoch == epochs - 1 and best < 0:             # the ramp never reached rate_to
            print(f'warning: rate_to not reached, saving the last epoch (rate {rate:.3f})', flush=True)
            torch.save(model, out_path)
        history.append(rec)
        log({f'qat/{k}': v for k, v in rec.items() if k != 'epoch'} | {'qat/best_val_acc_rules': max(best, 0.0)}, epoch)
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  rate {rate:.3f}  loss {loss[-1]:.4f}  val soft {rec['val_acc_soft']:.4f}  "
                  f"rules {rec['val_acc_rules']:.4f}  fid {rec['val_fidelity']:.4f}  best@R {best:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return history


def evaluate(model, loaders, device):
    """The network with every bit soft (full_eval: acc, balanced acc, AUC) and the rule model
    (accuracy from the rule margins), with their agreement and per-layer flips, val and test.
    interp_metrics.py gives the full interpretability report of the saved run."""
    rate = getattr(model, 'quant_rate', 0.0)
    model.set_quant_noise(0.0)
    res = {'soft': full_eval(model, loaders, device), 'rules': {}}
    for split in ('val', 'test'):
        res['rules'].update({k.replace('val_', f'{split}_', 1): v
                             for k, v in diagnostics(model, loaders[split], device).items()})
    model.set_quant_noise(rate)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='a trained prototype seed directory holding best.pt')
    ap.add_argument('--dataset', default=None, help='default: read from run_path')
    ap.add_argument('--rate_to', type=float, default=1.0, help='final share of binarised bits (1 = the rule model)')
    ap.add_argument('--epochs', type=int, default=300, help='fine-tuning epochs')
    ap.add_argument('--ramp_frac', type=float, default=2 / 3, help='fraction of the epochs over which rate goes 0 -> rate_to')
    ap.add_argument('--lr', type=float, default=None, help="default: the run's lr")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None, help='default: <run_path>/qat/<config>')
    a = ap.parse_args()
    if not 0 < a.rate_to <= 1:
        ap.error('--rate_to must be in (0, 1]')

    set_seed(a.seed)
    run = load_run(a.run_path, a.dataset)
    run_args, model, device, loaders = run['args'], run['model'], run['device'], run['loaders']
    eval_loaders = {'val': loaders['val'], 'test': loaders['test']}
    before = evaluate(model, eval_loaders, device)

    q_args = {'method': 'quant_noise', 'rate_to': a.rate_to, 'epochs': a.epochs, 'ramp_frac': round(a.ramp_frac, 4),
              'lr': a.lr or run_args['lr'], 'seed': a.seed}
    keys = ['rate_to', 'epochs', 'ramp_frac'] + (['lr'] if a.lr else [])
    cfg = '|'.join(f'{k}={q_args[k]}' for k in keys)
    out_dir = a.out_dir or os.path.join(a.run_path, 'qat', cfg)
    os.makedirs(out_dir, exist_ok=True)
    optimizer = build_optimizer(model, 'adamw', q_args['lr'], run_args['l2'])

    with tracking.run(f"qat/{run['dataset']}", run_name=f"{run_args['proto_level']}|{cfg}/seed{run['seed']}",
                      params={**q_args, **{f'source/{k}': v for k, v in run_args.items()}},
                      tags={'config': cfg, 'source_config': f"{run['proto_cfg']}/{run['base_cfg']}", 'seed': run['seed'],
                            'kind': 'seed', 'dataset': run['dataset'], 'proto_level': run_args['proto_level'],
                            'source_run_path': a.run_path, 'out_dir': out_dir}):
        tracking.log_metrics({f'before/{m}/{k}': v for m, r in before.items() for k, v in r.items()})
        history = qat(run['teacher'], model, run_args, loaders['train'], loaders['val'], device, run['num_classes'],
                      a.rate_to, a.epochs, a.ramp_frac, optimizer, os.path.join(out_dir, 'best.pt'),
                      log=tracking.log_metrics)
        model = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device, weights_only=False)
        after = evaluate(model, eval_loaders, device)
        for name, res in (('before', before), ('after', after)):
            for m, r in res.items():
                print(f'{name:6s} {m:5s}', {k: round(v, 4) for k, v in r.items() if k.startswith('test_')})
        path = write_run(out_dir, run, 'qat', q_args, {'before': before, 'after': after, 'history': history})
        tracking.log_metrics({f'after/{m}/{k}': v for m, r in after.items() for k, v in r.items()})
        tracking.log_artifact(path)
        print(f'\nwritten: {out_dir}')


if __name__ == '__main__':
    main()
