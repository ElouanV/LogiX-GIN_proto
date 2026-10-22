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
the loss stays on real values. The objective is ``--base_weight`` x the run's own
(train_full distillation against the frozen teacher + task loss + its regularisers), plus
optional terms on the rule model itself (rule_losses): ``--task_weight`` (cross-entropy of
the rule margins) and ``--distill_weight`` (self-distillation from the frozen starting
student). The epoch with the best validation accuracy of the rule model (every bit
binarised; ``--select_metric balanced_acc``: its balanced accuracy) at rate = rate_to is kept,
or the starting model if none beats it; saved with rate_to = 1, the network computes exactly
its rules. See docs/QAT.md.

Output: ``<run_path>/qat/<config>/`` (best.pt, args.json with 'qat', a link to data.pkl,
qat.json with before / after / per-epoch history), a regular run directory for
interp_metrics.py / sparsify_proto.py. MLflow experiment ``qat/<dataset>``, per epoch:
qat/rate, qat/train_loss, qat/val_acc_soft (every bit soft), qat/val_acc_rules (every bit
binarised = the rule model), qat/val_balanced_acc_rules, qat/val_fidelity (soft network vs rule model),
qat/val_flip_L<l> (share of conv-l unit states that differ between the two),
qat/grad_L<l>, qat/grad_head (gradient norm of the weight logits, last batch).

    python qat_proto.py --run_path "results_proto/Mutagenicity/<cfg>/<baseline cfg>/0"
"""
import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from interp_metrics import symbolic_head
from train_proto import loss_weights, train_epoch
from utils import tracking
from utils.evaluation import full_eval
from utils.optim import build_optimizer
from utils.runs import RunDir, load_run
from utils.training import finetune_run
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
    ys, rules = [], []
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
        ys.append(y.cpu())
        rules.append(p_r.cpu())
        for i, (a, b) in enumerate(zip(xs_s, xs_r)):
            flips[i] += ((a >= 0.5).float() != b).sum().item()
            cells[i] += a.numel()
    model.set_quant_noise(rate)
    model.train(was_training)
    out = {'val_acc_soft': n_soft / n, 'val_acc_rules': n_rules / n, 'val_fidelity': n_agree / n,
           'val_balanced_acc_rules': float(balanced_accuracy_score(torch.cat(ys), torch.cat(rules)))}
    out.update({f'val_flip_L{i}': f / max(c, 1) for i, (f, c) in enumerate(zip(flips, cells))})
    return out


def grad_norms(model):
    """Gradient norm of each weight-logit matrix after the last batch (vanishing gradients
    under binarisation show up here)."""
    g = {f'grad_L{i}': c.nn[0].weight_sigma.grad for i, c in enumerate(model.convs)}
    g['grad_head'] = model.fc.weight_sigma.grad
    return {k: float(v.norm()) if v is not None else 0.0 for k, v in g.items()}


def rule_losses(model, frozen, task_weight, distill_weight, T=1.0):
    """Extra loss on the rule model itself, whatever the current Quant-Noise rate: the
    forward pass at rate 1 (every bit binarised, straight-through gradients) read as the rule
    margins W.lit + b. task_weight: cross-entropy of the margins against the labels (the run's
    own task term is on the graded head output at the current rate, among six distillation
    terms). distill_weight: self-distillation (Polino et al., ICLR 2018, arXiv:1802.05668),
    T^2 KL from the frozen starting student's soft margins (every bit soft) to the rule
    margins, at temperature T. Returns a callable for train_proto.train_epoch(extra_loss=)."""
    def extra(x, edge_index, batch, y):
        if not (task_weight or distill_weight):
            return 0.0
        rate = model.quant_rate
        model.set_quant_noise(1.0)
        m = model(x, edge_index, batch, margin=True)
        model.set_quant_noise(rate)
        loss = task_weight * F.cross_entropy(m, y) if task_weight else 0.0
        if distill_weight:
            with torch.no_grad():
                t = frozen(x, edge_index, batch, margin=True)
            loss = loss + distill_weight * T * T * F.kl_div(F.log_softmax(m / T, -1), F.log_softmax(t / T, -1),
                                                            log_target=True, reduction='batchmean')
        return loss
    return extra


def qat(teacher, model, run_args, train_loader, val_loader, device, num_classes, rate_to, epochs, ramp_frac,
        optimizer, out_path, base_weight=1.0, task_weight=0.0, distill_weight=0.0, distill_T=1.0,
        select_metric='acc', log=lambda metrics, step: None):
    """Fine-tune ``model`` along the Quant-Noise ramp and return the per-epoch history.

    Saved to ``out_path``: the epoch with the best rule-model val ``select_metric`` (acc or
    balanced_acc) at rate = rate_to,
    or the starting model (epoch -1, saved at rate_to) when no QAT epoch beats its rules, so
    the fine-tune never returns a worse rule model than the one it starts from on val.
    Objective: base_weight x the run's own (train_full) + rule_losses(task_weight, distill_weight)."""
    common = dict(train_full=True, **loss_weights(run_args))
    frozen = copy.deepcopy(model).eval()
    frozen.set_quant_noise(0.0)
    for p in frozen.parameters():
        p.requires_grad_(False)
    extra = rule_losses(model, frozen, task_weight, distill_weight, distill_T)

    key = {'acc': 'val_acc_rules', 'balanced_acc': 'val_balanced_acc_rules'}[select_metric]
    start = diagnostics(model, val_loader, device)
    best, selected = start[key], -1
    model.set_quant_noise(rate_to)
    torch.save(model, out_path)
    history, qat_saved, t0 = [], False, time.time()
    for epoch in range(epochs):
        rate = ramp_rate(epoch, epochs, rate_to, ramp_frac)
        model.set_quant_noise(rate)
        loss, _ = train_epoch(teacher, model, train_loader, device, optimizer, num_classes, quant=model.quant_out,
                              base_weight=base_weight, extra_loss=extra, **common)
        grads = grad_norms(model)
        val_loss, _ = train_epoch(teacher, model, val_loader, device, None, num_classes, quant=model.quant_out,
                                  base_weight=base_weight, extra_loss=extra, **common)
        rec = {'epoch': epoch, 'rate': rate, 'train_loss': loss[-1], 'val_loss': val_loss[-1], **grads,
               **diagnostics(model, val_loader, device)}
        report = epoch % 25 == 0 or epoch == epochs - 1
        if report:
            rec['train_acc_rules'] = diagnostics(model, train_loader, device)['val_acc_rules']    # under- vs over-fit
        if rate >= rate_to - 1e-9 and rec[key] >= best:
            best, selected, qat_saved = rec[key], epoch, True
            torch.save(model, out_path)
        elif epoch == epochs - 1 and rate < rate_to - 1e-9 and not qat_saved:
            print(f'warning: rate_to not reached (rate {rate:.3f}), the starting model is kept', flush=True)
        history.append(rec)
        log({f'qat/{k}': v for k, v in rec.items() if k != 'epoch'}
            | {f'qat/best_{key}': best, 'qat/selected_epoch': selected}, epoch)
        if report:
            print(f"epoch {epoch:4d}  rate {rate:.3f}  loss {loss[-1]:.4f}  val soft {rec['val_acc_soft']:.4f}  "
                  f"rules {rec['val_acc_rules']:.4f} (train {rec['train_acc_rules']:.4f})  fid {rec['val_fidelity']:.4f}  "
                  f"best {best:.4f}@{selected}  ({time.time() - t0:.0f}s)", flush=True)
    return {'start_val': start, 'selected_epoch': selected, 'history': history}


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


def run_qat(run_path, dataset=None, rate_to=1.0, epochs=300, ramp_frac=2 / 3, lr=None, base_weight=1.0,
            task_weight=0.0, distill_weight=0.0, distill_T=1.0, seed=0, out_dir=None, tags=None, select_metric='acc'):
    """QAT fine-tune of the trained run at ``run_path`` (see the module docstring). Writes a
    regular run directory (default ``<run_path>/qat/<config>``) and an MLflow run in
    ``qat/<dataset>``; returns (out_dir, record with before / after / start_val /
    selected_epoch / history). Used by main() and by utils/hps.py / utils/experiment.py
    for models with a ``qat`` spec (configs/models.yaml)."""
    if not 0 < rate_to <= 1:
        raise ValueError('rate_to must be in (0, 1]')
    set_seed(seed)
    if RunDir(run_path).stage != 'proto':
        raise SystemExit('needs a prototype run')
    loaded = load_run(run_path, dataset, teacher=True)
    run, model, device, loaders = loaded.run, loaded.model, loaded.device, loaded.loaders
    run_args = run.args
    eval_loaders = {'val': loaders['val'], 'test': loaders['test']}
    before = evaluate(model, eval_loaders, device)

    q_args = {'method': 'quant_noise', 'rate_to': rate_to, 'epochs': epochs, 'ramp_frac': round(ramp_frac, 4),
              'lr': lr or run_args['lr'], 'base_weight': base_weight, 'task_weight': task_weight,
              'distill_weight': distill_weight, 'distill_T': distill_T, 'select': 'keep_start', 'seed': seed,
              'select_metric': select_metric}
    # step-1 runs (before rule_losses / keep_start) have none of the last keys in their directory name
    keys = ['rate_to', 'epochs', 'ramp_frac'] + (['lr'] if lr else []) + ['base_weight', 'task_weight', 'distill_weight'] \
        + (['distill_T'] if distill_weight and distill_T != 1 else []) + ['select'] \
        + (['select_metric'] if select_metric != 'acc' else [])
    cfg = '|'.join(f'{k}={q_args[k]}' for k in keys)
    out_dir = out_dir or os.path.join(run_path, 'qat', cfg)
    os.makedirs(out_dir, exist_ok=True)
    optimizer = build_optimizer(model, 'adamw', q_args['lr'], run_args['l2'])

    with finetune_run(run, 'qat', 'finetune/qat', cfg, q_args, out_dir, run_args['proto_level'], tags):
        tracking.log_metrics({f'before/{m}/{k}': v for m, r in before.items() for k, v in r.items()})
        fit = qat(loaded.teacher, model, run_args, loaders['train'], loaders['val'], device, loaded.num_classes,
                  rate_to, epochs, ramp_frac, optimizer, os.path.join(out_dir, 'best.pt'),
                  base_weight=base_weight, task_weight=task_weight, distill_weight=distill_weight,
                  distill_T=distill_T, select_metric=select_metric, log=tracking.log_metrics)
        model = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device, weights_only=False)
        tracking.write_pipeline(out_dir)
        after = evaluate(model, eval_loaders, device)
        for name, res in (('before', before), ('after', after)):
            for m, r in res.items():
                print(f'{name:6s} {m:5s}', {k: round(v, 4) for k, v in r.items() if k.startswith('test_')})
        print(f"selected epoch {fit['selected_epoch']} (-1 = the starting model)")
        record = {'before': before, 'after': after, **fit}
        path = run.write_finetune(out_dir, 'qat', q_args, record)
        tracking.log_metrics({f'after/{m}/{k}': v for m, r in after.items() for k, v in r.items()})
        tracking.log_artifact(path)
        print(f'\nwritten: {out_dir}')
    return out_dir, record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run_path', required=True, help='a trained prototype seed directory holding best.pt')
    ap.add_argument('--dataset', default=None, help='default: read from run_path')
    ap.add_argument('--rate_to', type=float, default=1.0, help='final share of binarised bits (1 = the rule model)')
    ap.add_argument('--epochs', type=int, default=300, help='fine-tuning epochs')
    ap.add_argument('--ramp_frac', type=float, default=2 / 3, help='fraction of the epochs over which rate goes 0 -> rate_to')
    ap.add_argument('--lr', type=float, default=None, help="default: the run's lr")
    ap.add_argument('--base_weight', type=float, default=1.0, help="weight of the run's own objective (0 = drop it)")
    ap.add_argument('--task_weight', type=float, default=0.0, help='cross-entropy of the rule margins (rate 1) vs the labels')
    ap.add_argument('--distill_weight', type=float, default=0.0,
                    help="self-distillation: KL from the frozen starting student's soft margins to the rule margins")
    ap.add_argument('--distill_T', type=float, default=1.0, help='self-distillation temperature')
    ap.add_argument('--select_metric', default='acc', choices=['acc', 'balanced_acc'],
                    help='rule-model val score the kept epoch is chosen on (default acc)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None, help='default: <run_path>/qat/<config>')
    a = ap.parse_args()
    if not 0 < a.rate_to <= 1:
        ap.error('--rate_to must be in (0, 1]')
    run_qat(a.run_path, a.dataset, a.rate_to, a.epochs, a.ramp_frac, a.lr, a.base_weight, a.task_weight,
            a.distill_weight, a.distill_T, a.seed, a.out_dir, select_metric=a.select_metric)


if __name__ == '__main__':
    main()
