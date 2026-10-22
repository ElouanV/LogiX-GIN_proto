"""Layer-wise distillation of a frozen GIN teacher into a LogiX-GIN student (GINTELL):
per-layer BCE on the trunk against the teacher's binary node states, warmup then full
end-to-end training, ReduceLROnPlateau, one run per seed (upstream training, unchanged).

--head_init center and --logit_loss fix the head saturation (sigmoid(10 z) = exactly 1 at the
random init, so no task gradient and a constant output on small datasets): the head bias
is centred on the training set before the first step (utils.training.center_head) and the
task BCE is computed from the logits (utils.training.class_loss). Off by default
(upstream); they only enter the results path when set.

    python train_logic.py --dataset MUTAG --baseline_path "results/MUTAG/<teacher cfg>" [--seed 0]

Results go to results_logic/<dataset>/<cfg>/<teacher cfg>/<seed> (utils/runs.py).
"""
import argparse
import json
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from models.model import GINTELL
from models_proto.gintell import GINTELLPool
from utils import tracking
from utils.evaluation import held_out
from utils.runs import EVAL_BATCH_SIZE, LOGIC_ROOT, best_teacher_dir, get_device, student_dir
from utils.training import (center_head, class_loss, distill_trunk, eval_student, fresh_dir, log_summary, register_final,
                            run_seeds, student_data, test_epoch, tracked_student_seed)
from utils.utils import set_seed, zero_nan_gradients


def run_dir(dataset_name, args, baseline_args, seed=None, root=LOGIC_ROOT):
    return student_dir(dataset_name, args, baseline_args, seed=seed, root=root)


def build_logic_model(args, baseline_args, num_features, num_classes, device):
    """Upstream GINTELL, or GINTELLPool when --pool_ops picks another readout."""
    kwargs = dict(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'],
                  num_layers=baseline_args['num_layers'])
    if args.get('pool_ops'):
        return GINTELLPool(**kwargs, pool_ops=tuple(args['pool_ops'].split(','))).to(device)
    return GINTELL(**kwargs).to(device)


def train_epoch(model, model_tell, loader, device, optimizer, num_classes, train_full=True, conv_reg=0.001, fc_reg=0.01,
                logit_loss=False):
    """One distillation epoch; optimizer=None: the same loss on loader without training (the
    val loss): student in eval mode and without input dropout, teacher sampled as in
    training, no gradient, RNG untouched (utils.evaluation.held_out)."""
    if optimizer is None:
        with held_out():
            return _epoch(model, model_tell, loader, device, None, num_classes, train_full, conv_reg, fc_reg, logit_loss)
    return _epoch(model, model_tell, loader, device, optimizer, num_classes, train_full, conv_reg, fc_reg, logit_loss)


def head_input(model_tell, layers_x):
    """What the head reads in the warmup: the teacher's pooled readout (mean | max | sum,
    or the columns this model's readout keeps) and its negation."""
    pooled = layers_x[-1]
    if hasattr(model_tell, 'select_pooled'):
        pooled = model_tell.select_pooled(pooled)
    return torch.hstack([pooled, 1 - pooled])


def center_head_bias(teacher, model_tell, loader, device):
    """--head_init center (utils.training.center_head) on the teacher's pooled readout."""
    return center_head(teacher, model_tell.fc, loader, device, lambda layers_x, layers_y, batch: head_input(model_tell, layers_x))


def _epoch(model, model_tell, loader, device, optimizer, num_classes, train_full, conv_reg, fc_reg, logit_loss):
    train = optimizer is not None
    model.train()
    model_tell.train(train)

    total_loss = [0]*(len(model_tell.convs)+1)
    total_correct = [0]*(len(model_tell.convs)+1)

    for data in loader:
        try:
            if data.x is None:
                data.x = torch.ones((data.num_nodes, model.num_features))
            if data.y.numel() == 0: continue
            if data.x.isnan().any(): continue
            if data.y.isnan().any(): continue
            y = data.y.reshape(-1).to(device).long()
            if train:
                optimizer.zero_grad()
            with torch.no_grad():
                layers_x, layers_y = model.forward_e(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))
            loss = distill_trunk(model_tell, layers_x, layers_y, data.edge_index.to(device), train, train_full, conv_reg,
                                 total_loss, total_correct, len(loader.dataset), len(loader))

            out = model_tell.fc(head_input(model_tell, layers_x))
            pred = out.argmax(-1)
            loss += class_loss(out, y, num_classes, model_tell.fc.logits if logit_loss else None)

            if train_full:
                model_tell.fc.phi_in.tau = 10
                out = model_tell(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))
                pred = out.argmax(-1)
                loss += class_loss(out, y, num_classes, model_tell.fc.logits if logit_loss else None)

            loss += fc_reg*(model_tell.fc.reg_loss + model_tell.fc.phi_in.entropy)

            if train:
                loss.backward()
                zero_nan_gradients(model_tell)
                optimizer.step()
            total_loss[-1] += loss.item() * data.num_graphs / len(loader.dataset)
            total_correct[-1] += pred.eq(y).sum().item() / len(loader.dataset)
        except Exception as e:
            print(e)

    return total_loss, total_correct


def train_seed(dataset_name, baseline_path, args, seed, device, out_root=None, epoch_callback=None, tags=None):
    """One seed, logged as one MLflow run of experiment logic/<dataset> (utils/tracking.py).

    ``out_root`` replaces the results root (e.g. a hyper-parameter study's directory) and
    ``epoch_callback(epoch, model, val_loader)`` is called after every epoch (optimize_optuna.py
    reports to Optuna and prunes there); ``tags`` are added to the MLflow run.
    """
    root = out_root or LOGIC_ROOT
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = run_dir(dataset_name, args, baseline_args, seed, root)
    return tracked_student_seed(
        dataset_name, baseline_path, args, seed, path, root, f'logic/{dataset_name}',
        f"logic/h{baseline_args['hidden_dim']}/seed{seed}", {'hidden_dim': baseline_args['hidden_dim'], **(tags or {})},
        lambda: _train_seed(dataset_name, baseline_path, args, seed, device, root, epoch_callback))


def _train_seed(dataset_name, baseline_path, args, seed, device, root=LOGIC_ROOT, epoch_callback=None):
    set_seed(seed)
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = fresh_dir(lambda: run_dir(dataset_name, args, baseline_args, seed, root))
    d = student_data(dataset_name, baseline_path, args, device, f'Training logic model on {dataset_name}', path)
    args, model = d.args, d.teacher
    train_loader, val_loader, test_loader = d.loaders['train'], d.loaders['val'], d.loaders['test']
    model_tell = build_logic_model(args, baseline_args, d.num_features, d.num_classes, device)

    optimizer = torch.optim.AdamW(model_tell.parameters(), lr=args['lr'], weight_decay=args['l2'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=300, min_lr=1e-5, verbose=True)

    if args.get('head_init') == 'center':
        push = DataLoader(d.datasets['train'], batch_size=EVAL_BATCH_SIZE, shuffle=False)
        print('head bias centred, shift', center_head_bias(model, model_tell, push, device).tolist())
    loss_args = dict(conv_reg=args['conv_reg'], fc_reg=args['fc_reg'], logit_loss=args.get('logit_loss', False))
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(args['epochs']):
        full = epoch > args['warmup_epochs']
        train_loss, train_acc = train_epoch(model, model_tell, train_loader, device, optimizer, d.num_classes,
                                            train_full=full, **loss_args)
        val_loss, _ = train_epoch(model, model_tell, val_loader, device, None, d.num_classes, train_full=full, **loss_args)
        val_acc = test_epoch(model_tell, val_loader, device)
        test_acc = test_epoch(model_tell, test_loader, device)

        if full:
            scheduler.step(val_acc)
        if full and val_acc >= best_val_acc:
            torch.save(model_tell, os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc

        # train_loss / train_acc: one entry per conv layer (distillation), then the head
        tracking.log_metrics({'val_acc': val_acc, 'test_acc': test_acc, 'best_val_acc': best_val_acc,
                              'best_test_acc': best_test_acc, 'train_loss': train_loss[-1], 'val_loss': val_loss[-1],
                              'train_acc': train_acc[-1], 'lr': optimizer.param_groups[0]['lr'],
                              'phase_full': float(full),
                              **{f'distill/L{i}_loss': l for i, l in enumerate(train_loss[:-1])},
                              **{f'distill/L{i}_val_loss': l for i, l in enumerate(val_loss[:-1])},
                              **{f'distill/L{i}_acc': c for i, c in enumerate(train_acc[:-1])}},
                             step=epoch)
        if epoch_callback is not None:
            epoch_callback(epoch, model_tell, val_loader)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

    torch.save(model_tell, os.path.join(path, 'last.pt'))
    model_tell = torch.load(os.path.join(path, 'best.pt'))
    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    # final evaluation of the kept checkpoint (imbalance-aware metrics, rule statistics), registration
    register_final(model_tell, d.loaders, device, path, f'logix-gin-{dataset_name}', ('models',), ('args.json',),
                   {'seed': seed, 'hidden_dim': baseline_args['hidden_dim'], 'val_acc': val_acc, 'test_acc': test_acc,
                    'path': path})
    return {'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc}


def eval_seed(dataset_name, baseline_path, args, seed, device):
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    return eval_student(dataset_name, baseline_path, args, seed, device, run_dir(dataset_name, args, baseline_args, seed),
                        f'Evaluating logic model on {dataset_name}')


def train_eval(dataset_name, baseline_path, args):
    device = get_device()
    baseline_args = json.load(open(os.path.join(baseline_path, '0', 'args.json'), 'r'))
    seed = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    if args.get('pool_ops') is None:               # upstream readout: keep existing dir names
        args.pop('pool_ops', None)
    path = run_dir(dataset_name, args, baseline_args)
    teacher = lambda s: os.path.join(baseline_path, str(s))  # noqa: E731
    hidden = baseline_args['hidden_dim']
    return run_seeds(path, lambda s: train_seed(dataset_name, teacher(s), args, s, device),
                     lambda s: eval_seed(dataset_name, teacher(s), args, s, device), seed, only_eval,
                     lambda df, ret: log_summary(dataset_name, f'logic/h{hidden}/summary', args, df, ret, path, LOGIC_ROOT,
                                                 {'hidden_dim': hidden}))


def get_parser():
    parser = argparse.ArgumentParser(description='train_logic.py')

    parser.add_argument('--dataset',       default='PROTEINS', type=str,     help='Dataset to use')
    parser.add_argument('--baseline_path', default=None,       type=str,     help='Baseline path')
    parser.add_argument('--epochs',        default=5000,       type=int,     help='Epochs')
    parser.add_argument('--warmup_epochs', default=3000,       type=int,     help='Epochs')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--lr',            default=0.001,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=0.0,      type=float,     help='Weight decay')
    parser.add_argument('--conv_reg',      default=0.001,      type=float,   help='Conv layer regularization')
    parser.add_argument('--fc_reg',        default=0.01,      type=float,    help='Last layer regularization')
    parser.add_argument('--pool_ops',      default=None,      type=str,     help='Comma-separated readout pooling, e.g. mean,max (default: upstream mean,max,sum; models_proto/gintell.py)')
    parser.add_argument('--head_init',     default=None,      choices=['center'], help='center: head bias set so each class\'s mean pre-activation on the training set is 0 (default: random, upstream)')
    parser.add_argument('--logit_loss',    action='store_true',              help='BCE of the task loss on the head logits, so a saturated head keeps its gradient (default: on the sigmoid output, upstream)')
    parser.add_argument('--only_eval',     action='store_true',              help='Only evaluate the kept checkpoints')
    parser.add_argument('--seed',          default=None,      type=int,    help='Single seed to run')
    return parser


def parse_cli(argv=None):
    """(dataset, baseline_path or None, args) from a command line; optimize_optuna.py
    builds its trials through it too."""
    args = get_parser().parse_args(argv).__dict__
    for k in ('head_init', 'logit_loss'):                            # off: results paths unchanged
        if not args[k]:
            args.pop(k)
    return args.pop('dataset'), args.pop('baseline_path'), args


if __name__ == '__main__':
    dataset_name, baseline_path, args = parse_cli()
    if baseline_path is None:
        baseline_path = best_teacher_dir(dataset_name)
        print('Baseline path found:', baseline_path)
    train_eval(dataset_name, baseline_path, args)
