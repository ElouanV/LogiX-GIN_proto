"""Teacher: a GIN with binary (Gumbel) node states trained on the graph labels, one run per
seed (upstream training, unchanged).

    python train_baseline.py --dataset MUTAG --hidden_dim 64 --num_layers 3 [--split kfold] [--seed 0]

Results go to results/<dataset>/<cfg>/<seed> (utils/runs.py); --split kfold reads the
committed fold indices (utils/splits.py).
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.model import GIN
from utils import tracking
from utils.datasets import get_dataset
from utils.evaluation import evaluate, held_out
from utils.runs import TEACHER_ROOT, get_device, make_loaders, num_features, save_split, split_datasets, teacher_dir, \
    write_args
from utils.splits import SPLITS, split_indices
from utils.training import NUM_SEEDS, fresh_dir, log_summary, run_seeds, summarize
from utils.utils import set_seed, zero_nan_gradients


def train_epoch(model, loader, device, optimizer, num_classes):
    """One training epoch; optimizer=None: the same loss on loader without training (eval
    mode, no gradient, RNG untouched, utils.evaluation.held_out), i.e. the val loss."""
    if optimizer is None:
        with held_out():
            return _epoch(model.eval(), loader, device, None, num_classes)
    return _epoch(model.train(), loader, device, optimizer, num_classes)


def _epoch(model, loader, device, optimizer, num_classes):
    train = optimizer is not None
    total_loss = 0
    total_correct = 0
    for data in loader:
        y = data.y.squeeze(-1).to(device).long()
        if train:
            optimizer.zero_grad()
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1)
        pred = out.argmax(-1)
        try:
            loss = F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1))
            nll_loss = F.nll_loss(F.log_softmax(out, dim=-1), y.long())
            loss = nll_loss + loss
            if train:
                loss.backward()
                zero_nan_gradients(model)
                # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        except: continue
        if train:
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_correct += pred.eq(y).sum().item()
    train_loss = total_loss / len(loader.dataset)
    train_acc = total_correct / len(loader.dataset)

    return train_loss, train_acc

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        y = data.y.squeeze(-1).to(device)
        pred = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1000).argmax(-1)
        total_correct += pred.eq(y).sum().item()
    val_acc = total_correct / len(loader.dataset)
    
    return val_acc

def split_data(dataset_name, args, seed):
    """The dataset, its split for this seed (utils.splits.split_indices) and the split datasets."""
    dataset = get_dataset(dataset_name)
    train, val, test = split_indices(dataset.data.y, seed, args.get('split', 'random'), dataset_name)
    indices = {'train_indices': train, 'val_indices': val, 'test_indices': test}
    return dataset, indices, split_datasets(dataset, indices)


def build_teacher(args, dataset, device):
    return GIN(num_features=num_features(dataset), num_classes=dataset.num_classes, hidden_dim=args['hidden_dim'],
               num_layers=args['num_layers'], nogumbel=args['nogumbel']).to(device)


def train_seed(dataset_name, args, seed, device):
    """One seed, logged as the teacher stage of its own pipeline run (utils/tracking.py);
    student pipelines refer to it (tracking.teacher_reference)."""
    path = teacher_dir(dataset_name, args, seed=seed)
    cfg = os.path.relpath(os.path.dirname(path), f'{TEACHER_ROOT}/{dataset_name}')
    with tracking.run(f'baseline/{dataset_name}', run_name=f'teacher/seed{seed}',
                      params={**args, 'seed': seed, 'dataset': dataset_name},
                      tags={'config': cfg, 'seed': seed, 'kind': 'seed', 'dataset': dataset_name},
                      stage='teacher', dataset=dataset_name):
        res = _train_seed(dataset_name, args, seed, device)
        tracking.write_pipeline(path)
        return res


def _train_seed(dataset_name, args, seed, device):
    set_seed(seed)
    path = fresh_dir(lambda: teacher_dir(dataset_name, args, seed=seed))
    args = write_args(path, args)
    dataset, indices, datasets = split_data(dataset_name, args, seed)
    num_classes = dataset.num_classes
    save_split(path, indices, datasets)
    loaders = make_loaders(datasets, args['batch_size'])
    train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']
    model = build_teacher(args, dataset, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['l2'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=100, min_lr=1e-5, verbose=True)

    best_val_acc = 0
    best_test_acc = 0
    max_patience = 200
    patience = 0
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, device, optimizer, num_classes)
        val_loss, _ = train_epoch(model, val_loader, device, None, num_classes)
        val_acc = test_epoch(model, val_loader, device)
        test_acc = test_epoch(model, test_loader, device)
        scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            patience = 0
            torch.save(model.state_dict(), os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc
        elif epoch > args['epochs']//2: patience += 1

        tracking.log_metrics({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc,
                              'test_acc': test_acc, 'best_val_acc': best_val_acc,
                              'best_test_acc': best_test_acc, 'lr': optimizer.param_groups[0]['lr']},
                             step=epoch)

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

        if patience >= max_patience: break

    torch.save(model.state_dict(), os.path.join(path, 'last.pt'))
    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    val_acc = test_epoch(model, val_loader, device)
    test_acc = test_epoch(model, test_loader, device)

    if tracking.enabled():
        tracking.log_metrics({**evaluate(model, val_loader, device, prefix='final/val_'),
                              **evaluate(model, test_loader, device, prefix='final/test_'),
                              'stopped_epoch': epoch})
        tracking.log_artifact(os.path.join(path, 'args.json'))
        tracking.log_model(model, f'gin-teacher-{dataset_name}', code_dirs=('models',),
                           tags={'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc, 'path': path})

    return {'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc}


def eval_seed(dataset_name, args, seed, device):
    set_seed(seed)
    path = teacher_dir(dataset_name, args, seed=seed)
    dataset, _, datasets = split_data(dataset_name, args, seed)
    loaders = make_loaders(datasets, args['batch_size'])
    model = build_teacher(args, dataset, device)
    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    return {'seed': seed, 'val_acc': test_epoch(model, loaders['val'], device),
            'test_acc': test_epoch(model, loaders['test'], device)}


def train(dataset_name, args):
    """Every seed, summarized (the grid search of optimize_baseline.py)."""
    device = get_device()
    path = teacher_dir(dataset_name, args)
    results = [train_seed(dataset_name, args, seed, device) for seed in range(NUM_SEEDS)]
    return summarize(results, path)[1]


def train_eval(dataset_name, args):
    device = get_device()
    seed = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    # the upstream split stays out of the config, so existing result dirs keep their names
    if args.get('split', 'random') == 'random':
        args.pop('split', None)
    path = teacher_dir(dataset_name, args)
    print(path)
    return run_seeds(path, lambda s: train_seed(dataset_name, args, s, device),
                     lambda s: eval_seed(dataset_name, args, s, device), seed, only_eval,
                     lambda df, ret: log_summary(dataset_name, 'teacher/summary', args, df, ret, path, TEACHER_ROOT))


def get_parser():
    parser = argparse.ArgumentParser(description='train_baseline.py')

    parser.add_argument('--dataset',       default='PROTEINS', type=str,     help='Dataset to use')
    parser.add_argument('--epochs',        default=1000,       type=int,     help='Epochs')
    parser.add_argument('--hidden_dim',    default=128,        type=int,     help='Hid Dim')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--num_layers',    default=5,          type=int,     help='Number of Convolutional Layers')
    parser.add_argument('--dropout',       default=0.15,       type=float,   help='Dropout')
    parser.add_argument('--lr',            default=0.01,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=1e-5,      type=float,   help='Weight Decay')
    parser.add_argument('--nogumbel',      action='store_true',             help='Deterministic teacher (no Gumbel sampling)')
    parser.add_argument('--only_eval',     action='store_true',              help='Only evaluate the kept checkpoints')
    parser.add_argument('--seed',          default=None,      type=int,    help='Single seed to run')
    parser.add_argument('--split',         default='random',  choices=SPLITS, help='random: 80/10/10 per seed (upstream); kfold: stratified 10-fold, seed = fold')
    return parser


def parse_cli(argv=None):
    """(dataset, args) from a command line; utils/experiment.py builds teacher runs through it."""
    args = get_parser().parse_args(argv).__dict__
    return args.pop('dataset'), args


if __name__ == '__main__':
    dataset_name, args = parse_cli()
    train_eval(dataset_name, args)
