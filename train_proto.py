"""Layer-wise distillation of a frozen GIN teacher into a LogiX-GIN *with prototypes*.

Same training logic as train_logic.py (frozen GIN teacher -> per-layer BCE on the
GINTELL trunk, warmup then full end-to-end, ReduceLROnPlateau, one run per seed);
only the head differs. Where train_logic.py feeds the teacher's pooled embedding
straight into `model_tell.fc`, the prototype models need the teacher's *node*
states so their own readout can pool them, so the head step goes through
`forward_from_layers`.

Extra knobs over train_logic.py:
    --proto_level {node,graph,both}   which readout (see models_proto/model_proto.py)
    --num_prototypes K
    --proto_div_reg / --proto_ent_reg weights of the two prototype regularisers
    --push_every                      ProtoPNet push period (0 = never)
    --proto_mask                      binary care mask per prototype: similarity becomes a
                                      soft AND over few cared bits (models_proto/proto.py);
                                      --mask_reg penalises the fraction of cared bits and the
                                      temperature is annealed from --mask_temp_start (default
                                      d/4) to --mask_temp_end over the first --mask_anneal_frac
                                      of the epochs, then held. --mask_ckpt_temp keeps a
                                      checkpoint only once T <= that value, so the kept model
                                      reads as ANDs (above it a prototype is an m-of-n rule).
                                      Off by default; the mask options only enter the results
                                      path when it is on, the last two only when set.

Results go to results_proto/ so they never collide with results_logic/.
"""
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from utils.syn_dataset import SynGraphDataset
from utils.spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from utils.utils import *
from sklearn.model_selection import train_test_split
import shutil
import glob
import traceback
import pandas as pd
import argparse
import hashlib
import pickle
import json
from models_proto.model import GIN
from models_proto.model_proto import get_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import tracking
from utils.evaluation import evaluate

SEEDS = 10


def create_folder_proto(dataset_name, args, baseline_args, seed=None):
    """Mirror of utils.create_folder_logic, under results_proto/.

    args already carries proto_level / num_prototypes / ..., so two prototype
    configurations never write into the same directory.
    """
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    full_args_s = args_s
    if len(args_s.encode()) > 255:        # filesystem limit on one path component
        args_s = f"{args['proto_level']}-{hashlib.sha1(args_s.encode()).hexdigest()[:12]}"
    baseline_args_s = '|'.join([f"{k}={baseline_args[k]}" for k in sorted(baseline_args.keys())])
    path = f'results_proto/{dataset_name}/{args_s}/{baseline_args_s}'
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    if args_s != full_args_s:             # the hashed name's configuration, for lookup
        with open(f'results_proto/{dataset_name}/{args_s}/config.txt', 'w') as f:
            f.write(full_args_s + '\n')
    return path


def get_best_baseline_path(dataset_name):
    l = glob.glob(f'results/{dataset_name}/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=False')]
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname


def train_epoch(model, model_proto, loader, device, optimizer, num_classes, train_full=True,
                conv_reg=0.001, fc_reg=0.01, proto_div_reg=0.0, proto_ent_reg=0.0,
                extra_loss=None, mask_reg=0.0):
    """One epoch of distillation + task loss.

    ``extra_loss(model_proto) -> tensor`` is added to every batch's loss; the
    post-training sparsity fine-tuning (sparsify_proto.py) passes its Hoyer term here
    so it optimises exactly this objective plus the penalty.
    """
    model.train()
    model_proto.train()

    total_loss = [0]*(len(model_proto.convs)+1)
    total_correct = [0]*(len(model_proto.convs)+1)
    n_failed = 0                      # see the guard after the loop

    for data in loader:
        try:
            if data.x is None:
                data.x = torch.ones((data.num_nodes, model.num_features))
            if data.y.numel() == 0: continue
            if data.x.isnan().any(): continue
            if data.y.isnan().any(): continue
            y = data.y.reshape(-1).to(device).long()
            batch = data.batch.to(device)
            optimizer.zero_grad()
            # Frozen teacher. layers_y[i] = node states after conv i (binary, gumbel
            # hard=True); layers_x[i] = the input that produced them.
            with torch.no_grad():
                layers_x, layers_y = model.forward_e(data.x.float().to(device), data.edge_index.to(device), batch)
            loss = 0
            last_layer_out = None
            # ---- trunk: identical to train_logic.py, the prototypes change nothing here ----
            for i, (layer, layer_x, layer_y) in enumerate(zip(model_proto.convs, layers_x[:-1], layers_y[:-1])):
                if train_full: layer.nn[0].phi_in.tau = 10
                if i !=0: layer_x = F.dropout(layer_x, p=0.2)*(1-0.2)
                layer_out = layer(torch.hstack([layer_x, 1-layer_x]), data.edge_index.to(device))
                layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
                loss += layer_loss
                if train_full and i!=0:
                    layer_x = last_layer_out
                    layer_out = layer(torch.hstack([layer_x, 1-layer_x]), data.edge_index.to(device))
                    layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
                    loss += layer_loss
                last_layer_out = layer_out
                total_loss[i] += layer_loss.item() / len(loader.dataset)
                total_correct[i] += ((layer_out >= 0.5).long() == layer_y).sum().item() / (layer_y.shape[-2]*layer_y.shape[-1]*len(loader))

            # ---- head: THE difference from train_logic.py ----
            # train_logic.py does  model_tell.fc(hstack([layers_x[-1], 1-layers_x[-1]]))
            # i.e. it feeds the teacher's already-pooled embedding to the head. The
            # prototype readout cannot consume that: it has to see per-node states to
            # compare them to prototypes (and to pool afterwards). forward_from_layers
            # runs readout+head on the teacher's node states, so the prototypes and the
            # head learn on teacher features while the trunk is still being distilled,
            # exactly as the fc did before.
            out = model_proto.forward_from_layers(layers_y[:-1], batch)
            pred = out.argmax(-1)
            loss += F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())

            if train_full:
                model_proto.fc.phi_in.tau = 10
                out = model_proto(data.x.float().to(device), data.edge_index.to(device), batch)
                pred = out.argmax(-1)
                loss += F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())

            loss += fc_reg*(model_proto.fc.reg_loss + model_proto.fc.phi_in.entropy)

            # Prototype regularisers. PrototypeLayer sets both attributes on every
            # forward: reg_loss = mean pairwise agreement between prototypes (push it
            # down to keep them distinct), proto_entropy = how undecided they are (push
            # it down so the soft prototypes converge to actual {0,1} patterns and the
            # straight-through binarisation stops lying). Without these two terms the
            # prototypes stay soft and redundant, which is what the variant exists to avoid.
            for p in model_proto.proto_layers:
                loss = loss + proto_div_reg * p.reg_loss + proto_ent_reg * p.proto_entropy
                if mask_reg and p.masked:
                    loss = loss + mask_reg * p.mask_size

            if extra_loss is not None:
                loss = loss + extra_loss(model_proto)

            loss.backward()
            zero_nan_gradients(model_proto)
            # NOTE kept verbatim from train_logic.py, including the fact that it clips
            # `model` (the frozen teacher), not `model_proto`. As written this is a no-op
            # and the student is effectively trained unclipped. Change to
            # model_proto.parameters() if you want the clipping the line intends.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss[-1] += loss.item() * data.num_graphs / len(loader.dataset)
            total_correct[-1] += pred.eq(y).sum().item() / len(loader.dataset)
        except Exception as e:
            # train_logic.py swallows batch errors to survive degenerate batches. Kept,
            # but counted: a shape/config error fails *every* batch and would otherwise
            # produce a full silent run at chance accuracy.
            n_failed += 1
            traceback.print_exc()

    if n_failed and n_failed == len(loader):
        raise RuntimeError(f'every batch failed ({n_failed}/{len(loader)}) - see the traceback above')

    return total_loss, total_correct


@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        if data.y.numel() == 0: continue
        if data.x.isnan().any(): continue
        if data.y.isnan().any(): continue
        y = data.y.reshape(-1).to(device)
        pred = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1000).argmax(-1)
        total_correct += pred.eq(y).sum().item()
    val_acc = total_correct / len(loader.dataset)

    return val_acc


def build_proto_model(args, baseline_args, num_features, num_classes, device):
    """The student. num_layers/hidden_dim must match the teacher: the trunk is
    distilled layer by layer, so the shapes have to line up."""
    return get_model(
        args['proto_level'],
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=baseline_args['hidden_dim'],
        num_layers=baseline_args['num_layers'],
        num_prototypes=args['num_prototypes'],
        proto_mask=args.get('proto_mask', False),
    ).to(device)


def mask_temperature(args, epoch, d):
    """Geometric annealing of the masked-prototype temperature over the first
    ``mask_anneal_frac`` of the epochs (all of them by default), then held at the end value."""
    t0 = args.get('mask_temp_start') or d / 4
    t1 = args.get('mask_temp_end', 1.0)
    f = min(1.0, epoch / max(1, args.get('mask_anneal_frac', 1.0) * args['epochs'] - 1))
    return t0 * (t1 / t0) ** f


def train_seed(dataset_name, baseline_path, args, seed, device):
    """One seed, logged as one MLflow run of experiment proto/<dataset> (utils/tracking.py)."""
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    cfg = os.path.relpath(os.path.dirname(create_folder_proto(dataset_name, args, baseline_args, seed=seed)),
                          f'results_proto/{dataset_name}')
    params = {**args, **{f'teacher/{k}': v for k, v in baseline_args.items()},
              'seed': seed, 'dataset': dataset_name, 'teacher_path': baseline_path}
    with tracking.run(f'proto/{dataset_name}', run_name=f"{args['proto_level']}/seed{seed}", params=params,
                      tags={'config': cfg, 'seed': seed, 'kind': 'seed', 'dataset': dataset_name,
                            'proto_level': args['proto_level']}):
        return _train_seed(dataset_name, baseline_path, args, seed, device)


def _train_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)


    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_proto(dataset_name, args, baseline_args, seed=seed)
    shutil.rmtree(path)
    path = create_folder_proto(dataset_name, args, baseline_args, seed=seed)

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args, f)


    dataset = get_dataset(dataset_name)

    print(f'Training prototype logic model ({args["proto_level"]}, K={args["num_prototypes"]}) on {dataset_name}')
    print(baseline_args)


    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if num_features == 0: num_features = 10

    # Reuse the baseline's split, otherwise the teacher has seen part of our test set.
    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))

    train_dataset = dataset[data['train_indices']]
    val_dataset = dataset[data['val_indices']]
    test_dataset = dataset[data['test_indices']]
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Unshuffled copy: the push must read the training set in a fixed order for the
    # returned row indices to identify the graphs the prototypes snapped to.
    push_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'train_indices': data['train_indices'],
            'val_indices': data['val_indices'],
            'test_indices': data['test_indices'],
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, f)

    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, test_loader, device))
    model_proto = build_proto_model(args, baseline_args, num_features, num_classes, device)

    optimizer = torch.optim.AdamW(model_proto.parameters(), lr=args['lr'], weight_decay=args['l2'])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=300, min_lr=1e-5, verbose=True)

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    push_log = []
    for epoch in range(args['epochs']):
        if args.get('proto_mask'):
            for p in model_proto.proto_layers:
                p.mask_temp = mask_temperature(args, epoch, p.in_features)
        train_loss, train_acc = train_epoch(model, model_proto, train_loader, device, optimizer, num_classes,
                                            train_full=epoch>args['warmup_epochs'], conv_reg=args['conv_reg'], fc_reg=args['fc_reg'],
                                            proto_div_reg=args['proto_div_reg'], proto_ent_reg=args['proto_ent_reg'],
                                            mask_reg=args.get('mask_reg', 0.0))

        # ProtoPNet push: snap every prototype onto the closest real training example,
        # so each one *is* an observed pattern and the extracted rules stay readable.
        # Only after warmup - before that the trunk still moves, so the features a
        # prototype would be snapped to are not yet meaningful.
        if args['push_every'] and epoch > args['warmup_epochs'] and epoch % args['push_every'] == 0:
            idxs = model_proto.project_prototypes(push_loader, device)
            push_log.append({'epoch': epoch, 'indices': [i.cpu().tolist() for i in idxs]})

        val_acc = test_epoch(model_proto, val_loader, device)
        test_acc = test_epoch(model_proto, test_loader, device)

        if epoch > args['warmup_epochs']:
            scheduler.step(val_acc)
        # masked prototypes are only ANDs once T is low: keep no checkpoint before that
        ckpt_open = args.get('mask_ckpt_temp') is None or \
            all(p.mask_temp <= args['mask_ckpt_temp'] + 1e-9 for p in model_proto.proto_layers)
        if epoch>args['warmup_epochs'] and ckpt_open and val_acc >= best_val_acc:
            torch.save(model_proto, os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc

        # train_loss / train_acc: one entry per conv layer (distillation), then the head
        tracking.log_metrics({'val_acc': val_acc, 'test_acc': test_acc, 'best_val_acc': best_val_acc,
                              'best_test_acc': best_test_acc, 'train_loss': train_loss[-1],
                              'train_acc': train_acc[-1], 'lr': optimizer.param_groups[0]['lr'],
                              'phase_full': float(epoch > args['warmup_epochs']),
                              **{f'distill/L{i}_loss': l for i, l in enumerate(train_loss[:-1])},
                              **{f'distill/L{i}_acc': c for i, c in enumerate(train_acc[:-1])}},
                             step=epoch)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    torch.save(model_proto, os.path.join(path, 'last.pt'))
    # Which training graph/node each prototype ended up standing for.
    with open(os.path.join(path, 'push_log.json'), 'w') as f:
        json.dump(push_log, f)
    model_proto = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_proto, val_loader, device)
    test_acc = test_epoch(model_proto, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    # Final evaluation of the kept checkpoint: imbalance-aware metrics and the size of the
    # extracted explanation (rules per unit, shortest rule, fidelity), then registration.
    if tracking.enabled():
        from rule_eval import rule_metrics, rule_stats
        final = {**evaluate(model_proto, val_loader, device, prefix='final/val_'),
                 **evaluate(model_proto, test_loader, device, prefix='final/test_')}
        stats = rule_stats(model_proto, val_loader, device)
        tracking.log_metrics({**final, **rule_metrics(stats, prefix='final/rules/')})
        tracking.log_dict({'metrics': final, 'rules': stats}, 'final_evaluation.json')
        for f in ('args.json', 'push_log.json'):
            tracking.log_artifact(os.path.join(path, f))
        tracking.log_model(model_proto, f"logix-proto-{dataset_name}-{args['proto_level']}",
                           code_dirs=('models_proto',),
                           tags={'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc, 'path': path})

    return results


def log_summary(dataset_name, args, df, path):
    """Mean/std over seeds as a kind=summary run next to the seed runs."""
    cfg = os.path.relpath(path, f'results_proto/{dataset_name}')
    with tracking.run(f'proto/{dataset_name}', run_name=f"{args['proto_level']}/summary",
                      params={**args, 'n_seeds': len(df)},
                      tags={'config': cfg, 'kind': 'summary', 'dataset': dataset_name,
                            'proto_level': args['proto_level']}):
        tracking.log_metrics({f'{c}_{s}': getattr(df[c], s)() for c in ('val_acc', 'test_acc')
                              for s in ('mean', 'std')})
        tracking.log_artifact(os.path.join(path, 'total_results.csv'))


def eval_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)

    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_proto(dataset_name, args, baseline_args, seed=seed)

    dataset = get_dataset(dataset_name)

    print(f'Evaluating prototype logic model on {dataset_name}')
    print(baseline_args)

    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if num_features == 0: num_features = 10

    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))

    train_dataset = dataset[data['train_indices']]
    val_dataset = dataset[data['val_indices']]
    test_dataset = dataset[data['test_indices']]
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, test_loader, device))
    model_proto = build_proto_model(args, baseline_args, num_features, num_classes, device)
    model_proto = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_proto, val_loader, device)
    test_acc = test_epoch(model_proto, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results


def train_eval(dataset_name, baseline_path, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    baseline_args = json.load(open(os.path.join(baseline_path, '0', 'args.json'), 'r'))

    seed_todo = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)

    path = create_folder_proto(dataset_name, args, baseline_args)

    seeds = range(SEEDS)
    if seed_todo is not None:
        seeds = [seed_todo]

    results = []
    if not only_eval:
        for seed in seeds:
            results.append(train_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device))

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            try:
                r = eval_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device)
                results.append(r)
                print(r)
            except Exception as e: print(e)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_acc_mean': df['val_acc'].mean(),
        'test_acc_mean': df['test_acc'].mean(),
        'val_acc_std': df['val_acc'].std(),
        'test_acc_std': df['test_acc'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    # parallel single-seed processes each re-evaluate every seed; only a full or
    # --only_eval pass writes the summary run, so it exists once per configuration
    if only_eval or seed_todo is None:
        log_summary(dataset_name, args, df, path)

    return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_proto.py')

    parser.add_argument('--dataset',        default='PROTEINS', type=str,   help='Dataset to use')
    parser.add_argument('--baseline_path',  default=None,       type=str,   help='Baseline path')
    parser.add_argument('--epochs',         default=5000,       type=int,   help='Epochs')
    parser.add_argument('--warmup_epochs',  default=3000,       type=int,   help='Epochs')
    parser.add_argument('--batch_size',     default=32,         type=int,   help='Batch Size')
    parser.add_argument('--lr',             default=0.001,      type=float, help='Learning Rate')
    parser.add_argument('--l2',             default=0.0,        type=float, help='Weight decay')
    parser.add_argument('--conv_reg',       default=0.001,      type=float, help='Conv layer regularization')
    parser.add_argument('--fc_reg',         default=0.01,       type=float, help='Last layer regularization')
    # --- prototype-specific ---
    parser.add_argument('--proto_level',    default='graph',    type=str,   choices=['node', 'graph', 'both'], help='Where the prototypes sit: after pooling (graph), before it (node), or both')
    parser.add_argument('--num_prototypes', default=16,         type=int,   help='Number of prototypes K')
    parser.add_argument('--proto_div_reg',  default=0.01,       type=float, help='Weight of the prototype diversity penalty')
    parser.add_argument('--proto_ent_reg',  default=0.01,       type=float, help='Weight of the prototype entropy penalty')
    parser.add_argument('--push_every',     default=0,          type=int,   help='Push prototypes onto real examples every N epochs after warmup (0 = never)')
    parser.add_argument('--proto_mask',     action='store_true',             help='Learn a binary care mask per prototype (soft AND over cared bits)')
    parser.add_argument('--mask_reg',       default=0.5,        type=float, help='Weight of the cared-bit fraction penalty (with --proto_mask)')
    parser.add_argument('--mask_temp_start', default=None,      type=float, help='Initial mask temperature (default d/4, with --proto_mask)')
    parser.add_argument('--mask_temp_end',  default=1.0,        type=float, help='Final mask temperature (with --proto_mask)')
    parser.add_argument('--mask_anneal_frac', default=None,     type=float, help='Fraction of the epochs over which T is annealed, then held (default 1, with --proto_mask)')
    parser.add_argument('--mask_ckpt_temp', default=None,       type=float, help='Keep checkpoints only once T <= this (default: always, with --proto_mask)')
    parser.add_argument('--only_eval',     action='store_true',             help='Only evaluate')
    parser.add_argument('--seed',           default=None,       type=int,   help='Single seed to run')

    args = parser.parse_args().__dict__
    if not args['proto_mask']:            # keep the results paths of unmasked configurations unchanged
        for k in ('proto_mask', 'mask_reg', 'mask_temp_start', 'mask_temp_end'):
            args.pop(k)
    for k in ('mask_anneal_frac', 'mask_ckpt_temp'):      # nor those of earlier masked ones
        if args[k] is None or not args.get('proto_mask'):
            args.pop(k)
    if args.get('mask_ckpt_temp') is not None and args['mask_ckpt_temp'] < args['mask_temp_end']:
        parser.error('--mask_ckpt_temp below --mask_temp_end would never keep a checkpoint')

    dataset_name = args.pop('dataset')
    baseline_path = args.pop('baseline_path')
    if baseline_path is None:
        baseline_path = get_best_baseline_path(dataset_name)
        print('Baseline path found:', baseline_path)
    train_eval(dataset_name, baseline_path, args)
