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
                                      --mask_layer_cost c0,c1,.. weights the cared-bit penalty
                                      by the conv layer a bit reads (e.g. 0.25,1,1 makes L0 bits,
                                      the only ones that decode to short rules, cheaper).
                                      --vocab_reg weights a group-sparsity penalty on the trunk units
                                      the prototypes care about (vocab_penalty), so they share few.
                                      Off by default; the mask options only enter the results
                                      path when it is on, the last four only when set.
    --harden_to B                     gradual hardening (GINTELLProtoBase.set_sharpness): the
                                      literals and conv units are sharpened towards {0, 1},
                                      the sharpness growing geometrically from 1 to B between
                                      --harden_start_frac (default: the end of the warmup) and
                                      --harden_end_frac (default 0.9) of the epochs, then held;
                                      checkpoints are kept only at sharpness B, so the kept
                                      network is close to the rules it is read as. Off by
                                      default (no harden_* entry in the results path).
    --proto_init data                 prototypes start on training nodes (node level) / graphs
                                      (graph level) of the teacher's states, farthest-point
                                      sampled, instead of random patterns, which are all about
                                      equally far from every node (no readout contrast at init).
                                      A searched hyper-parameter (configs/search_spaces.yaml);
                                      default random, upstream, out of the results path.
    --head_phi bounded                head literals are thresholds x >= t (models_proto/tell.py
                                      BoundedPhi: slope w <= pi, threshold t in (0, 1)) instead
                                      of Phi's periodic step, which reads as bands or unions on
                                      [0, 1] when w > pi and is true on every input at init.
                                      --head_threshold center (default) starts each t at the
                                      median of its column on the training set, mid at 0.5.
                                      The trunk keeps Phi. Off by default (upstream), out of
                                      the results path.
    --node_readout predicate          node level, masked: the head reads, per prototype k, the
                                      count and the share of the nodes satisfying its rule up to
                                      a learned tolerance (models_proto/model_proto_count.py)
                                      instead of mean / max similarities; the tolerances start
                                      at the median similarity of the training nodes. Its head
                                      literals are monotone thresholds (tell.RampPhi), centred
                                      with --head_phi bounded. Off by default.
    --head_init center, --logit_loss  fixes of the head saturation (sigmoid(10 z) = exactly 1 at
                                      init, zero task gradient, constant output on small datasets
                                      such as MUTAG): centre the head bias on the training set
                                      before the first step (utils.training.center_head),
                                      and compute the task BCE from the logits
                                      (utils.training.class_loss). Off by default (upstream);
                                      they only enter the results path when set.
    --teacher_eval                    the frozen teacher runs in eval mode while the student is
                                      distilled on its states. Upstream keeps it in train mode, so
                                      its dropout (0.15) is active: the inputs of conv layers 1-2
                                      are 0 or 1/0.85 (negation -0.18), outside the [0, 1] the
                                      student reads at test time. Its Gumbel sampling is the same
                                      in both modes (docs/GUMBEL_TEACHER.md). Off by default
                                      (upstream); in the results path only when set.
    --optimizer muon [--muon_lr]      Muon on the conv weight matrices, AdamW on the rest
                                      (utils/optim.py). Default AdamW, upstream; the
                                      optimizer only enters the results path when set.

Results go to results_proto/ so they never collide with results_logic/.
"""
import argparse
import json
import os
import traceback

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from models_proto.model_proto import get_model
from models_proto.model_proto_count import GINTELLProtoCount
from utils import tracking
from utils.evaluation import held_out
from utils.optim import build_optimizer
from utils.runs import EVAL_BATCH_SIZE, PROTO_ROOT, best_teacher_dir, get_device, student_dir
from utils.training import (center_head, center_thresholds, class_loss, distill_trunk, teacher_states, eval_student, fresh_dir, log_summary, register_final,
                            run_seeds, student_data, test_epoch, tracked_student_seed)
from utils.utils import set_seed, zero_nan_gradients


def run_dir(dataset_name, args, baseline_args, seed=None, root=PROTO_ROOT):
    """args carries proto_level / num_prototypes / ..., so two prototype configurations never
    write into the same directory; too long a name is hashed (utils.runs.student_dir)."""
    return student_dir(dataset_name, args, baseline_args, seed=seed, root=root, hash_long=True)


def train_epoch(model, model_proto, loader, device, optimizer, num_classes, train_full=True,
                conv_reg=0.001, fc_reg=0.01, proto_div_reg=0.0, proto_ent_reg=0.0,
                mask_reg=0.0, vocab_reg=0.0, logit_loss=False, quant=None, base_weight=1.0, extra_loss=None,
                teacher_eval=False):
    """One epoch of distillation + task loss. QAT (qat_proto.py): ``quant`` is applied to a
    layer's output before the next layer reads it in the chained (train_full) term, the
    layer's own loss stays on its soft output; the loss is ``base_weight`` x this objective
    + ``extra_loss(x, edge_index, batch, y)`` when given. ``teacher_eval`` (--teacher_eval): the
    frozen teacher runs in eval mode, so its dropout does not reach the states it is distilled
    on (upstream runs it in train mode: layer inputs 0 or 1/(1 - 0.15), outside [0, 1]).

    optimizer=None: the same loss on loader without training (the val loss): student in
    eval mode and without input dropout, teacher sampled as in training, no gradient, RNG
    untouched (utils.evaluation.held_out)."""
    kw = dict(train_full=train_full, conv_reg=conv_reg, fc_reg=fc_reg, proto_div_reg=proto_div_reg,
              proto_ent_reg=proto_ent_reg, mask_reg=mask_reg, vocab_reg=vocab_reg, quant=quant,
              logit_loss=logit_loss, base_weight=base_weight, extra_loss=extra_loss, teacher_eval=teacher_eval)
    if optimizer is None:
        with held_out():
            return _epoch(model, model_proto, loader, device, None, num_classes, **kw)
    return _epoch(model, model_proto, loader, device, optimizer, num_classes, **kw)


def _epoch(model, model_proto, loader, device, optimizer, num_classes, train_full, conv_reg, fc_reg,
           proto_div_reg, proto_ent_reg, mask_reg, vocab_reg, logit_loss, quant, base_weight, extra_loss,
           teacher_eval):
    train = optimizer is not None
    model.train(not teacher_eval)
    model_proto.train(train)

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
            if train:
                optimizer.zero_grad()
            # Frozen teacher. layers_y[i] = node states after conv i (binary, gumbel
            # hard=True); layers_x[i] = the input that produced them.
            with torch.no_grad():
                layers_x, layers_y = model.forward_e(data.x.float().to(device), data.edge_index.to(device), batch)
            # ---- trunk: identical to train_logic.py, the prototypes change nothing here ----
            loss = distill_trunk(model_proto, layers_x, layers_y, data.edge_index.to(device), train, train_full, conv_reg,
                                 total_loss, total_correct, len(loader.dataset), len(loader), quant)

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
            loss += class_loss(out, y, num_classes, model_proto.fc.logits if logit_loss else None)

            if train_full:
                model_proto.fc.phi_in.tau = 10
                out = model_proto(data.x.float().to(device), data.edge_index.to(device), batch)
                pred = out.argmax(-1)
                loss += class_loss(out, y, num_classes, model_proto.fc.logits if logit_loss else None)

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
            if vocab_reg:
                loss = loss + vocab_reg * model_proto.vocab_penalty()
            if extra_loss is not None:
                loss = base_weight * loss + extra_loss(data.x.float().to(device), data.edge_index.to(device), batch, y)

            if train:
                loss.backward()
                zero_nan_gradients(model_proto)
                optimizer.step()
            total_loss[-1] += loss.item() * data.num_graphs / len(loader.dataset)
            total_correct[-1] += pred.eq(y).sum().item() / len(loader.dataset)
        except Exception:
            # train_logic.py swallows batch errors to survive degenerate batches. Kept,
            # but counted: a shape/config error fails *every* batch and would otherwise
            # produce a full silent run at chance accuracy.
            n_failed += 1
            traceback.print_exc()

    if n_failed and n_failed == len(loader):
        raise RuntimeError(f'every batch failed ({n_failed}/{len(loader)}) - see the traceback above')

    return total_loss, total_correct


def loss_weights(args):
    """train_epoch's regulariser weights of a run, from its arguments (train_proto.py, and the
    fine-tunes harden_proto.py / qat_proto.py, which keep the objective that trained it)."""
    return dict(conv_reg=args['conv_reg'], fc_reg=args['fc_reg'], proto_div_reg=args['proto_div_reg'],
                proto_ent_reg=args['proto_ent_reg'], mask_reg=args.get('mask_reg', 0.0),
                vocab_reg=args.get('vocab_reg') or 0.0, logit_loss=args.get('logit_loss', False),
                teacher_eval=args.get('teacher_eval', False))


@torch.no_grad()
def build_proto_model(args, baseline_args, num_features, num_classes, device):
    """The student. num_layers/hidden_dim must match the teacher: the trunk is
    distilled layer by layer, so the shapes have to line up."""
    if args.get('node_readout') == 'predicate':
        return GINTELLProtoCount(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'],
                                 num_layers=baseline_args['num_layers'], num_prototypes=args['num_prototypes'],
                                 proto_mask=args.get('proto_mask', False)).to(device)
    kwargs = {}
    if args.get('pool_ops'):                       # default: each class's mean / max
        ops = tuple(args['pool_ops'].split(','))
        kwargs = ({'node_pool_ops': ops, 'graph_pool_ops': ops} if args['proto_level'] == 'both'
                  else {'pool_ops': ops})
    return get_model(
        args['proto_level'],
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=baseline_args['hidden_dim'],
        num_layers=baseline_args['num_layers'],
        num_prototypes=args['num_prototypes'],
        proto_mask=args.get('proto_mask', False),
        **kwargs,
    ).to(device)


def mask_temperature(args, epoch, d):
    """Geometric annealing of the masked-prototype temperature over the first
    ``mask_anneal_frac`` of the epochs (all of them by default), then held at the end value."""
    t0 = args.get('mask_temp_start') or d / 4
    t1 = args.get('mask_temp_end', 1.0)
    f = min(1.0, epoch / max(1, args.get('mask_anneal_frac', 1.0) * args['epochs'] - 1))
    return t0 * (t1 / t0) ** f


def set_mask_temperature(model_proto, args, epoch):
    if args.get('proto_mask'):
        for p in model_proto.proto_layers:
            p.mask_temp = mask_temperature(args, epoch, p.in_features)


def center_head_bias(teacher, model_proto, loader, device):
    """--head_init center (utils.training.center_head) on the teacher's node states of the
    graphs in loader, through this model's readout."""
    return center_head(teacher, model_proto.fc, loader, device,
                       lambda layers_x, layers_y, batch: model_proto.head_input(model_proto.readout(layers_y[:-1], batch)))


def init_prototypes(teacher, model_proto, loader, device, seed):
    """--proto_init data: every prototype set to a training node (node level) or graph (graph
    level) of the teacher's states, by farthest-point sampling (GINTELLProtoBase.init_prototypes;
    first row drawn with a generator seeded by the run seed). RNG forked as in center_head.
    Returns the chosen row indices per prototype layer, in loader order."""
    with held_out():
        states = [(layers_y[:-1], batch) for _, layers_y, batch in teacher_states(teacher, loader, device)]
        return model_proto.init_prototypes(states, torch.Generator().manual_seed(seed))


def sharpness(args, epoch):
    """Geometric hardening schedule: 1 until --harden_start_frac of the epochs (default the
    end of the warmup), harden_to from --harden_end_frac on."""
    b = args.get('harden_to')
    if b is None:
        return 1.0
    e0 = args['harden_start_frac'] * args['epochs'] if args.get('harden_start_frac') is not None \
        else args['warmup_epochs'] + 1
    e1 = max(e0 + 1, args.get('harden_end_frac', 0.9) * args['epochs'])
    return float(b ** min(1.0, max(0.0, (epoch - e0) / (e1 - e0))))


def train_seed(dataset_name, baseline_path, args, seed, device, out_root=None, epoch_callback=None, tags=None):
    """One seed, logged as one MLflow run of experiment proto/<dataset> (utils/tracking.py).

    ``out_root`` replaces the results root (e.g. a hyper-parameter study's directory) and
    ``epoch_callback(epoch, model, val_loader)`` is called after every epoch (optimize_optuna.py
    reports to Optuna and prunes there); ``tags`` are added to the MLflow run.
    """
    root = out_root or PROTO_ROOT
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = run_dir(dataset_name, args, baseline_args, seed, root)
    return tracked_student_seed(
        dataset_name, baseline_path, args, seed, path, root, f'proto/{dataset_name}',
        f"proto/{args['proto_level']}/seed{seed}", {'proto_level': args['proto_level'], **(tags or {})},
        lambda: _train_seed(dataset_name, baseline_path, args, seed, device, root, epoch_callback))


def _train_seed(dataset_name, baseline_path, args, seed, device, root=PROTO_ROOT, epoch_callback=None):
    set_seed(seed)
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = fresh_dir(lambda: run_dir(dataset_name, args, baseline_args, seed, root))
    d = student_data(dataset_name, baseline_path, args, device,
                     f'Training prototype logic model ({args["proto_level"]}, K={args["num_prototypes"]}) on {dataset_name}',
                     path)
    args, model = d.args, d.teacher
    train_loader, val_loader, test_loader = d.loaders['train'], d.loaders['val'], d.loaders['test']
    # Unshuffled copy: the push must read the training set in a fixed order for the
    # returned row indices to identify the graphs the prototypes snapped to.
    push_loader = DataLoader(d.datasets['train'], batch_size=EVAL_BATCH_SIZE, shuffle=False)
    model_proto = build_proto_model(args, baseline_args, d.num_features, d.num_classes, device)
    if args.get('head_phi') == 'bounded':
        model_proto.bound_head()
    if args.get('mask_layer_cost'):
        model_proto.set_mask_layer_cost([float(c) for c in args['mask_layer_cost'].split(',')])

    optimizer = build_optimizer(model_proto, args.get('optimizer'), args['lr'], args['l2'],
                                **({'muon_lr': args['muon_lr']} if args.get('muon_lr') is not None else {}))

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=300, min_lr=1e-5, verbose=True)

    weights = loss_weights(args)
    push_log = []
    if args.get('proto_init') == 'data':          # before centring: the head reads the prototype similarities
        idxs = init_prototypes(model, model_proto, push_loader, device, seed)
        push_log.append({'epoch': -1, 'init': 'data', 'indices': [i.cpu().tolist() for i in idxs]})
    if args.get('node_readout') == 'predicate':    # before the head: it reads the node predicates
        set_mask_temperature(model_proto, args, 0)
        t = center_thresholds(model, model_proto.predicate_phi, push_loader, device,
                              lambda layers_x, layers_y, batch: model_proto.proto.similarity(torch.hstack(layers_y[:-1])))
        print('node predicate thresholds centred, median tolerance', float(model_proto.tolerance().median()))
    if args.get('head_phi') == 'bounded' and args.get('head_threshold', 'center') == 'center':
        set_mask_temperature(model_proto, args, 0)     # before the bias: it reads the literals
        t = center_thresholds(model, model_proto.fc.phi_in, push_loader, device,
                                   lambda layers_x, layers_y, batch: model_proto.head_input(model_proto.readout(layers_y[:-1], batch)))
        print('head thresholds centred, median t', float(t.median()))
    if args.get('head_init') == 'center':
        set_mask_temperature(model_proto, args, 0)
        print('head bias centred, shift', center_head_bias(model, model_proto, push_loader, device).tolist())
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(args['epochs']):
        set_mask_temperature(model_proto, args, epoch)
        beta = sharpness(args, epoch)
        if args.get('harden_to') is not None:
            model_proto.set_sharpness(beta)
        train_loss, train_acc = train_epoch(model, model_proto, train_loader, device, optimizer, d.num_classes,
                                            train_full=epoch > args['warmup_epochs'], **weights)

        # ProtoPNet push: snap every prototype onto the closest real training example,
        # so each one *is* an observed pattern and the extracted rules stay readable.
        # Only after warmup - before that the trunk still moves, so the features a
        # prototype would be snapped to are not yet meaningful.
        if args['push_every'] and epoch > args['warmup_epochs'] and epoch % args['push_every'] == 0:
            idxs = model_proto.project_prototypes(push_loader, device)
            push_log.append({'epoch': epoch, 'indices': [i.cpu().tolist() for i in idxs]})

        val_loss, _ = train_epoch(model, model_proto, val_loader, device, None, d.num_classes,
                                  train_full=epoch > args['warmup_epochs'], **weights)
        val_acc = test_epoch(model_proto, val_loader, device)
        test_acc = test_epoch(model_proto, test_loader, device)

        if epoch > args['warmup_epochs']:
            scheduler.step(val_acc)
        # masked prototypes are only ANDs once T is low: keep no checkpoint before that
        ckpt_open = args.get('mask_ckpt_temp') is None or \
            all(p.mask_temp <= args['mask_ckpt_temp'] + 1e-9 for p in model_proto.proto_layers)
        # hardened models: only once fully sharpened
        ckpt_open = ckpt_open and (args.get('harden_to') is None or beta >= args['harden_to'] - 1e-9)
        if epoch>args['warmup_epochs'] and ckpt_open and val_acc >= best_val_acc:
            torch.save(model_proto, os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc

        # train_loss / train_acc: one entry per conv layer (distillation), then the head
        tracking.log_metrics({'val_acc': val_acc, 'test_acc': test_acc, 'best_val_acc': best_val_acc,
                              'best_test_acc': best_test_acc, 'train_loss': train_loss[-1], 'val_loss': val_loss[-1],
                              'train_acc': train_acc[-1], 'lr': optimizer.param_groups[0]['lr'],
                              'phase_full': float(epoch > args['warmup_epochs']), 'sharpness': beta,
                              **{f'distill/L{i}_loss': l for i, l in enumerate(train_loss[:-1])},
                              **{f'distill/L{i}_val_loss': l for i, l in enumerate(val_loss[:-1])},
                              **{f'distill/L{i}_acc': c for i, c in enumerate(train_acc[:-1])}},
                             step=epoch)
        if epoch_callback is not None:
            epoch_callback(epoch, model_proto, val_loader)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

    torch.save(model_proto, os.path.join(path, 'last.pt'))
    # Which training graph/node each prototype ended up standing for.
    with open(os.path.join(path, 'push_log.json'), 'w') as f:
        json.dump(push_log, f)
    model_proto = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_proto, val_loader, device)
    test_acc = test_epoch(model_proto, test_loader, device)

    # How each prototype separates the classes (agreement.{png,json,npz} + MLflow), for
    # every run: Optuna trials and final folds go through here too.
    from utils.agreement import log_agreement
    log_agreement(model_proto, {'train': push_loader, 'val': val_loader, 'test': test_loader}, device, path,
                  title=f"{dataset_name} {args['proto_level']} seed {seed}")

    # Final evaluation of the kept checkpoint: imbalance-aware metrics and the size of the
    # extracted explanation (rules per unit, shortest rule, fidelity), then registration.
    register_final(model_proto, d.loaders, device, path, f"logix-proto-{dataset_name}-{args['proto_level']}",
                   ('models_proto',), ('args.json', 'push_log.json'),
                   {'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc, 'path': path})
    return {'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc}


def eval_seed(dataset_name, baseline_path, args, seed, device):
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    return eval_student(dataset_name, baseline_path, args, seed, device, run_dir(dataset_name, args, baseline_args, seed),
                        f'Evaluating prototype logic model on {dataset_name}')


def train_eval(dataset_name, baseline_path, args):
    device = get_device()
    baseline_args = json.load(open(os.path.join(baseline_path, '0', 'args.json'), 'r'))
    seed = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    if args.get('pool_ops') is None:               # default readout: keep existing dir names
        args.pop('pool_ops', None)
    path = run_dir(dataset_name, args, baseline_args)
    teacher = lambda s: os.path.join(baseline_path, str(s))  # noqa: E731
    level = args['proto_level']
    return run_seeds(path, lambda s: train_seed(dataset_name, teacher(s), args, s, device),
                     lambda s: eval_seed(dataset_name, teacher(s), args, s, device), seed, only_eval,
                     lambda df, ret: log_summary(dataset_name, f'proto/{level}/summary', args, df, ret, path, PROTO_ROOT,
                                                 {'proto_level': level}))


def get_parser():
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
    parser.add_argument('--mask_layer_cost', default=None,      type=str,   help='Comma-separated cared-bit cost per conv layer, e.g. 0.25,1,1 (default: 1 each, with --proto_mask)')
    parser.add_argument('--vocab_reg',      default=None,       type=float, help='Weight of the shared-vocabulary penalty: fraction of trunk units any prototype cares about (with --proto_mask)')
    parser.add_argument('--mask_ckpt_temp', default=None,       type=float, help='Keep checkpoints only once T <= this (default: always, with --proto_mask)')
    parser.add_argument('--harden_to',      default=None,       type=float, help='Gradual hardening: final sharpness of literals and conv units (default: off)')
    parser.add_argument('--harden_start_frac', default=None,    type=float, help='Fraction of the epochs where hardening starts (default: end of warmup, with --harden_to)')
    parser.add_argument('--harden_end_frac', default=None,      type=float, help='Fraction of the epochs where the final sharpness is reached (default 0.9, with --harden_to)')
    parser.add_argument('--proto_init',     default='random',   choices=['random', 'data'], help='random (upstream uniform logits) or data: prototypes set to training nodes / graphs of the teacher, farthest-point sampled')
    parser.add_argument('--head_init',      default=None,       choices=['center'], help='center: head bias set so each class\'s mean pre-activation on the training set is 0 (default: random, upstream)')
    parser.add_argument('--head_phi',       default=None,       choices=['bounded'], help='bounded: head literals are thresholds x >= t (slope <= pi, t in (0, 1)); default: the periodic Phi of upstream')
    parser.add_argument('--head_threshold', default=None,       choices=['center', 'mid'], help='Init of the bounded head thresholds: center (default, median of each column on the training set) or mid (0.5); with --head_phi bounded')
    parser.add_argument('--teacher_eval',   action='store_true',             help='Frozen teacher in eval mode while distilling, so its dropout does not reach the distilled states (default: train mode, upstream)')
    parser.add_argument('--logit_loss',     action='store_true',             help='BCE of the task loss on the head logits, so a saturated head keeps its gradient (default: on the sigmoid output, upstream)')
    parser.add_argument('--optimizer',      default=None,       choices=['adamw', 'muon'], help='Default adamw (upstream); muon: Muon on the conv weights (utils/optim.py)')
    parser.add_argument('--muon_lr',        default=None,       type=float, help='Muon learning rate (default 0.02, with --optimizer muon)')
    parser.add_argument('--node_readout',   default=None,       choices=['predicate'], help='predicate (with --proto_level node --proto_mask): the head reads, per prototype, how many / which share of the nodes satisfy its rule (models_proto/model_proto_count.py); default: pooled similarities')
    parser.add_argument('--pool_ops',       default=None,       type=str,   help='Comma-separated readout pooling, e.g. mean,max,sum (default mean,max; sum: see models_proto/model_proto.py)')
    parser.add_argument('--only_eval',     action='store_true',             help='Only evaluate')
    parser.add_argument('--seed',           default=None,       type=int,   help='Single seed to run')
    return parser


def parse_cli(argv=None):
    """(dataset, baseline_path or None, args) from a command line, normalised like the
    results paths expect; optimize_optuna.py builds its trials through it too."""
    parser = get_parser()
    args = parser.parse_args(argv).__dict__
    if not args['proto_mask']:            # keep the results paths of unmasked configurations unchanged
        for k in ('proto_mask', 'mask_reg', 'mask_temp_start', 'mask_temp_end'):
            args.pop(k)
    for k in ('mask_anneal_frac', 'mask_ckpt_temp', 'mask_layer_cost', 'vocab_reg'):      # nor those of earlier masked ones
        if args[k] is None or not args.get('proto_mask'):
            args.pop(k)
    hardened = args['harden_to'] is not None
    for k in ('harden_to', 'harden_start_frac', 'harden_end_frac'):     # off: results paths unchanged
        if args[k] is None or not hardened:
            args.pop(k)
    for k in ('head_init', 'logit_loss', 'teacher_eval'):            # off: results paths unchanged
        if not args[k]:
            args.pop(k)
    if args['head_phi'] is None:                                     # off: results paths unchanged
        args.pop('head_phi')
        args.pop('head_threshold')
    elif args['head_threshold'] is None:
        args['head_threshold'] = 'center'
    if args['node_readout'] is None:                                 # off: results paths unchanged
        args.pop('node_readout')
    elif args['proto_level'] != 'node' or not args.get('proto_mask') or args.get('pool_ops'):
        parser.error('--node_readout predicate needs --proto_level node --proto_mask and no --pool_ops')
    if args['proto_init'] == 'random':
        args.pop('proto_init')
    if args['optimizer'] in (None, 'adamw'):                         # default: results paths unchanged
        args.pop('optimizer')
    if args['muon_lr'] is None or 'optimizer' not in args:
        args.pop('muon_lr')
    if args.get('harden_to') is not None and args['harden_to'] < 1:
        parser.error('--harden_to must be >= 1 (1 = no hardening)')
    if args.get('mask_ckpt_temp') is not None and args['mask_ckpt_temp'] < args['mask_temp_end']:
        parser.error('--mask_ckpt_temp below --mask_temp_end would never keep a checkpoint')
    return args.pop('dataset'), args.pop('baseline_path'), args


if __name__ == '__main__':
    dataset_name, baseline_path, args = parse_cli()
    if baseline_path is None:
        baseline_path = best_teacher_dir(dataset_name)
        print('Baseline path found:', baseline_path)
    train_eval(dataset_name, baseline_path, args)
