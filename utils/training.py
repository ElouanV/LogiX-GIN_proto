"""Seed orchestration shared by the training scripts (train_baseline.py, train_logic.py,
train_proto.py): run directories, split loaders, the frozen teacher of a student, the
per-seed MLflow wrapping, the final evaluation of the kept checkpoint and the
mean ± std summary over seeds. Each script keeps only its model and its epoch loop.
"""
import json
import os
import pickle
import shutil
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from utils import tracking
from utils.datasets import get_dataset
from utils.evaluation import evaluate
from utils.runs import load_teacher, make_loaders, num_features, save_split, split_datasets, write_args
from utils.utils import set_seed

NUM_SEEDS = 10


def fresh_dir(make):
    """``make()`` (a run directory, created), emptied of any earlier run."""
    shutil.rmtree(make())
    return make()


@torch.no_grad()
def test_epoch(model, loader, device):
    """Accuracy of a LogiX-GIN model (sharp literals: tau=1000); skips empty / NaN batches."""
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
    return total_correct / len(loader.dataset)


def distill_trunk(student, layers_x, layers_y, edge_index, train, train_full, conv_reg, total_loss, total_correct,
                  n_graphs, n_batches, quant=None):
    """Layer-wise distillation loss of the student's conv trunk on one batch (train_logic.py
    and train_proto.py share it): BCE of each conv layer against the teacher's binary node
    states ``layers_y`` from the teacher's input ``layers_x`` (input dropout 0.2 above layer 0
    while training) plus its literal regularisers; in the ``train_full`` regime also from the
    student's own previous layer (chained). QAT: ``quant`` is applied to a layer's output
    before the next layer reads it in the chained term. Adds each layer's loss / bit accuracy
    to ``total_loss[i]`` / ``total_correct[i]``; returns the summed loss."""
    loss = 0
    last_layer_out = None
    for i, (layer, layer_x, layer_y) in enumerate(zip(student.convs, layers_x[:-1], layers_y[:-1])):
        if train_full: layer.nn[0].phi_in.tau = 10
        if i != 0 and train: layer_x = F.dropout(layer_x, p=0.2)*(1-0.2)
        layer_out = layer(torch.hstack([layer_x, 1-layer_x]), edge_index)
        layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg * (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
        loss += layer_loss
        if train_full and i != 0:
            layer_x = last_layer_out
            layer_out = layer(torch.hstack([layer_x, 1-layer_x]), edge_index)
            layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg * (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
            loss += layer_loss
        last_layer_out = layer_out if quant is None else quant(layer_out)
        total_loss[i] += layer_loss.item() / n_graphs
        total_correct[i] += ((layer_out >= 0.5).long() == layer_y).sum().item() / (layer_y.shape[-2]*layer_y.shape[-1]*n_batches)
    return loss


def class_loss(out, y, num_classes):
    """BCE on the per-class sigmoid outputs + NLL of their log-softmax (the task loss of the
    LogiX-GIN students)."""
    return F.binary_cross_entropy(out.reshape(-1), F.one_hot(y, num_classes=num_classes).float().reshape(-1)) + \
        F.nll_loss(F.log_softmax(out, dim=-1), y.long())


@dataclass
class StudentData:
    """What a student seed trains on: its split of the teacher's data and the frozen teacher."""
    args: dict
    baseline_args: dict
    datasets: dict
    loaders: dict
    teacher: torch.nn.Module
    num_features: int
    num_classes: int


def student_data(dataset_name, baseline_path, args, device, title, path=None):
    """Load the dataset on the teacher's split (otherwise the teacher has seen part of the test
    set) and the frozen teacher, printing ``title`` and the teacher's test accuracy. With
    ``path`` (a fresh run directory), args.json and data.pkl are written there first."""
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    if path is not None:
        args = write_args(path, args)
    dataset = get_dataset(dataset_name)
    print(title)
    print(baseline_args)
    num_classes, n_features = dataset.num_classes, num_features(dataset)
    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))
    datasets = split_datasets(dataset, data)
    loaders = make_loaders(datasets, args['batch_size'])
    if path is not None:
        save_split(path, data, datasets)
    teacher = load_teacher(baseline_path, n_features, num_classes, device)
    print('Baseline Acc:', test_epoch(teacher, loaders['test'], device))
    return StudentData(args, baseline_args, datasets, loaders, teacher, n_features, num_classes)


def eval_student(dataset_name, baseline_path, args, seed, device, path, title):
    """val / test accuracy of the student checkpoint kept at ``path`` (``--only_eval``)."""
    set_seed(seed)
    d = student_data(dataset_name, baseline_path, args, device, title)
    model = torch.load(os.path.join(path, 'best.pt'))
    return {'seed': seed, 'val_acc': test_epoch(model, d.loaders['val'], device),
            'test_acc': test_epoch(model, d.loaders['test'], device)}


def tracked_student_seed(dataset_name, baseline_path, args, seed, path, root, experiment, name, tags, train):
    """``train()`` inside the MLflow runs of one student seed: its pipeline run (one per seed
    when run alone), the teacher reference and the distillation stage run in ``experiment``."""
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    cfg = os.path.relpath(os.path.dirname(path), f'{root}/{dataset_name}')
    params = {**args, **{f'teacher/{k}': v for k, v in baseline_args.items()},
              'seed': seed, 'dataset': dataset_name, 'teacher_path': baseline_path}
    tags = {'config': cfg, 'seed': seed, 'kind': 'seed', 'dataset': dataset_name, **tags}
    with tracking.ensure_pipeline(dataset_name, name, tags=tags):
        tracking.teacher_reference(baseline_path, dataset_name, seed)
        with tracking.run(experiment, run_name=name, params=params, tags=tags, stage='distillation',
                          dataset=dataset_name):
            res = train()
            tracking.write_pipeline(path)
            return res


def register_final(model, loaders, device, path, registered_name, code_dirs, artifacts, tags):
    """Final evaluation of the kept student checkpoint (imbalance-aware metrics, rule
    statistics on val) and its registration, when tracking is on."""
    if not tracking.enabled():
        return
    from rule_eval import rule_metrics, rule_stats
    final = {**evaluate(model, loaders['val'], device, prefix='final/val_'),
             **evaluate(model, loaders['test'], device, prefix='final/test_')}
    stats = rule_stats(model, loaders['val'], device)
    tracking.log_metrics({**final, **rule_metrics(stats, prefix='final/rules/')})
    tracking.log_dict({'metrics': final, 'rules': stats}, 'final_evaluation.json')
    for f in artifacts:
        tracking.log_artifact(os.path.join(path, f))
    tracking.log_model(model, registered_name, code_dirs=code_dirs, tags=tags)


def summarize(results, path):
    """total_results.csv and results.json (mean / std of val and test accuracy over seeds)."""
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))
    ret = {'val_acc_mean': df['val_acc'].mean(), 'test_acc_mean': df['test_acc'].mean(),
           'val_acc_std': df['val_acc'].std(), 'test_acc_std': df['test_acc'].std()}
    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)
    return df, ret


def run_seeds(path, train_seed, eval_seed, seed=None, only_eval=False, log_summary=None, n_seeds=NUM_SEEDS):
    """Train every seed (or only ``seed``), then summarize over seeds into ``path``.

    A single ``seed`` or ``only_eval`` re-evaluates every seed's kept checkpoint (missing ones
    are reported and skipped). Parallel single-seed processes each re-evaluate every seed, so
    only a full or an ``only_eval`` pass calls ``log_summary(df, ret)`` (the summary run
    exists once per configuration)."""
    results = []
    if not only_eval:
        for s in [seed] if seed is not None else range(n_seeds):
            results.append(train_seed(s))
    print(results)
    if only_eval or seed is not None:
        results = []
        for s in range(n_seeds):
            try:
                r = eval_seed(s)
                results.append(r)
                print(r)
            except Exception as e:
                print(e)
    df, ret = summarize(results, path)
    if log_summary is not None and (only_eval or seed is None):
        log_summary(df, ret)
    return ret


def log_summary(dataset_name, run_name, args, df, ret, path, root, tags=None):
    """Mean / std over seeds as a kind=summary run of the dataset's pipeline experiment."""
    with tracking.run(tracking.pipeline_experiment(dataset_name), run_name=run_name,
                      params={**args, 'n_seeds': len(df)},
                      tags={'config': os.path.relpath(path, f'{root}/{dataset_name}'), 'kind': 'summary',
                            'dataset': dataset_name, **(tags or {})}):
        tracking.log_metrics(ret)
        tracking.log_artifact(os.path.join(path, 'total_results.csv'))


def finetune_run(run, kind, stage, cfg, settings, out_dir, level, tags=None):
    """The MLflow run of a fine-tune of ``run`` (a utils.runs.RunDir): experiment
    ``<kind>/<dataset>``, nested under the source run's pipeline (tracking.fine_tune_parent),
    with the fine-tune ``settings`` and the source run's arguments as params."""
    return tracking.run(f'{kind}/{run.dataset}', run_name=f'{level}|{cfg}/seed{run.seed}',
                        params={**settings, **{f'source/{k}': v for k, v in run.args.items()}},
                        tags={'config': cfg, 'source_config': f'{run.config}/{run.teacher_config}', 'seed': run.seed,
                              'kind': 'seed', 'dataset': run.dataset, 'proto_level': level,
                              'source_run_path': run.path, 'out_dir': out_dir, **(tags or {})},
                        stage=stage, dataset=run.dataset, parent_run_id=tracking.fine_tune_parent(run.path))
