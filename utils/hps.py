"""Hyper-parameter search (Optuna) for the LogiX-GIN students, one study per
(dataset, model, fold).

What is searched
    ``MODELS`` maps the six students of the sum-pooling ablation to their training script,
    the arguments that define them (``fixed``) and the ranges searched (``space``):

        classic / classic_nosum               train_logic.py, with / without the upstream sum readout
        node_mask_push / node_mask_push_sum   train_proto.py NMP, without / with sum
        graph / graph_sum                     train_proto.py graph-level prototypes, without / with sum

    Each pair shares its search space, so the with/without-sum comparison is tuned the
    same way. The teacher is fixed (the k-fold teacher of the fold), and so are the
    trunk shape it imposes and the split.

Protocol
    A trial trains on the fold's train set, keeps the checkpoint with the best validation
    accuracy (as the training scripts do), and is scored by the validation **balanced**
    accuracy of that checkpoint, so a head stuck on the majority class (AIDS: 0.8
    accuracy) scores 0.5. Test metrics are recorded for every trial but never used by the
    search. Trials report the validation balanced accuracy every ``report_every`` epochs
    after warmup, and the median pruner stops those below the median of earlier trials.

    Per fold (``fold=k`` for every k, nested model selection) every reported test number
    comes from a configuration chosen without that fold's test set. With a single tuning
    fold (fold 0) and the best set re-run on all folds, fold 0's validation graphs are
    test graphs of other folds, a mild optimistic bias to state with the results.

Reproducibility (all under ``results_hps/<dataset>/<model>/fold<k>/``)
    journal.log        the Optuna study (JournalFileBackend: several worker processes)
    study.json         search space, fixed arguments, sampler / pruner settings, objective,
                       teacher, git commit of every worker that joined
    runs/trial<n>/     each trial's run directory (best.pt, args.json, data.pkl)
    trials.csv         every trial: state, value, parameters, val / test metrics, argv
    best.json          best trial: parameters, metrics and the exact command that
                       reproduces it (``argv``)
    Every trial is also an MLflow run (tags hps_study, hps_trial). The sampler is seeded;
    with one worker per study the sequence of trials is deterministic (up to GPU
    nondeterminism), with several it depends on their timing.
"""
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = 'results_hps'

TEACHER_CFG = 'batch_size=128|dropout=0.15|epochs=500|hidden_dim=64|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3|split=kfold'


def _f(low, high, log=False):
    return {'type': 'float', 'low': low, 'high': high, 'log': log}


def _c(*choices):
    return {'type': 'categorical', 'choices': list(choices)}


COMMON = {                        # every student (train_logic.py and train_proto.py)
    'lr': _f(1e-4, 1e-2, log=True),
    'batch_size': _c(16, 32, 64, 128),
    'l2': _c(0.0, 1e-4),
    'conv_reg': _f(1e-4, 1e-1, log=True),
    'fc_reg': _f(1e-4, 1e-1, log=True),
    'epochs': _c(500, 1000, 2000),
    'warmup_frac': _c(0.2, 0.33, 0.5),         # --warmup_epochs = round(frac * epochs)
}
PROTO = {
    'num_prototypes': _c(8, 16, 32),
    'proto_div_reg': _f(1e-3, 1e-1, log=True),
    'proto_ent_reg': _f(1e-3, 1e-1, log=True),
}
NMP = {
    'mask_reg': _f(0.05, 1.0, log=True),
    'push_every': _c(50, 100, 200),
    'vocab_reg': _c(0.0, 0.1),
}
NMP_FIXED = {'proto_level': 'node', 'proto_mask': True, 'mask_anneal_frac': 0.8, 'mask_ckpt_temp': 1.0}
SUM = {'pool_ops': 'mean,max,sum'}

MODELS = {
    'classic':            {'script': 'logic', 'fixed': {}, 'space': COMMON},
    'classic_nosum':      {'script': 'logic', 'fixed': {'pool_ops': 'mean,max'}, 'space': COMMON},
    'node_mask_push':     {'script': 'proto', 'fixed': NMP_FIXED, 'space': {**COMMON, **PROTO, **NMP}},
    'node_mask_push_sum': {'script': 'proto', 'fixed': {**NMP_FIXED, **SUM}, 'space': {**COMMON, **PROTO, **NMP}},
    'graph':              {'script': 'proto', 'fixed': {'proto_level': 'graph', 'push_every': 0},
                           'space': {**COMMON, **PROTO}},
    'graph_sum':          {'script': 'proto', 'fixed': {'proto_level': 'graph', 'push_every': 0, **SUM},
                           'space': {**COMMON, **PROTO}},
}

SAMPLER = {'name': 'TPESampler', 'multivariate': True, 'n_startup_trials': 10}
PRUNER = {'name': 'MedianPruner', 'n_startup_trials': 8, 'n_warmup_steps': 0}
OBJECTIVE = 'val_balanced_acc of the checkpoint kept by validation accuracy (maximise)'


def git_commit():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                             cwd=REPO, text=True).strip())
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return None


def study_dir(dataset, model, fold, root=ROOT):
    return os.path.join(root, dataset, model, f'fold{fold}')


def teacher_path(dataset):
    return os.path.join('results', dataset, TEACHER_CFG)


def suggest(trial, space):
    params = {}
    for name, s in space.items():
        if s['type'] == 'float':
            params[name] = trial.suggest_float(name, s['low'], s['high'], log=s['log'])
        else:
            params[name] = trial.suggest_categorical(name, s['choices'])
    return params


def to_argv(dataset, model, params, fold):
    """The training script's command line for a parameter set (what best.json stores)."""
    spec = MODELS[model]
    p = dict(params)
    if 'warmup_frac' in p:
        p['warmup_epochs'] = int(round(p.pop('warmup_frac') * p['epochs']))
    argv = ['--dataset', dataset, '--baseline_path', teacher_path(dataset), '--seed', str(fold)]
    for k, v in {**spec['fixed'], **p}.items():
        if v is True:
            argv.append(f'--{k}')
        elif v is not False and v is not None:
            argv += [f'--{k}', str(v)]
    return argv


def script_module(model):
    if MODELS[model]['script'] == 'logic':
        import train_logic as m
    else:
        import train_proto as m
    return m


def run_dir(model, dataset, args, teacher_fold_path, seed, root):
    """Seed directory a train_seed call with these arguments writes to."""
    m = script_module(model)
    baseline_args = json.load(open(os.path.join(teacher_fold_path, 'args.json')))
    if MODELS[model]['script'] == 'logic':
        from utils.utils import create_folder_logic
        return create_folder_logic(dataset, args, baseline_args, seed=seed, root=root)
    return m.create_folder_proto(dataset, args, baseline_args, seed=seed, root=root)


def study_attrs(dataset, model, fold):
    return {'dataset': dataset, 'model': model, 'fold': fold, 'script': f"train_{MODELS[model]['script']}.py",
            'fixed': MODELS[model]['fixed'], 'search_space': MODELS[model]['space'],
            'sampler': SAMPLER, 'pruner': PRUNER, 'objective': OBJECTIVE,
            'teacher': os.path.join(teacher_path(dataset), str(fold)), 'split': 'kfold'}


# ---------------------------------------------------------------------------
# study, objective, worker
# ---------------------------------------------------------------------------

def open_study(dataset, model, fold, root=ROOT, sampler_seed=0, register=True):
    """Create or join the study. Refuses to join one whose search space differs from the
    current code, so a study never mixes two spaces. ``register`` records this checkout's
    git commit (workers); exports only read."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    d = study_dir(dataset, model, fold, root)
    os.makedirs(d, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(os.path.join(d, 'journal.log')))
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, multivariate=SAMPLER['multivariate'],
                                         n_startup_trials=SAMPLER['n_startup_trials'])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=PRUNER['n_startup_trials'],
                                         n_warmup_steps=PRUNER['n_warmup_steps'])
    name = f'{dataset}/{model}/fold{fold}'
    study = optuna.create_study(study_name=name, storage=storage, direction='maximize',
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    attrs = {**study_attrs(dataset, model, fold), 'sampler_seed': sampler_seed}
    if 'search_space' in study.user_attrs:
        old = {k: study.user_attrs.get(k) for k in ('search_space', 'fixed', 'objective')}
        new = {k: json.loads(json.dumps(attrs[k])) for k in old}
        if old != new:
            raise RuntimeError(f'{name}: the stored search space differs from utils/hps.py; '
                               f'use another --root for a new space')
    else:
        for k, v in attrs.items():
            study.set_user_attr(k, v)
    if register:
        commits = set(study.user_attrs.get('git_commits', [])) | {git_commit()}
        study.set_user_attr('git_commits', sorted(c for c in commits if c))
        with open(os.path.join(d, 'study.json'), 'w') as f:
            json.dump(dict(study.user_attrs), f, indent=1)
    return study


class PruneCallback:
    """epoch_callback of train_seed: validation balanced accuracy every ``every`` epochs
    after warmup, reported to the trial; raises TrialPruned when the pruner says so."""

    def __init__(self, trial, warmup, device, every=50):
        self.trial, self.warmup, self.device, self.every = trial, warmup, device, every

    def __call__(self, epoch, model, val_loader):
        import optuna
        from utils import tracking
        from utils.evaluation import evaluate
        if epoch <= self.warmup or (epoch - self.warmup) % self.every:
            return
        v = evaluate(model, val_loader, self.device)['balanced_acc']
        self.trial.report(v, step=epoch)
        if self.trial.should_prune():
            tracking.set_tags({'hps_pruned_at': epoch})
            raise optuna.TrialPruned(f'val balanced acc {v:.3f} at epoch {epoch}')


def objective(trial, dataset, model, fold, root, device, report_every=50):
    import pickle
    import torch
    from torch_geometric.loader import DataLoader
    from utils.evaluation import evaluate
    params = suggest(trial, MODELS[model]['space'])
    argv = to_argv(dataset, model, params, fold)
    trial.set_user_attr('argv', argv)
    trial.set_user_attr('git_commit', git_commit())
    m = script_module(model)
    ds, baseline_path, args = m.parse_cli(argv)
    seed = args.pop('seed')
    args.pop('only_eval', None)
    if args.get('pool_ops') is None:                     # as train_eval does
        args.pop('pool_ops', None)
    teacher = os.path.join(baseline_path, str(fold))
    out_root = os.path.join(study_dir(dataset, model, fold, root), 'runs', f'trial{trial.number:03d}')
    tags = {'hps_study': f'{dataset}/{model}/fold{fold}', 'hps_trial': trial.number, 'kind': 'hps'}
    m.train_seed(ds, teacher, args, seed, device, out_root=out_root,
                 epoch_callback=PruneCallback(trial, args['warmup_epochs'], device, report_every), tags=tags)
    path = run_dir(model, ds, args, teacher, seed, out_root)
    trial.set_user_attr('run_dir', path)
    ckpt = os.path.join(path, 'best.pt')
    if not os.path.exists(ckpt):                         # no checkpoint was ever kept
        raise RuntimeError(f'no best.pt in {path}')
    net = torch.load(ckpt, map_location=device, weights_only=False)
    data = pickle.load(open(os.path.join(path, 'data.pkl'), 'rb'))
    metrics = {**evaluate(net, DataLoader(data['val_dataset'], batch_size=64), device, prefix='val_'),
               **evaluate(net, DataLoader(data['test_dataset'], batch_size=64), device, prefix='test_')}
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    return metrics['val_balanced_acc']


def run_worker(dataset, model, fold, n_trials, root=ROOT, sampler_seed=0, report_every=50):
    """Run trials until the study holds n_trials finished ones (complete, pruned or failed);
    several workers on one study share that budget."""
    import optuna
    import torch
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    study = open_study(dataset, model, fold, root, sampler_seed)
    done = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    if len(study.get_trials(deepcopy=False, states=done)) >= n_trials:
        return study
    study.optimize(lambda t: objective(t, dataset, model, fold, root, device, report_every),
                   callbacks=[MaxTrialsCallback(n_trials, states=done)], catch=(Exception,), gc_after_trial=True)
    export_study(study, study_dir(dataset, model, fold, root))
    return study


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_study(study, d):
    """trials.csv and best.json of one study."""
    from optuna.trial import TrialState
    rows = []
    for t in study.get_trials(deepcopy=False):
        rows.append({'number': t.number, 'state': t.state.name, 'value': t.value,
                     'duration_s': t.duration.total_seconds() if t.duration else None,
                     'last_step': t.last_step,
                     **{f'param_{k}': v for k, v in t.params.items()},
                     **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in t.user_attrs.items()}})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(d, 'trials.csv'), index=False)
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not complete:
        return None
    b = study.best_trial
    best = {'study': study.study_name, 'number': b.number, 'value': b.value, 'params': b.params,
            'argv': b.user_attrs.get('argv'), 'run_dir': b.user_attrs.get('run_dir'),
            'metrics': {k: v for k, v in b.user_attrs.items() if k.startswith(('val_', 'test_'))},
            'n_complete': len(complete),
            'n_pruned': len(study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))),
            'n_failed': len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,))),
            'git_commit': b.user_attrs.get('git_commit'), 'study_attrs': dict(study.user_attrs)}
    with open(os.path.join(d, 'best.json'), 'w') as f:
        json.dump(best, f, indent=1)
    return best


def export_all(root=ROOT):
    """Re-export every study under root and write root/best_params.csv (one row per study)."""
    import glob
    import pandas as pd
    rows = []
    for journal in sorted(glob.glob(os.path.join(root, '*', '*', 'fold*', 'journal.log'))):
        d = os.path.dirname(journal)
        ds, model, fold = d.split(os.sep)[-3:]
        study = open_study(ds, model, int(fold[4:]), root, register=False)
        best = export_study(study, d)
        if best:
            rows.append({'dataset': ds, 'model': model, 'fold': int(fold[4:]), 'value': best['value'],
                         **best['metrics'], 'n_complete': best['n_complete'], 'n_pruned': best['n_pruned'],
                         'n_failed': best['n_failed'], 'params': json.dumps(best['params']),
                         'argv': ' '.join(best['argv'] or [])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, 'best_params.csv'), index=False)
    return df
