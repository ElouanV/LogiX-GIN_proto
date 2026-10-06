"""Experiments from YAML configs (configs/experiments/*.yaml): a dataset's whole pipeline
on one machine, run as units with dependencies, reported to the progress table.

Units of a dataset D (key ``<experiment>/<D>/<stage>[/<model>][/fold<k>]``):

    teacher/fold<k>            train_baseline.py --seed k with the config's teacher arguments
    teacher_summary            train_baseline.py --only_eval (results.json, total_results.csv)
    hps/<model>/fold<t>        one Optuna study per tuning fold t (utils/hps.py)
    final/<model>/fold<k>      fold k re-trained with the best set of its study (the study of
                               fold k if k is a tuning fold, else of the first tuning fold)
    summary                    mean ± std per model over the final folds (CPU)

A unit runs when the units it needs are done (an hps study needs only its teacher fold),
each in its own process (``run_experiment.py <config> unit ...``), ``jobs`` at a time.
Units already done on disk are skipped, so a run resumes where it stopped. Outputs:
teachers in results/<D>/<teacher cfg>/, studies in <hps.root>/<D>/<model>/fold<t>/,
final runs in <final.root>/<experiment>/<D>/<model>/fold<k>/ (final.json: metrics,
command, study it came from, git commit), the summary in
<final.root>/<experiment>/<D>/summary.csv. Each run also copies the config it was started
with (and its git commit) to <final.root>/<experiment>/config_<time>.yaml.
"""
import concurrent.futures as cf
import datetime
import json
import os
import pickle
import shutil
import subprocess
import sys

import optuna
import yaml

from utils import hps, progress
from utils.splits import same_split, split_file

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Unit:
    def __init__(self, stage, dataset, model=None, fold=None, deps=()):
        self.stage, self.dataset, self.model, self.fold, self.deps = stage, dataset, model, fold, tuple(deps)

    @property
    def key(self):
        return progress.unit_key('', self.dataset, self.stage, self.model, self.fold).lstrip('/')

    def argv(self):
        a = ['--stage', self.stage, '--dataset', self.dataset]
        if self.model:
            a += ['--model', self.model]
        if self.fold is not None:
            a += ['--fold', str(self.fold)]
        return a


class Experiment:
    def __init__(self, path):
        self.path = path
        self.cfg = yaml.safe_load(open(path))
        self.name = self.cfg['name']
        unknown = set(self.cfg['models']) - set(hps.MODELS)
        if unknown:
            raise ValueError(f'models not in configs/models.yaml: {sorted(unknown)}')
        self.folds = list(self.cfg['folds'])
        self.tune_folds = list(self.cfg['hps']['tune_folds'])

    # ---- locations ----
    def teacher_dir(self, dataset):
        import train_baseline
        from utils.utils import create_folder
        _, args = train_baseline.parse_cli(['--dataset', dataset, *self._teacher_argv()])
        args.pop('seed'); args.pop('only_eval')
        if args.get('split', 'random') == 'random':
            args.pop('split', None)
        return os.path.join('results', dataset,
                            '|'.join(f'{k}={args[k]}' for k in sorted(args)))   # utils.create_folder's name

    def _teacher_argv(self):
        argv = []
        for k, v in self.cfg['teacher'].items():
            if v is True:
                argv.append(f'--{k}')
            elif v is not False and v is not None:
                argv += [f'--{k}', str(v)]
        return argv

    def final_dir(self, dataset, model=None, fold=None):
        d = os.path.join(self.cfg['final']['root'], self.name, dataset)
        if model:
            d = os.path.join(d, model)
        if fold is not None:
            d = os.path.join(d, f'fold{fold}')
        return d

    def source_study(self, fold):
        return fold if fold in self.tune_folds else self.tune_folds[0]

    # ---- plan ----
    def units(self, datasets):
        out = []
        for ds in datasets:
            teach = [Unit('teacher', ds, fold=k) for k in self.folds]
            out += teach
            out.append(Unit('teacher_summary', ds, deps=[u.key for u in teach]))
            for m in self.cfg['models']:
                studies = [Unit('hps', ds, m, t, deps=[Unit('teacher', ds, fold=t).key]) for t in self.tune_folds]
                out += studies
                for k in self.folds:
                    src = Unit('hps', ds, m, self.source_study(k)).key
                    out.append(Unit('final', ds, m, k, deps=[src, Unit('teacher', ds, fold=k).key]))
            out.append(Unit('summary', ds, deps=[u.key for u in out if u.dataset == ds and u.stage == 'final']))
        return out

    def is_done(self, u):
        if u.stage == 'teacher':
            return os.path.exists(os.path.join(self.teacher_dir(u.dataset), str(u.fold), 'last.pt'))
        if u.stage == 'teacher_summary':
            f = os.path.join(self.teacher_dir(u.dataset), 'total_results.csv')
            if not os.path.exists(f):
                return False
            import pandas as pd
            return set(self.folds) <= set(pd.read_csv(f)['seed'])
        if u.stage == 'hps':
            from optuna.trial import TrialState
            if not os.path.exists(os.path.join(hps.study_dir(u.dataset, u.model, u.fold, self.cfg['hps']['root']),
                                               'journal.log')):
                return False
            st = hps.open_study(u.dataset, u.model, u.fold, self.cfg['hps']['root'], register=False,
                                teacher=self.teacher_dir(u.dataset))
            done = st.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))
            return len(done) >= self.cfg['hps']['n_trials']
        if u.stage == 'final':
            return os.path.exists(os.path.join(self.final_dir(u.dataset, u.model, u.fold), 'final.json'))
        if u.stage == 'summary':
            return False                                   # cheap: always refreshed
        raise ValueError(u.stage)

    # ---- unit execution (in the unit's own process) ----
    def run_unit(self, stage, dataset, model=None, fold=None):
        started = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        rep = dict(experiment=self.name, dataset=dataset, stage=stage, model=model, fold=fold,
                   git_commit=hps.git_commit(), started=started)
        progress.report(status='running', **rep)
        try:
            metrics, prog = getattr(self, f'_unit_{stage}')(dataset, model, fold, rep)
        except BaseException as e:
            progress.report(status='failed', note=repr(e)[:500], **rep)
            raise
        progress.report(status='done', metrics=metrics, progress=prog, **rep)

    def _unit_teacher(self, dataset, model, fold, rep):
        import train_baseline
        ds, args = train_baseline.parse_cli(['--dataset', dataset, *self._teacher_argv(), '--seed', str(fold)])
        res = train_baseline.train_eval(ds, args)
        return {'test_acc': res['test_acc_mean']}, None

    def _unit_teacher_summary(self, dataset, model, fold, rep):
        import train_baseline
        ds, args = train_baseline.parse_cli(['--dataset', dataset, *self._teacher_argv(), '--only_eval'])
        res = train_baseline.train_eval(ds, args)
        return {'test_acc': res['test_acc_mean']}, f"test {res['test_acc_mean']:.3f} ± {res['test_acc_std']:.3f}"

    def _unit_hps(self, dataset, model, fold, rep):
        h = self.cfg['hps']
        n = h['n_trials']

        def on_trial(study, trial):
            from optuna.trial import TrialState
            finished = [t for t in study.get_trials(deepcopy=False)
                        if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
            best = study.best_trial if any(t.state == TrialState.COMPLETE for t in finished) else None
            progress.report(status='running', progress=f'{len(finished)}/{n} trials',
                            metrics={'val_balanced_acc': best.value} if best else None, **rep)

        study = hps.run_worker(dataset, model, fold, n, h['root'], h['sampler_seed'], h['report_every'],
                               teacher=self.teacher_dir(dataset), on_trial=on_trial, sole_worker=True)
        best = hps.export_study(study, hps.study_dir(dataset, model, fold, h['root']))
        if best is None:
            raise RuntimeError('no trial completed')
        return best['metrics'], f"{best['n_complete']} complete, {best['n_pruned']} pruned, best #{best['number']}"

    def _unit_final(self, dataset, model, fold, rep):
        import torch
        src = self.source_study(fold)
        best = json.load(open(os.path.join(hps.study_dir(dataset, model, src, self.cfg['hps']['root']), 'best.json')))
        argv = hps.to_argv(dataset, model, best['params'], fold, self.teacher_dir(dataset))
        d = self.final_dir(dataset, model, fold)
        if os.path.exists(os.path.join(d, 'run')):
            shutil.rmtree(os.path.join(d, 'run'))          # an interrupted attempt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path, _ = hps.train_from_argv(model, argv, os.path.join(d, 'run'), device,
                                      tags={'experiment': self.name, 'kind': 'final', 'model_name': model})
        metrics = hps.evaluate_run(path, device)
        split_sha = same_split(dataset, fold, pickle.load(open(os.path.join(path, 'data.pkl'), 'rb')))
        with open(os.path.join(d, 'final.json'), 'w') as f:
            json.dump({'experiment': self.name, 'dataset': dataset, 'model': model, 'fold': fold,
                       'metrics': metrics, 'argv': argv, 'params': best['params'], 'run_dir': path,
                       'study': best['study'], 'study_trial': best['number'], 'git_commit': hps.git_commit(),
                       'split_file': os.path.relpath(split_file(dataset), REPO), 'split_sha256': split_sha},
                      f, indent=1)
        return metrics, None

    def _unit_summary(self, dataset, model, fold, rep):
        df = self.summary(dataset)
        return {}, f'{len(df)} models' if df is not None else 'no final run yet'

    def summary(self, dataset):
        import glob
        import pandas as pd
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(self.final_dir(dataset), '*', 'fold*', 'final.json'))]
        if not rows:
            return None
        df = pd.DataFrame([{'model': r['model'], 'fold': r['fold'], **r['metrics']} for r in rows])
        cols = [c for c in df.columns if c.startswith(('val_', 'test_')) and not c.endswith('_n')]
        agg = df.groupby('model')[cols].agg(['mean', 'std'])
        agg.columns = [f'{c}_{s}' for c, s in agg.columns]
        agg.insert(0, 'n_folds', df.groupby('model').size())
        agg = agg.reindex([m for m in self.cfg['models'] if m in agg.index])
        agg.to_csv(os.path.join(self.final_dir(dataset), 'summary.csv'))
        return agg

    # ---- scheduling (orchestrator process) ----
    def run(self, datasets, stages=None, jobs=None, dry=False):
        jobs = jobs or self.cfg.get('jobs', 1)
        units = [u for u in self.units(datasets) if stages is None or u.stage in stages]
        by_key = {u.key: u for u in units}
        done = {u.key for u in units if self.is_done(u)}
        todo = [u for u in units if u.key not in done]
        # a unit outside the selected stages counts as satisfied only if done on disk
        all_units = {u.key: u for u in self.units(datasets)}
        for u in todo:
            for d in u.deps:
                if d not in by_key and not self.is_done(all_units[d]):
                    raise RuntimeError(f'{u.key} needs {d}, which is neither done nor selected')
        print(f'{self.name}: {len(units)} units, {len(done)} done, {len(todo)} to run, {jobs} at a time')
        if dry:
            for u in todo:
                print('  ', u.key)
            return True
        self._snapshot()
        failed, running = set(), {}
        log_dir = os.path.join('logs', self.name)
        with cf.ThreadPoolExecutor(jobs) as pool:
            while todo or running:
                for u in list(todo):
                    if len(running) >= jobs:
                        break
                    if any(d in failed for d in u.deps):
                        todo.remove(u)
                        failed.add(u.key)
                        progress.report(self.name, u.dataset, u.stage, 'skipped', u.model, u.fold,
                                        note='a unit it needs failed')
                        continue
                    if all(d in done or d not in by_key for d in u.deps):
                        todo.remove(u)
                        running[pool.submit(self._spawn, u, log_dir)] = u
                if not running:
                    break
                fin, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                for f in fin:
                    u = running.pop(f)
                    ok = f.result()
                    (done if ok else failed).add(u.key)
                    print(f"[{datetime.datetime.now():%H:%M:%S}] {u.key}: {'done' if ok else 'FAILED'}", flush=True)
        print(f'{self.name}: {len(failed)} failed' if failed else f'{self.name}: all done')
        return not failed

    def _spawn(self, u, log_dir):
        log = os.path.join(log_dir, u.key.replace('/', '_') + '.log')
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = [sys.executable, os.path.join(REPO, 'run_experiment.py'), self.path, 'unit', *u.argv()]
        with open(log, 'a') as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode == 0

    def _snapshot(self):
        d = os.path.join(self.cfg['final']['root'], self.name)
        os.makedirs(d, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        with open(os.path.join(d, f'config_{stamp}.yaml'), 'w') as f:
            f.write(f'# started {stamp} on {progress.machine()}, git {hps.git_commit()}\n')
            f.write(open(self.path).read())

    def status(self, datasets):
        """Per dataset and stage: done / total units, from the files on disk."""
        rows = {}
        for u in self.units(datasets):
            if u.stage == 'summary':
                continue
            r = rows.setdefault((u.dataset, u.stage), [0, 0])
            r[1] += 1
            r[0] += self.is_done(u)
        return rows

    def plan(self, datasets):
        """Report every unit not done yet as queued, so the progress table lists the work left."""
        n = 0
        for u in self.units(datasets):
            if u.stage != 'summary' and not self.is_done(u):
                progress.report(self.name, u.dataset, u.stage, 'queued', u.model, u.fold)
                n += 1
        return n
