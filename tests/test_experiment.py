"""utils/experiment.py and utils/progress.py (no training).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_experiment.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import hps, progress
    from utils.experiment import Experiment
except Exception as e:
    raise unittest.SkipTest(f'experiment modules not importable: {e}')

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs/experiments/sum_ablation.yaml')


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.exp = Experiment(CFG)

    def test_config_models_and_spaces_load(self):
        for m in self.exp.cfg['models']:
            self.assertIn(m, hps.MODELS)
            self.assertIn('lr', hps.MODELS[m]['space'])

    def test_teacher_dir_matches_train_baseline_naming(self):
        self.assertEqual(self.exp.teacher_dir('MUTAG'), os.path.join('results', 'MUTAG', hps.TEACHER_CFG))

    def test_units_and_dependencies(self):
        units = {u.key: u for u in self.exp.units(['MUTAG'])}
        n_models, n_folds = len(self.exp.cfg['models']), len(self.exp.folds)
        self.assertEqual(len(units), n_folds + 1 + n_models * (len(self.exp.tune_folds) + n_folds) + 1)
        self.assertEqual(units['MUTAG/hps/graph_sum/fold0'].deps, ('MUTAG/teacher/fold0',))
        self.assertEqual(set(units['MUTAG/final/graph_sum/fold7'].deps),
                         {'MUTAG/hps/graph_sum/fold0', 'MUTAG/teacher/fold7'})
        for u in units.values():
            for d in u.deps:
                self.assertIn(d, units)

    def test_per_fold_selection_uses_each_folds_study(self):
        cfg = yaml.safe_load(open(CFG))
        cfg['hps']['tune_folds'] = cfg['folds']
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(cfg, f)
        try:
            units = {u.key: u for u in Experiment(f.name).units(['MUTAG'])}
            self.assertIn('MUTAG/hps/classic/fold3', units['MUTAG/final/classic/fold3'].deps)
        finally:
            os.unlink(f.name)


class TestProgress(unittest.TestCase):

    def test_local_record_and_latest(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(progress, 'REPO', d), \
                mock.patch.dict(os.environ, {'NOTION_TOKEN': ''}):
            progress.report('exp', 'MUTAG', 'final', 'running', 'classic', 3)
            progress.report('exp', 'MUTAG', 'final', 'done', 'classic', 3, metrics={'test_acc': 0.9, 'x': None})
            last = progress.latest()
            rec = last['exp/MUTAG/final/classic/fold3']
            self.assertEqual((rec['status'], rec['metrics']), ('done', {'test_acc': 0.9}))

    def test_unpacked_teacher_counts_as_done(self):
        """A teacher bundle ships best.pt without last.pt: done only if its fold is in total_results.csv."""
        from utils.experiment import Unit
        self.exp = Experiment(CFG)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(Experiment, 'teacher_dir', lambda self, ds: d):
            os.makedirs(os.path.join(d, '3'))
            open(os.path.join(d, '3', 'best.pt'), 'w').close()
            u = Unit('teacher', 'MUTAG', fold=3)
            self.assertFalse(self.exp.is_done(u))               # interrupted run
            with open(os.path.join(d, 'total_results.csv'), 'w') as f:
                f.write(',seed,val_acc,test_acc\n0,3,0.9,0.9\n')
            self.assertTrue(self.exp.is_done(u))

    def test_pack_unpack_roundtrip(self):
        """pack() drops data.pkl and trial checkpoints; unpack() rebuilds data.pkl from the teacher."""
        import pickle
        import shutil
        from utils.splits import load_split
        self.exp = Experiment(CFG)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                os.makedirs('teacher/0')
                f0 = load_split('MUTAG', 0)
                pickle.dump({f'{s}_indices': f0[s] for s in ('train', 'val', 'test')}, open('teacher/0/data.pkl', 'wb'))
                with mock.patch.object(Experiment, 'teacher_dir', lambda self, ds: 'teacher'):
                    run = os.path.join(self.exp.final_dir('MUTAG', 'classic', 0), 'run', 'x', '0')
                    os.makedirs(run)
                    for f in ('best.pt', 'last.pt', 'data.pkl'):
                        open(os.path.join(run, f), 'w').close()
                    json.dump({'dataset': 'MUTAG', 'fold': 0}, open(os.path.join(run, '..', '..', '..', 'final.json'), 'w'))
                    trial = os.path.join(self.exp.cfg['hps']['root'], 'MUTAG', 'classic', 'fold0', 'runs', 't', 'best.pt')
                    os.makedirs(os.path.dirname(trial))
                    open(trial, 'w').close()
                    open(os.path.join(os.path.dirname(trial), '..', '..', 'best.json'), 'w').close()
                    self.exp.pack(['MUTAG'], 'out.tar.gz')
                    import tarfile
                    names = tarfile.open('out.tar.gz').getnames()
                    self.assertFalse([n for n in names if n.endswith(('data.pkl', 'last.pt')) or '/runs/' in n])
                    self.assertIn(os.path.join(run, 'best.pt'), names)
                    shutil.rmtree(self.exp.cfg['final']['root'])
                    shutil.rmtree(self.exp.cfg['hps']['root'])
                    self.exp.unpack('out.tar.gz')
                    self.assertEqual(pickle.load(open(os.path.join(run, 'data.pkl'), 'rb'))['test_indices'], f0['test'])
            finally:
                os.chdir(cwd)

    def test_notion_properties_follow_schema(self):
        rec = {'unit': 'e/D/final/m/fold1', 'experiment': 'e', 'dataset': 'D', 'stage': 'final', 'status': 'done',
               'model': 'm', 'fold': 1, 'machine': 'host', 'progress': None, 'note': None, 'git_commit': 'abc',
               'started': '2026-10-06T10:00:00+02:00', 'time': '2026-10-06T11:00:00+02:00',
               'metrics': {'test_acc': 0.91234, 'val_balanced_acc': 0.8}}
        props = progress.notion_properties(rec)
        self.assertLessEqual(set(props), set(progress.NOTION_SCHEMA))
        for name, value in props.items():
            self.assertIn(progress.NOTION_SCHEMA[name], value)
        self.assertEqual(props['Test acc'], {'number': 0.9123})


if __name__ == '__main__':
    unittest.main()
