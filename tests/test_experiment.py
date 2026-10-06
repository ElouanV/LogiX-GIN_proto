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
