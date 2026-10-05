"""utils/tracking.py and utils/evaluation.py.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_tracking.py -v

Each test logs into a throwaway sqlite store in a temporary directory, never into the
repository's mlflow.db / mlartifacts.
"""
import os
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import tracking            # noqa: E402
from utils.evaluation import evaluate  # noqa: E402


class TinyGraphModel(torch.nn.Module):
    """Class-1 output = mean of feature 0 over the graph (sigmoid-like range)."""
    num_features = 2

    def forward(self, x, edge_index, batch, **kw):
        n = int(batch.max()) + 1
        s = torch.zeros(n).index_add_(0, batch, x[:, 0]) / torch.bincount(batch, minlength=n)
        return torch.stack([1 - s, s], 1)


def graphs(labels_scores):
    return [Data(x=torch.tensor([[s, 0.0], [s, 0.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]),
                 y=torch.tensor([y])) for y, s in labels_scores]


@unittest.skipIf(tracking.mlflow is None, 'mlflow not installed')
class TestTracking(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {k: os.environ.get(k) for k in ('MLFLOW_TRACKING_URI', 'LOGIX_MLFLOW_ARTIFACTS', 'LOGIX_MLFLOW',
                                                   'LOGIX_CO2')}
        os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{self.tmp.name}/t.db'
        os.environ['LOGIX_MLFLOW_ARTIFACTS'] = os.path.join(self.tmp.name, 'art')
        os.environ.pop('LOGIX_MLFLOW', None)
        os.environ.setdefault('LOGIX_CO2', '0')      # only test_co2_is_logged measures
        self.client = lambda: tracking.mlflow.MlflowClient(os.environ['MLFLOW_TRACKING_URI'])

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_run_logs_params_metrics_tags_artifacts(self):
        p = os.path.join(self.tmp.name, 'a.json')
        open(p, 'w').write('{}')
        with tracking.run('test/exp', 'r1', params={'lr': 0.1}, tags={'kind': 'seed'}) as r:
            for step in range(3):
                tracking.log_metrics({'val_acc': 0.5 + step / 10, 'skip': None}, step=step)
            tracking.log_artifact(p)
            rid = r.info.run_id
        c = self.client()
        run = c.get_run(rid)
        self.assertEqual(run.info.status, 'FINISHED')
        self.assertEqual(run.data.params['lr'], '0.1')
        self.assertEqual(run.data.tags['kind'], 'seed')
        self.assertIn('git_commit', run.data.tags)
        self.assertEqual([m.value for m in c.get_metric_history(rid, 'val_acc')], [0.5, 0.6, 0.7])
        self.assertNotIn('skip', run.data.metrics)
        self.assertEqual([a.path for a in c.list_artifacts(rid)], ['a.json'])
        self.assertTrue(c.get_experiment_by_name('test/exp').artifact_location.startswith(
            'file://' + os.path.join(self.tmp.name, 'art')))

    def test_step_metrics_are_batched_then_flushed(self):
        os.environ['LOGIX_MLFLOW_FLUSH_STEPS'] = '2'
        try:
            with tracking.run('test/exp', 'batched') as r:
                rid = r.info.run_id
                tracking.log_metrics({'a': 1.0}, step=0)
                self.assertEqual(self.client().get_metric_history(rid, 'a'), [])     # buffered
                tracking.log_metrics({'a': 2.0}, step=1)                             # 2 steps: sent
                self.assertEqual(len(self.client().get_metric_history(rid, 'a')), 2)
                tracking.log_metrics({'a': 3.0}, step=2)
                tracking.log_metrics({'final': 9.0})                                  # flushes first
                self.assertEqual(len(self.client().get_metric_history(rid, 'a')), 3)
        finally:
            os.environ.pop('LOGIX_MLFLOW_FLUSH_STEPS')
        hist = self.client().get_metric_history(rid, 'a')
        self.assertEqual([(m.step, m.value) for m in hist], [(0, 1.0), (1, 2.0), (2, 3.0)])

    @unittest.skipIf(tracking.OfflineEmissionsTracker is None, 'codecarbon not installed')
    def test_co2_is_logged(self):
        os.environ['LOGIX_CO2'] = '1'
        with tracking.run('test/exp', 'co2') as r:
            rid = r.info.run_id
        m = self.client().get_run(rid).data.metrics
        for k in ('co2/emissions_kg', 'co2/energy_kwh', 'co2/gpu_kwh', 'co2/cpu_kwh', 'co2/duration_s'):
            self.assertIn(k, m)
        self.assertGreaterEqual(m['co2/emissions_kg'], 0)

    def test_exception_marks_run_failed_and_propagates(self):
        with self.assertRaises(ValueError):
            with tracking.run('test/exp', 'boom') as r:
                rid = r.info.run_id
                raise ValueError
        self.assertEqual(self.client().get_run(rid).info.status, 'FAILED')

    def test_register_model_versions(self):
        model = torch.nn.Linear(2, 2)
        for seed in range(2):
            with tracking.run('test/exp', f's{seed}'):
                tracking.log_model(model, 'm-test', tags={'seed': seed})
        versions = self.client().search_model_versions("name='m-test'")
        self.assertEqual(sorted(v.tags['seed'] for v in versions), ['0', '1'])

    def test_disabled_is_a_noop(self):
        os.environ['LOGIX_MLFLOW'] = '0'
        with tracking.run('test/exp', 'off') as r:
            self.assertIsNone(r)
            tracking.log_metrics({'a': 1.0})
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, 't.db')))

    def test_logging_outside_a_run_is_ignored(self):
        tracking.log_metrics({'a': 1.0})
        tracking.log_params({'a': 1})


class TestEvaluation(unittest.TestCase):

    def test_metrics_on_imbalanced_labels(self):
        # 6 negatives, 2 positives; one positive is missed
        data = graphs([(0, .1), (0, .2), (0, .3), (0, .1), (0, .2), (0, .3), (1, .9), (1, .4)])
        m = evaluate(TinyGraphModel(), DataLoader(data, batch_size=3), torch.device('cpu'), prefix='t_')
        self.assertAlmostEqual(m['t_acc'], 7 / 8)
        self.assertAlmostEqual(m['t_balanced_acc'], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(m['t_f1'], 2 / 3)             # minority (positive) class: P=1, R=.5
        self.assertAlmostEqual(m['t_auc'], 1.0)              # ranking is perfect even if the cut is not
        self.assertEqual(m['t_n'], 8)


if __name__ == '__main__':
    unittest.main()
