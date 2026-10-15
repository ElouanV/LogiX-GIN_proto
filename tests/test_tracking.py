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


class TmpStore(unittest.TestCase):
    """A throwaway sqlite store per test."""

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


@unittest.skipIf(tracking.mlflow is None, 'mlflow not installed')
class TestTracking(TmpStore):

    def test_log_to_run_adds_metrics_to_the_latest_matching_run(self):
        for i in range(2):
            with tracking.run('proto/X', f'r{i}', tags={'kind': 'final', 'seed': 3}):
                tracking.log_metrics({'test_acc': i})
        with tracking.run('proto/X', 'other', tags={'kind': 'final', 'seed': 4}):
            pass
        run_id = tracking.log_to_run(['proto/X', 'logic/X'], {'kind': 'final', 'seed': 3}, {'interp/f': 0.5, 'none': None})
        r = self.client().get_run(run_id)
        self.assertEqual(r.info.run_name, 'r1')
        self.assertEqual(r.data.metrics['interp/f'], 0.5)
        self.assertNotIn('none', r.data.metrics)
        self.assertIsNone(tracking.log_to_run(['proto/X'], {'seed': 9}, {'a': 1}))
        self.assertIsNone(tracking.log_to_run(['missing/X'], {'seed': 3}, {'a': 1}))

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


@unittest.skipIf(tracking.mlflow is None, 'mlflow not installed')
class TestNestedPipelines(TmpStore):
    """One parent run per trained model, one nested run per stage (teacher, distillation,
    finetune/<method>), in experiment pipeline/<dataset>."""

    def _runs(self, dataset='D'):
        c = self.client()
        exp = c.get_experiment_by_name(f'pipeline/{dataset}')
        return {r.info.run_name: r for r in c.search_runs([exp.experiment_id])}

    def test_stages_nest_under_the_open_pipeline(self):
        with tracking.pipeline('D', 'trial000', tags={'kind': 'hps'}):
            with tracking.run('proto/D', 'proto/node/seed0', stage='distillation', dataset='D'):
                tracking.log_metrics({'train_loss': 1.0, 'val_loss': 2.0}, step=0)
            with tracking.run('qat/D', 'qat', stage='finetune/qat', dataset='D'):
                tracking.log_metrics({'val_loss': 3.0}, step=0)
        runs = self._runs()
        parent = runs['trial000']
        self.assertEqual(parent.data.tags['kind'], 'pipeline')
        self.assertEqual(parent.data.tags['pipeline_kind'], 'hps')
        for name, stage in (('distillation', 'distillation'), ('finetune/qat', 'finetune/qat')):
            self.assertEqual(runs[name].data.tags['mlflow.parentRunId'], parent.info.run_id)
            self.assertEqual(runs[name].data.tags['stage'], stage)
        self.assertEqual(runs['distillation'].data.tags['stage_experiment'], 'proto/D')
        self.assertEqual(runs['distillation'].data.metrics['val_loss'], 2.0)

    def test_stage_alone_opens_its_own_pipeline(self):
        with tracking.run('proto/D', 'proto/node/seed3', stage='distillation', dataset='D'):
            pass
        runs = self._runs()
        self.assertEqual(runs['distillation'].data.tags['mlflow.parentRunId'], runs['proto/node/seed3'].info.run_id)
        self.assertFalse(tracking.in_pipeline())

    def test_later_fine_tune_nests_under_the_recorded_parent(self):
        run_dir = os.path.join(self.tmp.name, 'run')
        os.makedirs(run_dir)
        with tracking.run('proto/D', 'proto/node/seed0', stage='distillation', dataset='D'):
            tracking.write_pipeline(run_dir)
        rec = tracking.read_pipeline(run_dir)
        self.assertEqual(tracking.fine_tune_parent(run_dir), rec['parent_run_id'])
        with tracking.run('sparsify/D', 'hoyer', stage='finetune/hoyer', dataset='D',
                          parent_run_id=tracking.fine_tune_parent(run_dir)):
            pass
        runs = self._runs()
        self.assertEqual(runs['finetune/hoyer'].data.tags['mlflow.parentRunId'], rec['parent_run_id'])
        self.assertEqual(len([r for r in runs.values() if r.data.tags.get('kind') == 'pipeline']), 1)

    def test_teacher_reference_once_per_pipeline_with_the_teacher_metrics(self):
        teacher_dir = os.path.join(self.tmp.name, 'results', 'D', 'cfg', '0')
        os.makedirs(teacher_dir)
        with tracking.run('baseline/D', 'teacher/seed0', stage='teacher', dataset='D'):
            tracking.log_metrics({'final/test_acc': 0.8})
            tracking.write_pipeline(teacher_dir)
        with tracking.pipeline('D', 'trial001'):
            tracking.teacher_reference(teacher_dir, 'D', 0)
            tracking.teacher_reference(teacher_dir, 'D', 0)
        c = self.client()
        exp = c.get_experiment_by_name('pipeline/D')
        refs = c.search_runs([exp.experiment_id], "tags.kind = 'reference'")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].data.metrics['final/test_acc'], 0.8)
        self.assertEqual(refs[0].data.tags['teacher_run_id'], tracking.read_pipeline(teacher_dir)['run_id'])

    def test_failed_stage_marks_both_runs_failed_and_closes_the_pipeline(self):
        with self.assertRaises(RuntimeError):
            with tracking.pipeline('D', 'trial002'):
                with tracking.run('proto/D', 'x', stage='distillation', dataset='D'):
                    raise RuntimeError('boom')
        runs = self._runs()
        self.assertEqual(runs['distillation'].info.status, 'FAILED')
        self.assertEqual(runs['trial002'].info.status, 'FAILED')
        self.assertIsNone(tracking.mlflow.active_run())
