"""utils/hps.py: search spaces, command lines, study guards (no training).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_hps.py -v
"""
import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from utils import hps
except Exception as e:
    raise unittest.SkipTest(f'optuna / hps not importable: {e}')


def some_params(model):
    t = optuna.trial.FixedTrial({k: (s['choices'][0] if s['type'] == 'categorical' else s['low'])
                                 for k, s in hps.MODELS[model]['space'].items()})
    return hps.suggest(t, hps.MODELS[model]['space'])


class TestHPS(unittest.TestCase):

    def test_spaces_are_json_and_pairs_match(self):
        json.dumps({m: hps.study_attrs('MUTAG', m, 0) for m in hps.MODELS})
        for base, other in (('classic', 'classic_nosum'), ('node_mask_push', 'node_mask_push_sum'),
                            ('graph', 'graph_sum')):
            self.assertEqual(hps.MODELS[base]['space'], hps.MODELS[other]['space'])

    def test_argv_parses_into_the_model(self):
        expect = {'classic': None, 'classic_nosum': 'mean,max', 'node_mask_push': None,
                  'node_mask_push_sum': 'mean,max,sum', 'graph': None, 'graph_sum': 'mean,max,sum'}
        for model, pool in expect.items():
            p = some_params(model)
            ds, bp, args = hps.script_module(model).parse_cli(hps.to_argv('MUTAG', model, p, 3))
            self.assertEqual((ds, args['seed'], args.get('pool_ops')), ('MUTAG', 3, pool), model)
            self.assertEqual(args['warmup_epochs'], int(round(p['warmup_frac'] * p['epochs'])))
            self.assertEqual(bp, hps.teacher_path('MUTAG'))
            if model.startswith('node'):
                self.assertTrue(args['proto_mask'])
                self.assertEqual(args['proto_level'], 'node')
            self.assertEqual((args['head_init'], args['logit_loss']), ('center', True), model)   # head saturation fix

    def test_qat_parameters_go_to_run_qat_not_to_the_training_script(self):
        params = some_params('node_mask_push_qat')
        self.assertTrue(any(k.startswith(hps.QAT_PREFIX) for k in params))
        argv = hps.to_argv('MUTAG', 'node_mask_push_qat', params, 0)
        self.assertFalse(any(a.startswith('--qat_') for a in argv))
        q = hps.qat_args('node_mask_push_qat', params)
        self.assertEqual(q['rate_to'], 1.0)
        self.assertEqual(q['lr'], params['qat_lr'])
        import inspect
        from qat_proto import run_qat
        self.assertLessEqual(set(q), set(inspect.signature(run_qat).parameters))
        self.assertEqual(hps.study_attrs('MUTAG', 'node_mask_push_qat', 0)['objective'], hps.OBJECTIVE_QAT)
        self.assertEqual(hps.finish_run('node_mask_push', '/some/run', params, 0), '/some/run')

    def test_study_refuses_another_space(self):
        with tempfile.TemporaryDirectory() as root:
            hps.open_study('MUTAG', 'graph', 0, root)
            saved = hps.MODELS['graph']['space']
            hps.MODELS['graph']['space'] = {**saved, 'lr': hps._f(1e-5, 1e-1, log=True)}
            try:
                with self.assertRaises(RuntimeError):
                    hps.open_study('MUTAG', 'graph', 0, root)
            finally:
                hps.MODELS['graph']['space'] = saved
            hps.open_study('MUTAG', 'graph', 0, root)               # unchanged space: joins



class FakeTrial:
    def __init__(self):
        self.reports, self.user_attrs = [], {}

    def report(self, v, step):
        self.reports.append((step, v))

    def should_prune(self):
        return False

    def set_user_attr(self, k, v):
        self.user_attrs[k] = v


class Scores(torch.nn.Module):
    """Class outputs scripted per call: 'const' -> class 0 for every graph, else the label."""
    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self.calls = 0

    def forward(self, x, edge_index, batch, **kw):
        kind = self.script[min(self.calls // 2, len(self.script) - 1)]      # evaluate + predict per report
        self.calls += 1
        n = int(batch.max()) + 1
        if kind == 'const':
            return torch.tensor([[1.0, 0.0]]).repeat(n, 1)
        return torch.stack([torch.arange(n) % 2 == 0, torch.arange(n) % 2 == 1], 1).float()


class TestCollapsePrune(unittest.TestCase):

    def setUp(self):
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        g = [Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([i % 2]))
             for i in range(6)]
        self.loader = DataLoader(g, batch_size=6)

    def _run(self, script, epochs=(10, 15, 20, 25)):
        trial, model = FakeTrial(), Scores(script)
        cb = hps.PruneCallback(trial, warmup=5, device='cpu', every=5)
        for e in range(epochs[-1] + 1):
            cb(e, model, self.loader)
        return trial

    def test_two_consecutive_one_class_reports_prune(self):
        with self.assertRaises(optuna.TrialPruned):
            self._run(['ok', 'const', 'const', 'ok'])

    def test_one_isolated_one_class_report_does_not(self):
        trial = self._run(['const', 'ok', 'const', 'ok'])
        self.assertEqual(len(trial.reports), 4)
        self.assertNotIn('collapsed', trial.user_attrs)


if __name__ == '__main__':
    unittest.main()
