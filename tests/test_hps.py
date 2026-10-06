"""utils/hps.py: search spaces, command lines, study guards (no training).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_hps.py -v
"""
import json
import os
import sys
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
