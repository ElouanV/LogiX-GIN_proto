import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.runs import RunDir, config_name, student_dir, teacher_dir  # noqa: E402


def make_run(path, args):
    os.makedirs(path)
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(args, f)
    return path


class TestRunDir(unittest.TestCase):
    def test_student_under_any_root(self):
        """results_proto/, hps trials and k-fold final runs (results_final/<exp>/<ds>/...)."""
        r = RunDir('results_proto/MUTAG/node-x/base/2')
        self.assertEqual((r.dataset, r.config, r.teacher_config, r.seed), ('MUTAG', 'node-x', 'base', '2'))
        with tempfile.TemporaryDirectory() as t:
            d = make_run(os.path.join(t, 'results_final', 'exp', 'MUTAG', 'nmp', 'fold3', 'run', 'MUTAG', 'node-x',
                                      'base|split=kfold', '3'), {'proto_level': 'node'})
            r = RunDir(d)
            self.assertEqual((r.stage, r.dataset, r.config, r.teacher_config, r.seed),
                             ('proto', 'MUTAG', 'node-x', 'base|split=kfold', '3'))
            self.assertEqual(r.teacher_path, os.path.join('results', 'MUTAG', 'base|split=kfold', '3'))
            with self.assertRaises(ValueError):
                RunDir(os.path.dirname(d)).stage

    def test_finetunes_are_stripped(self):
        r = RunDir('results_logic/AIDS/cfg/base/0/qat/rate_to=1.0/sparse/layerwise=True')
        self.assertEqual(r.finetunes, [('qat', 'rate_to=1.0'), ('sparse', 'layerwise=True')])
        self.assertEqual((r.dataset, r.seed, r.source), ('AIDS', '0', 'results_logic/AIDS/cfg/base/0'))

    def test_teacher_and_stages(self):
        with tempfile.TemporaryDirectory() as t:
            cwd = os.getcwd()
            os.chdir(t)
            try:
                tp = make_run('results/PROTEINS/batch_size=32/1', {'hidden_dim': 32, 'dropout': 0.1})
                lp = make_run('results_logic/PROTEINS/c/batch_size=32/1', {'conv_reg': 0.001})
                self.assertEqual((RunDir(tp).stage, RunDir(tp).dataset, RunDir(tp).seed), ('teacher', 'PROTEINS', '1'))
                self.assertEqual(RunDir(lp).stage, 'logic')
                self.assertEqual(RunDir(lp, dataset='X').dataset, 'X')
            finally:
                os.chdir(cwd)

    def test_directory_names(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(config_name({'b': 1, 'a': 'x'}), 'a=x|b=1')
            self.assertEqual(teacher_dir('D', {'lr': 0.1}, seed=0, root=t), f'{t}/D/lr=0.1/0')
            long = {'proto_level': 'node', **{f'k{i:03d}': 'v' * 8 for i in range(30)}}
            p = student_dir('D', long, {'lr': 0.1}, seed=2, root=t, hash_long=True)
            name = p.split('/')[-3]
            self.assertTrue(name.startswith('node-') and len(name) == 17)
            self.assertEqual(open(f'{t}/D/{name}/config.txt').read().strip(), config_name(long))


if __name__ == '__main__':
    unittest.main()
