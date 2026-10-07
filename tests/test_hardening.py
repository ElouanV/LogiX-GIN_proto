"""Gradual hardening schedule and options of train_proto.py (--harden_to).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_hardening.py -v

The model side (set_sharpness) is tested in tests/test_interp_metrics.py.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from train_proto import parse_cli, sharpness
except Exception as e:
    raise unittest.SkipTest(f'train_proto not importable: {e}')


class TestHardening(unittest.TestCase):

    def test_off_keeps_the_results_path_arguments(self):
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--proto_mask'])
        self.assertFalse(any(k.startswith('harden') for k in args))
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--harden_end_frac', '0.5'])   # ignored without --harden_to
        self.assertFalse(any(k.startswith('harden') for k in args))
        self.assertEqual(sharpness(args, 10 ** 6), 1.0)

    def test_geometric_schedule(self):
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--epochs', '3000', '--warmup_epochs', '999',
                                '--harden_to', '100'])
        self.assertEqual(set(k for k in args if k.startswith('harden')), {'harden_to'})
        self.assertEqual(sharpness(args, 0), 1.0)
        self.assertEqual(sharpness(args, 1000), 1.0)                  # starts after the warmup
        self.assertAlmostEqual(sharpness(args, 1850), 10.0, places=6)   # halfway (1000 -> 2700)
        self.assertEqual(sharpness(args, 2700), 100.0)
        self.assertEqual(sharpness(args, 2999), 100.0)
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--epochs', '100', '--harden_to', '4',
                                '--harden_start_frac', '0.5', '--harden_end_frac', '0.7'])
        self.assertEqual(sharpness(args, 50), 1.0)
        self.assertAlmostEqual(sharpness(args, 60), 2.0, places=6)

    def test_rejects_softening(self):
        with self.assertRaises(SystemExit):
            parse_cli(['--dataset', 'MUTAG', '--harden_to', '0.5'])


if __name__ == '__main__':
    unittest.main()
