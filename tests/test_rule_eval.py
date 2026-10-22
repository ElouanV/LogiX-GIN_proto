"""rule_eval.rule_stats: the shortest rule of a unit that always fires (threshold -b <= 0) is
the empty set, 0 literals.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_rule_eval.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.loader import DataLoader
    from models.model import GINTELL
    from rule_eval import rule_stats
    from tests.test_val_loss import graphs
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'training modules not importable: {e}')


class TestShortest(unittest.TestCase):
    def test_always_on_unit_needs_no_literal(self):
        torch.manual_seed(0)
        m = GINTELL(3, 2, num_layers=2, hidden_dim=8)
        with torch.no_grad():
            m.fc.b.fill_(1.0)                            # S = -b < 0: both class units always fire
        head = rule_stats(m, DataLoader(graphs(8, 0), batch_size=4), 'cpu')[-1]
        self.assertEqual(head['constant_units'], 2)
        self.assertEqual(head['shortest_per_node_median'], 0.0)


if __name__ == '__main__':
    unittest.main()
