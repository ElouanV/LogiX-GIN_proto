"""--select balanced_acc (train_proto.py, train_logic.py) and qat_proto's select_metric: the
checkpoint is kept on the validation balanced accuracy, the objective of the search.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_select.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import qat_proto
    import train_logic
    import train_proto
    from models_proto.model_proto import get_model
    from utils.training import val_score
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'training modules not importable: {e}')


class Majority(torch.nn.Module):
    """Predicts class 0 for every graph."""
    num_features = 3

    def forward(self, x, edge_index, batch, *args, **kwargs):
        n = int(batch.max()) + 1
        return torch.tensor([[1.0, 0.0]]).repeat(n, 1)


def imbalanced(n0=9, n1=1):
    g = [Data(x=torch.ones(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([c]))
         for c in [0] * n0 + [1] * n1]
    return DataLoader(g, batch_size=4)


class TestValScore(unittest.TestCase):
    def test_scores(self):
        loader = imbalanced()
        self.assertAlmostEqual(val_score(Majority(), loader, 'cpu'), 0.9)
        self.assertAlmostEqual(val_score(Majority(), loader, 'cpu', 'acc'), 0.9)
        self.assertAlmostEqual(val_score(Majority(), loader, 'cpu', 'balanced_acc'), 0.5)
        with self.assertRaises(ValueError):
            val_score(Majority(), loader, 'cpu', 'f1')

    def test_cli(self):
        for m in (train_proto, train_logic):
            _, _, args = m.parse_cli(['--dataset', 'MUTAG', '--select', 'balanced_acc'])
            self.assertEqual(args['select'], 'balanced_acc')
            for argv in ([], ['--select', 'acc']):          # default: results paths unchanged
                _, _, args = m.parse_cli(['--dataset', 'MUTAG', *argv])
                self.assertNotIn('select', args)


class TestQATSelect(unittest.TestCase):
    def test_diagnostics_balanced_acc(self):
        torch.manual_seed(0)
        m = get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=4)
        d = qat_proto.diagnostics(m, imbalanced(), 'cpu')
        self.assertIn('val_balanced_acc_rules', d)
        self.assertTrue(0 <= d['val_balanced_acc_rules'] <= 1)


if __name__ == '__main__':
    unittest.main()
