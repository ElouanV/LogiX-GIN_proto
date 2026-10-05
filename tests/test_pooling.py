"""Readout pooling options: GINTELLPool (models_proto/gintell.py) and sum pooling in the
prototype models (models_proto/model_proto.py, bounded_mask / phi_sum).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_pooling.py -v
"""
import glob
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.model import GINTELL
    from models_proto.gintell import GINTELLPool
    from models_proto.model_proto import get_model
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'models not importable: {e}')

F, C, L, H, K = 5, 2, 3, 8, 4


def toy_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 12
    x = (torch.rand(n, F, generator=g) > 0.5).float()
    src = torch.randint(0, n, (30,), generator=g)
    dst = torch.randint(0, n, (30,), generator=g)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    batch = torch.tensor([0] * 5 + [1] * 7)
    return x, edge_index, batch


class TestGINTELLPool(unittest.TestCase):

    def test_upstream_ops_compute_gintell(self):
        torch.manual_seed(0)
        ref = GINTELL(F, C, num_layers=L, hidden_dim=H).eval()
        model = GINTELLPool(F, C, num_layers=L, hidden_dim=H).eval()
        model.load_state_dict(ref.state_dict())
        x, ei, b = toy_batch()
        torch.testing.assert_close(model(x, ei, b), ref(x, ei, b))

    def test_nosum_head_and_teacher_columns(self):
        model = GINTELLPool(F, C, num_layers=L, hidden_dim=H, pool_ops=('max', 'mean'))
        self.assertEqual(model.pool_ops, ('mean', 'max'))                 # upstream order
        self.assertEqual(model.fc.in_features, 2 * 2 * L * H)
        n = L * H
        pooled = torch.arange(3 * n).float()[None]                       # mean | max | sum
        torch.testing.assert_close(model.select_pooled(pooled), pooled[:, :2 * n])
        x, ei, b = toy_batch()
        self.assertEqual(model(x, ei, b).shape, (2, C))
        with self.assertRaises(ValueError):
            GINTELLPool(F, C, pool_ops=('median',))


class TestProtoSumPooling(unittest.TestCase):

    def test_default_readout_is_unchanged(self):
        for level in ('node', 'graph', 'both'):
            model = get_model(level, F, C, num_layers=L, hidden_dim=H, num_prototypes=K)
            R = model.readout_dim
            self.assertEqual(model.fc.in_features, 2 * R, level)
            s = torch.rand(3, R)
            torch.testing.assert_close(model.head_input(s), torch.hstack([s, 1 - s]))
            self.assertIsNone(getattr(model, 'phi_sum', None))

    def test_pickles_without_bounded_mask(self):
        model = get_model('node', F, C, num_layers=L, hidden_dim=H, num_prototypes=K)
        del model._buffers['_bounded']
        s = torch.rand(3, model.readout_dim)
        torch.testing.assert_close(model.head_input(s), torch.hstack([s, 1 - s]))
        self.assertEqual(len(model.head_columns()), 2 * model.readout_dim)

    def test_node_sum_is_not_negated(self):
        model = get_model('node', F, C, num_layers=L, hidden_dim=H, num_prototypes=K,
                          pool_ops=('mean', 'max', 'sum'))
        self.assertEqual(model.readout_dim, 3 * K)
        self.assertEqual(model.fc.in_features, 3 * K + 2 * K)
        cols = model.head_columns()
        self.assertEqual([c for c, pos in cols if not pos], list(range(2 * K)))
        x, ei, b = toy_batch()
        self.assertEqual(model(x, ei, b).shape, (2, C))

    def test_graph_sum_is_thresholded_into_unit_interval(self):
        model = get_model('graph', F, C, num_layers=L, hidden_dim=H, num_prototypes=K,
                          pool_ops=('mean', 'max', 'sum'))
        self.assertIsNotNone(model.phi_sum)
        self.assertEqual(model.fc.in_features, 2 * K)
        x, ei, b = toy_batch()
        xs = []
        h = x
        for conv in model.convs:
            h = conv(torch.hstack([h, 1 - h]), ei)
            xs.append(h)
        z = model.prototype_inputs(xs, b)[0]
        self.assertEqual(z.shape, (2, 3 * L * H))
        self.assertTrue(bool(((z >= 0) & (z <= 1)).all()))
        zs = model.graph_input(torch.hstack(xs), b, symbolic=True)[:, 2 * L * H:]
        self.assertTrue(bool(((zs == 0) | (zs == 1)).all()))
        model.set_hard(True)
        out = model(x, ei, b)
        out.sum().backward()
        self.assertIsNotNone(model.phi_sum.b.grad)                       # straight-through

    def test_existing_checkpoint_accuracy(self):
        """An NMP checkpoint saved before bounded_mask still scores its recorded test accuracy."""
        runs = sorted(glob.glob('results_proto/Mutagenicity/node-facbe6d14f72/*/total_results.csv'))
        if not runs:
            self.skipTest('no NMP run')
        import pandas as pd
        import pickle
        from torch_geometric.loader import DataLoader
        from train_proto import test_epoch
        root = os.path.dirname(runs[0])
        rec = pd.read_csv(runs[0]).set_index('seed')['test_acc']
        seed = int(rec.index[0])
        model = torch.load(os.path.join(root, str(seed), 'best.pt'), map_location='cpu', weights_only=False)
        self.assertIsNone(getattr(model, '_bounded', None))
        data = pickle.load(open(os.path.join(root, str(seed), 'data.pkl'), 'rb'))
        acc = test_epoch(model, DataLoader(data['test_dataset'], batch_size=64), torch.device('cpu'))
        self.assertAlmostEqual(acc, rec.loc[seed], places=6)


if __name__ == '__main__':
    unittest.main()
