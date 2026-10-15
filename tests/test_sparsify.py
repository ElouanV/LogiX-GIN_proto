"""Layer-by-layer Hoyer fine-tune of sparsify_proto.py and its unit statistics.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_sparsify.py -v
"""
import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models_proto.tell import LogicalLayer
    from models_proto.model_proto import get_model
    from sparsify_proto import layerwise_sparsify, layerwise_steps, notebook_hoyer_loss, unit_groups, used_units
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'sparsify_proto not importable: {e}')


class Tiny(nn.Module):
    """Just the attributes sparsify_proto touches: convs[i].nn[0] and fc."""
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Module() for _ in range(2)])
        for c in self.convs:
            c.nn = nn.Sequential(LogicalLayer(8, 5))
        self.fc = LogicalLayer(10, 2)


class TestUnitGroups(unittest.TestCase):

    def test_base_head_groups_conv_units_over_pooling_and_polarity(self):
        m = Tiny()                                       # head reads 5 inputs = [s, 1-s]
        m.convs[0].nn[0] = LogicalLayer(8, 5)
        groups = unit_groups(m)
        n = m.fc.in_features // 2
        L, h = 2, 5
        self.assertTrue(torch.equal(groups[-1][1], (torch.arange(2 * n) % n) % (L * h)))

    def test_prototype_head_groups_by_prototype(self):
        K, Kg = 3, 2
        for level, expect in (('node', [0, 1, 2, 0, 1, 2]), ('graph', [0, 1, 2]),
                              ('both', [0, 1, 2, 0, 1, 2, 3, 4])):
            m = get_model(level, num_features=4, num_classes=2, num_layers=2, hidden_dim=5,
                          num_prototypes=K, **({'num_graph_prototypes': Kg} if level == 'both' else {}))
            ll, g = unit_groups(m)[-1]
            self.assertIs(ll, m.fc)
            self.assertEqual(g.tolist(), expect + expect)          # [s, 1-s]
            self.assertEqual(len(g), m.fc.in_features)
            self.assertEqual(len(used_units(m)), 2)                # L1 and the head


class TestLayerwise(unittest.TestCase):
    """--layerwise: the notebook's layer-by-layer procedure (nbs/LayerWiseRules.ipynb)."""

    def test_notebook_hoyer_normalises_by_all_entries(self):
        w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        # one-hot rows: |w|_1/|w|_2 = 1, normalised by sqrt(8) (all entries), not sqrt(4)
        expected = 1 - (8 ** 0.5 - 1) / (8 ** 0.5 - 1)
        self.assertAlmostEqual(notebook_hoyer_loss(w).item(), expected, places=5)

    def test_order_head_then_last_to_first_conv(self):
        model = get_model('node', num_features=3, num_classes=2, num_layers=3, hidden_dim=4, num_prototypes=2)
        self.assertEqual([s[0] for s in layerwise_steps(model)], ['head', 'L2', 'L1', 'L0'])

    def test_lr_and_patience_overrides(self):
        model = get_model('node', num_features=3, num_classes=2, num_layers=3, hidden_dim=4, num_prototypes=2)
        default = layerwise_steps(model)
        self.assertEqual(default[0][2:], (0.01, 100, 0.1, 0.1))
        self.assertTrue(all(s[2:] == (0.005, 50, 0.1, 0.01) for s in default[1:]))
        steps = {s[0]: s for s in layerwise_steps(model, head_lr=1e-3, conv_lr=[1e-4, 2e-4, 5e-3], conv_patience=200)}
        self.assertEqual(steps['head'][2:4], (1e-3, 100))
        self.assertEqual([steps[f'L{l}'][2] for l in range(3)], [1e-4, 2e-4, 5e-3])
        self.assertTrue(all(steps[f'L{l}'][3] == 200 for l in range(3)))
        self.assertTrue(all(s[2] == 1e-4 for s in layerwise_steps(model, conv_lr=[1e-4])[1:]))
        with self.assertRaises(ValueError):
            layerwise_steps(model, conv_lr=[1e-4, 1e-4])

    def test_each_step_trains_only_its_layer(self):
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        torch.manual_seed(0)
        graphs = [Data(x=(torch.rand(5, 3) > 0.5).float(), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
                       y=torch.tensor([i % 2])) for i in range(16)]
        loader = DataLoader(graphs, batch_size=8)
        model = get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=4, num_prototypes=2,
                          proto_mask=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        out, hist = layerwise_sparsify(model, loader, loader, 'cpu', 2, max_epochs=2)
        self.assertEqual([h['step'] for h in hist], ['head', 'L1', 'L0'])
        after = out.state_dict()
        trained = ('fc.weight_sigma', 'fc.weight_exp') + tuple(
            f'convs.{l}.nn.0.weight_{p}' for l in range(2) for p in ('sigma', 'exp'))
        for k, v in before.items():
            if k not in trained:
                self.assertTrue(torch.equal(v, after[k]), k)       # prototypes, masks, thresholds fixed
        for h in hist:
            self.assertLessEqual(h['epochs'], 2)


if __name__ == '__main__':
    unittest.main()
