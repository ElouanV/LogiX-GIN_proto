"""utils/agreement.py: per-prototype agreement distributions (no training).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_agreement.py -v
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_proto.proto import PrototypeLayer
from models_proto.model_proto import get_model
from utils.agreement import layer_agreement, log_agreement, separation


def toy_graphs(n=24, f=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = 4 + i % 3
        ei = torch.stack([torch.arange(k - 1), torch.arange(1, k)])
        out.append(Data(x=(torch.rand(k, f, generator=g) > 0.5).float(), edge_index=torch.cat([ei, ei.flip(0)], 1),
                        y=torch.tensor([i % 2])))
    return out


class TestAgreement(unittest.TestCase):

    def test_layer_agreement_counts_cared_bits(self):
        layer = PrototypeLayer(4, 2, mask=True)
        with torch.no_grad():
            layer.proto_logits.copy_(torch.tensor([[5., 5., -5., -5.], [5., -5., 5., -5.]]))
            layer.mask_logits.copy_(torch.tensor([[5., 5., -5., -5.], [-5., -5., -5., -5.]]))
        x = torch.tensor([[1., 0., 1., 1.]])
        a = layer_agreement(layer, x)
        self.assertAlmostEqual(a[0, 0].item(), 0.5)          # cares about bits 0, 1: matches bit 0 only
        self.assertAlmostEqual(a[0, 1].item(), 1.0)          # cares about nothing: vacuously true

    def test_separation(self):
        y = np.array([0, 0, 1, 1])
        a = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
        np.testing.assert_allclose(separation(a, y), [1.0, 0.0])

    def test_log_agreement_every_level(self):
        loader = DataLoader(toy_graphs(), batch_size=8)
        for level in ('node', 'graph', 'both'):
            kw = {'num_graph_prototypes': 3} if level == 'both' else {}
            model = get_model(level, num_features=5, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=4, **kw)
            with tempfile.TemporaryDirectory() as d:
                m = log_agreement(model, {'val': loader, 'test': loader}, 'cpu', d, title=level)
                self.assertIsNotNone(m, level)
                rec = json.load(open(os.path.join(d, 'agreement.json')))
                self.assertEqual(set(rec['splits']), {'val', 'test'})
                pngs = [f for f in os.listdir(d) if f.endswith('.png')]
                self.assertEqual(len(pngs), 2 if level == 'both' else 1, level)
                arr = np.load(os.path.join(d, 'agreement.npz'))
                self.assertEqual(arr['test_layer0'].shape, (24, 4))
                self.assertTrue(((arr['test_layer0'] >= 0) & (arr['test_layer0'] <= 1)).all())

    def test_failure_never_raises(self):
        with tempfile.TemporaryDirectory() as d, self.assertWarns(UserWarning):
            self.assertIsNone(log_agreement(object(), {'test': []}, 'cpu', d))


if __name__ == '__main__':
    unittest.main()
