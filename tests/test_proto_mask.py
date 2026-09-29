"""The optional binary care mask of PrototypeLayer (models_proto/proto.py).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_proto_mask.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models_proto.proto import PrototypeLayer
    from models_proto.model_proto import get_model
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'models_proto not importable: {e}')


class TestMaskedPrototype(unittest.TestCase):

    def test_unmasked_is_unchanged_hamming(self):
        torch.manual_seed(0)
        layer = PrototypeLayer(12, 3)
        x = (torch.rand(5, 12) > 0.5).float()
        p = layer.prototypes.detach()
        ref = (x[:, None, :] == p[None]).float().mean(2)
        torch.testing.assert_close(layer.similarity(x), ref)
        self.assertTrue(torch.equal(layer.care, torch.ones(3, 12)))

    def test_similarity_is_a_conjunction_over_cared_bits(self):
        torch.manual_seed(1)
        layer = PrototypeLayer(10, 4, mask=True)
        with torch.no_grad():
            layer.mask_logits.copy_(torch.where(torch.rand(4, 10) > 0.6, 3.0, -3.0))
        layer.mask_temp = 0.7
        x = (torch.rand(50, 10) > 0.5).float()
        p, c = layer.prototypes.detach(), layer.care.detach()
        mism = ((x[:, None, :] != p[None]) & (c[None] > 0)).float().sum(2)
        s = layer.similarity(x).detach()
        torch.testing.assert_close(s, torch.exp(-mism / 0.7))
        self.assertTrue(torch.equal(s == 1, mism == 0))      # 1 exactly when every cared bit matches

    def test_uncared_bits_are_ignored_and_mask_is_binary(self):
        torch.manual_seed(2)
        layer = PrototypeLayer(8, 2, mask=True)
        with torch.no_grad():
            layer.mask_logits[:, 4:] = -3.0
        x = (torch.rand(6, 8) > 0.5).float()
        y = x.clone()
        y[:, 4:] = 1 - y[:, 4:]                               # flip only the uncared bits
        torch.testing.assert_close(layer.similarity(x), layer.similarity(y))
        self.assertTrue(set(layer.care.detach().unique().tolist()) <= {0.0, 1.0})

    def test_mask_penalty_shrinks_the_cared_set(self):
        torch.manual_seed(3)
        layer = PrototypeLayer(16, 2, mask=True)
        opt = torch.optim.Adam(layer.parameters(), lr=0.1)
        x = torch.rand(4, 16)
        for _ in range(100):
            opt.zero_grad()
            layer(x)
            layer.mask_size.backward()
            opt.step()
        layer(x)
        self.assertLess(float(layer.mask_size), 0.5)

    def test_models_build_with_and_without_mask(self):
        for level in ('node', 'graph', 'both'):
            for mask in (False, True):
                m = get_model(level, num_features=5, num_classes=2, hidden_dim=4, num_layers=2,
                              num_prototypes=3, proto_mask=mask)
                x = torch.rand(7, 5)
                ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
                out = m(x, ei, torch.tensor([0, 0, 0, 1, 1, 1, 1]))
                self.assertEqual(tuple(out.shape), (2, 2))
                self.assertTrue(all(p.masked == mask for p in m.proto_layers))

    def test_old_pickles_without_the_attribute_still_work(self):
        layer = PrototypeLayer(6, 2)
        del layer.mask                                        # as unpickled from before the option
        x = torch.rand(3, 6)
        self.assertFalse(layer.masked)
        self.assertEqual(tuple(layer.similarity(x).shape), (3, 2))


class TestMaskTemperature(unittest.TestCase):
    def test_default_anneals_over_all_epochs(self):
        from train_proto import mask_temperature
        a = {'epochs': 101, 'mask_temp_end': 1.0}
        self.assertAlmostEqual(mask_temperature(a, 0, 64), 16.0)
        self.assertAlmostEqual(mask_temperature(a, 50, 64), 4.0)
        self.assertAlmostEqual(mask_temperature(a, 100, 64), 1.0)

    def test_anneal_frac_reaches_the_end_early_then_holds(self):
        from train_proto import mask_temperature
        a = {'epochs': 101, 'mask_temp_end': 1.0, 'mask_anneal_frac': 0.5}
        self.assertAlmostEqual(mask_temperature(a, 0, 64), 16.0)
        self.assertAlmostEqual(mask_temperature(a, 49.5 / 2, 64), 4.0)
        self.assertAlmostEqual(mask_temperature(a, 50, 64), 1.0, places=1)
        self.assertEqual(mask_temperature(a, 80, 64), 1.0)


if __name__ == '__main__':
    unittest.main()
