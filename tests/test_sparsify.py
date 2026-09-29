"""Hoyer measure and pruning used by sparsify_proto.py.

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
    from sparsify_proto import hoyer, hoyer_penalty, prune
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


class TestHoyer(unittest.TestCase):

    def test_extremes(self):
        W = torch.tensor([[0., 0., 3., 0.], [1., 1., 1., 1.]])
        h = hoyer(W)
        self.assertAlmostEqual(h[0].item(), 1.0, places=5)     # one weight per row
        self.assertAlmostEqual(h[1].item(), 0.0, places=5)     # uniform row

    def test_scale_invariant_and_per_row(self):
        W = torch.rand(4, 16)
        torch.testing.assert_close(hoyer(W), hoyer(7.0 * W))
        torch.testing.assert_close(hoyer(W)[2], hoyer(W[2:3])[0])

    def test_penalty_moves_mass_onto_few_weights(self):
        torch.manual_seed(0)
        m = Tiny()
        f = hoyer_penalty(m, 1.0, 1.0)
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        before = [hoyer(c.nn[0].weight).mean().item() for c in m.convs]
        for _ in range(200):
            opt.zero_grad()
            f(m).backward()
            opt.step()
        after = [hoyer(c.nn[0].weight).mean().item() for c in m.convs]
        for b, a in zip(before, after):
            self.assertGreater(a, b + 0.1)            # plateaus once pushed weights saturate the sigmoid


class TestPrune(unittest.TestCase):

    def test_prunes_exactly_the_small_weights(self):
        torch.manual_seed(1)
        m = Tiny()
        eps = float(m.convs[0].nn[0].weight.median())
        old = [ll.weight.detach().clone() for ll in [c.nn[0] for c in m.convs] + [m.fc]]
        frac = prune(m, eps)
        new = [ll.weight.detach() for ll in [c.nn[0] for c in m.convs] + [m.fc]]
        n_small = sum(int((o <= eps).sum()) for o in old)
        self.assertAlmostEqual(frac, n_small / sum(o.numel() for o in old))
        for o, n in zip(old, new):
            torch.testing.assert_close(n, torch.where(o > eps, o, torch.zeros_like(o)))

    def test_pruned_weights_stay_zero_under_training(self):
        torch.manual_seed(2)
        m = Tiny()
        prune(m, float(m.fc.weight.median()))
        zero = m.fc.weight.detach() == 0
        opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        for _ in range(20):
            opt.zero_grad()
            m.fc(torch.rand(6, 10)).sum().backward()
            opt.step()
        self.assertTrue((m.fc.weight.detach()[zero] == 0).all())


if __name__ == '__main__':
    unittest.main()
