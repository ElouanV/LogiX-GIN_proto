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
    from models_proto.model_proto import get_model
    from sparsify_proto import (hoyer, hoyer_penalty, layerwise_sparsify, layerwise_steps, notebook_hoyer_loss, prune,
                                unit_groups, unit_hoyer_penalty, used_units)
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


class TestUnitPenalty(unittest.TestCase):

    def test_leaves_lower_units_unused(self):
        torch.manual_seed(0)
        m = Tiny()
        f = unit_hoyer_penalty(m, 1.0)
        before = used_units(m, 0.05)
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            f(m).backward()
            opt.step()
        after = used_units(m, 0.05)
        self.assertTrue(all(a < b for a, b in zip(after, before)), (before, after))

    def test_scale_invariant(self):
        torch.manual_seed(1)
        m = Tiny()
        f = unit_hoyer_penalty(m, 1.0)
        p = float(f(m))
        with torch.no_grad():
            for ll in [c.nn[0] for c in m.convs] + [m.fc]:
                ll.weight_exp += 1.0                          # every weight x e
        self.assertAlmostEqual(float(f(m)), p, places=5)


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

    def test_max_fanin_keeps_the_k_largest_per_unit(self):
        torch.manual_seed(3)
        m = Tiny()
        old = [ll.weight.detach().clone() for ll in [c.nn[0] for c in m.convs] + [m.fc]]
        prune(m, 0.0, max_fanin=3)
        for o, ll in zip(old, [c.nn[0] for c in m.convs] + [m.fc]):
            w = ll.weight.detach()
            self.assertTrue(((w > 0).sum(1) <= 3).all())
            top = o.topk(3, dim=1).values
            torch.testing.assert_close(w.sort(1, descending=True).values[:, :3], top)

    def test_schedule_steps_nest_and_head_has_its_own_cap(self):
        torch.manual_seed(4)
        m = Tiny()
        prev = None
        for k in (6, 4, 2):
            prune(m, 0.0, max_fanin=k, fc_fanin=max(k, 5))
            for c in m.convs:
                self.assertTrue(((c.nn[0].weight > 0).sum(1) <= k).all())
            self.assertTrue(((m.fc.weight > 0).sum(1) <= max(k, 5)).all())
            alive = [ll.weight.detach() > 0 for ll in [c.nn[0] for c in m.convs] + [m.fc]]
            if prev is not None:                               # a later cut never revives a weight
                self.assertTrue(all((a <= p).all() for a, p in zip(alive, prev)))
            prev = alive

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
