"""models_proto/model_proto_count.py: the head reads how many / which share of the nodes
satisfy each prototype rule (tell.RampPhi literals, interp_metrics reading).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_count_readout.py -v
"""
import io
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interp_metrics import (head_literal_constants, head_literal_map, interp_metrics, level_of,  # noqa: E402
                            readout_values, symbolic_head, trunk_states)
from latent_logic import ATOMS  # noqa: E402
from min_covers import phi_true_intervals  # noqa: E402
from models_proto.model_proto_count import GINTELLProtoCount  # noqa: E402
from models_proto.tell import RampPhi  # noqa: E402
from tests.test_interp_metrics import one_hot, undirected  # noqa: E402


class TestRampPhi(unittest.TestCase):
    def test_binarised_literal_is_the_threshold_for_any_slope(self):
        torch.manual_seed(0)
        phi = RampPhi(6, bounded=[True, True, True, False, False, False])
        with torch.no_grad():
            phi.w_.copy_(torch.tensor([-2.0, 0.0, 4.0, -2.0, 0.0, 4.0]))       # slopes 0.14 .. 55
            phi.t_.copy_(torch.tensor([-1.0, 0.5, 2.0, 0.5, 3.5, 40.5]))
        x = torch.cat([torch.rand(200, 3), torch.randint(0, 60, (200, 3)).float()], 1)
        away = (x - phi.t).abs() > 1e-4
        torch.testing.assert_close((phi(x) >= 0.5)[away], (x >= phi.t)[away])
        for i in range(6):
            lo, hi = float(x[:, i].min()), float(x[:, i].max())
            self.assertEqual(phi_true_intervals(phi, i, lo, hi), [(max(lo, float(phi.t[i])), hi)])

    def test_monotone(self):
        phi = RampPhi(1, bounded=False, slope=0.3)
        x = torch.linspace(-30, 30, 601).unsqueeze(1)
        self.assertTrue((phi(x).diff(dim=0) >= 0).all())

    def test_set_thresholds_offsets_counts(self):
        phi = RampPhi(3, bounded=[True, False, False], offset=[0.0, 0.5, -0.5])
        phi.set_thresholds(torch.tensor([0.3, 2.0, -2.0]))
        torch.testing.assert_close(phi.t, torch.tensor([0.3, 2.5, -2.5]))


class TestCountModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = GINTELLProtoCount(len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4).eval()
        self.x = one_hot(['C', 'N', 'O', 'O', 'C', 'H'])
        self.e = undirected([(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)])
        self.batch = torch.zeros(6, dtype=torch.long)
        g = torch.Generator().manual_seed(0)
        self.h = torch.randint(0, 2, (6, 16), generator=g).float()     # binary node states, 2 layers x 8
        with torch.no_grad():                      # prototype k = node k's states on 4 cared bits,
            p = self.m.proto                       # tolerances of 0, 0, 1 and 2 mismatches
            p.proto_logits.copy_(6 * self.h[:4] - 3)
            p.mask_logits.fill_(-3.0)
            p.mask_logits[:, :4] = 3.0
            self.m.predicate_phi.t_.copy_(torch.logit(torch.tensor([0.9, 0.5, 0.3, 0.12])))

    def test_needs_masked_prototypes(self):
        with self.assertRaises(ValueError):
            GINTELLProtoCount(len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4, proto_mask=False)

    def test_layout(self):
        self.assertEqual(level_of(self.m), 'node')
        self.assertEqual(self.m.pool_ops, ('count', 'share'))
        self.assertEqual(len(self.m.head_columns()), 4 * 4)
        self.assertEqual(self.m.fc.in_features, 16)
        self.assertIsInstance(self.m.fc.phi_in, RampPhi)
        self.assertEqual([op for _, _, op, _ in head_literal_map(self.m)], ['count'] * 4 + ['share'] * 4 + ['count'] * 4 + ['share'] * 4)

    def test_predicate_is_the_tolerant_rule(self):
        """q_k(v) = 1 exactly when node v has at most mu_k mismatches on P_k's cared bits; the
        readout counts those nodes and divides by |V|."""
        h = self.h
        xs = [h[:, :8], h[:, 8:]]
        p = self.m.proto
        mism = h @ (p.care * (1 - p.prototypes)).t() + (1 - h) @ (p.care * p.prototypes).t()
        q = (mism <= self.m.tolerance() + 1e-5).float()
        self.assertTrue(0 < q.sum() < q.numel())
        s, q_sym = readout_values(self.m, xs, self.batch)
        torch.testing.assert_close(q_sym, q)
        torch.testing.assert_close(s, torch.hstack([q.sum(0), q.mean(0)]).unsqueeze(0))

    def test_hard_mode_is_the_rule_model(self):
        """set_hard: the network predicts the argmax of the symbolic rule margins (fidelity 1)."""
        self.m.set_hard(True)
        margin = symbolic_head(self.m, trunk_states(self.m, self.x, self.e), self.batch)[0]
        torch.testing.assert_close(self.m(self.x, self.e, self.batch), torch.sigmoid(10 * margin))

    def test_full_quant_noise_is_the_rule_model(self):
        self.m.set_quant_noise(1.0)
        margin = symbolic_head(self.m, trunk_states(self.m, self.x, self.e), self.batch)[0]
        torch.testing.assert_close(self.m(self.x, self.e, self.batch, margin=True), margin)

    def test_no_literal_constant_on_the_unit_interval(self):
        const = head_literal_constants(self.m)
        self.assertTrue(all(c is None for c in const))       # counts: unknown range; shares: [t, 1]

    def test_gradients_reach_tolerances_and_head_thresholds(self):
        self.m.train()
        self.m(self.x, self.e, self.batch).sum().backward()
        self.assertGreater(float(self.m.predicate_phi.t_.grad.abs().sum()), 0)
        self.assertGreater(float(self.m.fc.phi_in.t_.grad.abs().sum()), 0)

    def test_pickle(self):
        out = self.m(self.x, self.e, self.batch)
        buf = io.BytesIO()
        torch.save(self.m, buf)
        buf.seek(0)
        torch.testing.assert_close(torch.load(buf, weights_only=False)(self.x, self.e, self.batch), out)

    def test_metrics_cite_the_satisfying_nodes(self):
        from torch_geometric.data import Data
        data = [Data(x=self.x, edge_index=self.e, y=torch.tensor([i % 2])) for i in range(4)]
        res = interp_metrics(self.m, data, 'cpu', 'Mutagenicity')
        self.assertEqual(res['logic_fidelity'] is not None, True)
        self.assertIn('gt_precision_evidence', res)


if __name__ == '__main__':
    unittest.main()
