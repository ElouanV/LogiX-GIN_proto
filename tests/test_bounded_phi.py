"""models_proto/tell.py BoundedPhi: head literals that are exact thresholds x >= t.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_bounded_phi.py -v
"""
import io
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interp_metrics import head_literal_constants  # noqa: E402
from latent_logic import ATOMS  # noqa: E402
from min_covers import literal_true_intervals  # noqa: E402
from models_proto.model_proto import MODELS  # noqa: E402
from models_proto.tell import BoundedPhi  # noqa: E402


class TestBoundedPhi(unittest.TestCase):
    def test_every_literal_is_a_threshold(self):
        """Whatever w_ and t_, the truth set on [0, 1] is exactly [t, 1]: no band, no union."""
        torch.manual_seed(0)
        phi = BoundedPhi(2000)
        with torch.no_grad():
            phi.w_.normal_(0, 6)                  # slopes from ~0 to ~pi
            phi.t_.normal_(0, 4)                  # thresholds from ~0 to ~1
        for w, b, t in zip(phi.w.tolist(), phi.b.tolist(), phi.t.tolist()):
            self.assertLessEqual(w, math.pi + 1e-6)     # float32 pi
            ivs = literal_true_intervals(w, b, 0.0, 1.0)
            self.assertEqual(len(ivs), 1)
            self.assertAlmostEqual(ivs[0][0], t, places=6)
            self.assertAlmostEqual(ivs[0][1], 1.0, places=6)
            self.assertGreater(ivs[0][0], 0.0)    # never true on all of [0, 1]

    def test_binarised_output_matches_threshold(self):
        phi = BoundedPhi(3, threshold=0.3)
        phi.tau = 10                              # the end-to-end phase's sharper step
        x = torch.linspace(0, 1, 101).unsqueeze(1).repeat(1, 3)
        away = (x - phi.t).abs() > 1e-4               # float32 rounding at x = t itself
        torch.testing.assert_close((phi(x) >= 0.5)[away], (x >= phi.t)[away])

    def test_init_and_set_thresholds(self):
        phi = BoundedPhi(4)
        torch.testing.assert_close(phi.w, torch.ones(4))
        torch.testing.assert_close(phi.t, torch.full((4,), 0.5))
        phi.set_thresholds(torch.tensor([0.0, 0.2, 0.7, 1.0]))
        torch.testing.assert_close(phi.t, torch.tensor([0.02, 0.2, 0.7, 0.98]))

    def test_gradients_reach_slope_and_threshold(self):
        phi = BoundedPhi(5)
        phi(torch.rand(8, 5)).sum().backward()
        self.assertTrue(phi.w_.grad.abs().sum() > 0 and phi.t_.grad.abs().sum() > 0)


class TestBoundHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MODELS['node'](len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4, proto_mask=True).eval()

    def test_bounded_head_has_no_constant_literal_on_the_domain(self):
        self.assertTrue(all(c is True for c in head_literal_constants(self.model)))   # Phi init: always true
        self.model.bound_head()
        self.assertIsInstance(self.model.fc.phi_in, BoundedPhi)
        self.assertTrue(all(c is None for c in head_literal_constants(self.model)))

    def test_forward_and_pickle(self):
        self.model.bound_head()
        x = torch.rand(5, len(ATOMS)).round()
        e = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        out = self.model(x, e, torch.zeros(5, dtype=torch.long))
        buf = io.BytesIO()
        torch.save(self.model, buf)
        buf.seek(0)
        again = torch.load(buf, weights_only=False)
        torch.testing.assert_close(again(x, e, torch.zeros(5, dtype=torch.long)), out)

    def test_unbounded_readout_refused(self):
        model = MODELS['node'](len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4, pool_ops=('mean', 'max', 'sum'))
        with self.assertRaises(ValueError):
            model.bound_head()


if __name__ == '__main__':
    unittest.main()
