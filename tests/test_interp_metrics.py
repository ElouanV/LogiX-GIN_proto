"""interp_metrics.py: shortest explanation, toxicophore ground truth, symbolic forward.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_interp_metrics.py -v
"""
import itertools
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interp_metrics import (head_literal_map, shortest_explanation, symbolic_head,  # noqa: E402
                            toxicophore_nodes, trunk_states)
from latent_logic import ATOMS  # noqa: E402
from models_proto.model_proto import MODELS  # noqa: E402


def one_hot(atoms):
    x = torch.zeros(len(atoms), len(ATOMS))
    x[torch.arange(len(atoms)), [ATOMS.index(a) for a in atoms]] = 1
    return x


def undirected(pairs):
    e = torch.tensor(pairs).t()
    return torch.cat([e, e.flip(0)], 1)


class TestShortestExplanation(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = 8
            w = rng.exponential(1.0, n)
            true = rng.random(n) < 0.6
            S = rng.uniform(0, w[true].sum() + 0.5)
            got = shortest_explanation(w, S, true)
            idx = np.flatnonzero(true)
            best = next((k for k in range(len(idx) + 1)
                         if any(w[list(c)].sum() >= S for c in itertools.combinations(idx, k))), None)
            if best is None:
                self.assertIsNone(got)
            else:
                self.assertEqual(len(got), best)
                self.assertTrue(true[got].all())
                self.assertGreaterEqual(w[got].sum(), S)

    def test_nonpositive_threshold_needs_nothing(self):
        self.assertEqual(len(shortest_explanation(np.ones(3), 0.0, np.ones(3, bool))), 0)


class TestToxicophore(unittest.TestCase):
    def test_nitro_and_amine(self):
        # nitro 1(2,3) on C0, amine 5(6,7) on C4; H8 and C9 hang off C4, two hops from the amine N
        atoms = ['C', 'N', 'O', 'O', 'C', 'N', 'H', 'H', 'H', 'C']
        e = undirected([(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (5, 6), (5, 7), (4, 8), (8, 9)])
        strict = toxicophore_nodes(one_hot(atoms), e, hops=0)
        self.assertEqual(torch.nonzero(strict).flatten().tolist(), [1, 2, 3, 5, 6, 7])
        grown = toxicophore_nodes(one_hot(atoms), e, hops=1)
        self.assertEqual(torch.nonzero(grown).flatten().tolist(), [0, 1, 2, 3, 4, 5, 6, 7])

    def test_single_o_is_not_nitro(self):
        e = undirected([(0, 1), (1, 2)])
        self.assertFalse(toxicophore_nodes(one_hot(['C', 'N', 'O']), e).any())


class TestSymbolicForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = one_hot(['C', 'N', 'O', 'O', 'C', 'H'])
        self.e = undirected([(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)])
        self.batch = torch.zeros(6, dtype=torch.long)

    def test_every_level(self):
        for level, cls in MODELS.items():
            for mask in (False, True):
                m = cls(len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4, proto_mask=mask).eval()
                xs = trunk_states(m, self.x, self.e)
                self.assertTrue(all(((x == 0) | (x == 1)).all() for x in xs))
                margin, lit, s_node = symbolic_head(m, xs, self.batch)
                self.assertEqual(margin.shape, (1, 2))
                self.assertEqual(lit.shape[1], len(head_literal_map(m)))
                self.assertEqual(s_node is None, level == 'graph')

    def test_saturated_network_equals_symbolic(self):
        """With steep literals and units, the network's states are the rules' truth values."""
        m = MODELS['node'](len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4).eval()
        for conv in m.convs:
            conv.nn[0].phi_in.tau = 50
            with torch.no_grad():
                conv.nn[0].weight_exp.fill_(3.0)
        net = trunk_states(m, self.x, self.e, symbolic=False)
        sym = trunk_states(m, self.x, self.e)
        for n, s in zip(net, sym):
            far = (n - 0.5).abs() > 0.1
            self.assertTrue(((n[far] >= 0.5).float() == s[far]).all())


if __name__ == '__main__':
    unittest.main()
