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

from interp_metrics import (base_literal_map, head_literal_map, interp_metrics, readout_values,  # noqa: E402
                            shortest_explanation, symbolic_head, toxicophore_nodes, trunk_states)
from latent_logic import ATOMS  # noqa: E402
from models.model import GINTELL  # noqa: E402
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


    def test_hard_mode_computes_exactly_the_rules(self):
        """set_hard: the network's trunk states are the symbolic ones, and its prediction is
        the argmax of the rule margins."""
        torch.manual_seed(1)
        for level, cls in MODELS.items():
            for mask in (False, True):
                m = cls(len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4, proto_mask=mask).eval()
                m.set_hard(True)
                net = trunk_states(m, self.x, self.e, symbolic=False)
                sym = trunk_states(m, self.x, self.e)
                for n, s in zip(net, sym):
                    self.assertTrue(torch.equal(n, s))
                margin = symbolic_head(m, sym, self.batch)[0]
                out = m(self.x, self.e, self.batch)
                torch.testing.assert_close(out, torch.sigmoid(10 * margin))


class TestClassicLogiXGIN(unittest.TestCase):
    """interp_metrics on models/model.py GINTELL (no prototypes)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = GINTELL(len(ATOMS), 2, num_layers=2, hidden_dim=8).eval()
        self.x = one_hot(['C', 'N', 'O', 'O', 'C', 'H'])
        self.e = undirected([(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)])
        self.batch = torch.zeros(6, dtype=torch.long)

    def test_readout_matches_forward(self):
        """Fed the network's own states, the readout reproduces the model's head input."""
        xs = trunk_states(self.model, self.x, self.e, symbolic=False)
        s, s_node = readout_values(self.model, xs, self.batch)
        self.assertIsNone(s_node)
        torch.testing.assert_close(self.model.fc(torch.hstack([s, 1 - s])), self.model(self.x, self.e, self.batch))

    def test_literal_map(self):
        lits = base_literal_map(self.model)
        self.assertEqual(len(lits), self.model.fc.in_features)
        self.assertEqual(lits[0], (0, 'mean', True))
        self.assertEqual(lits[16], (0, 'max', True))         # 2 layers x 8 units per pool block
        self.assertEqual(lits[48 + 33], (1, 'sum', False))

    def test_metrics_run(self):
        from torch_geometric.data import Data
        data = [Data(x=self.x, edge_index=self.e, y=torch.tensor([i % 2])) for i in range(8)]
        res = interp_metrics(self.model, data, 'cpu', 'Mutagenicity')
        self.assertIsNone(res['protos_cited'])
        self.assertLessEqual(res['units_cited'], 16)
        if res['expl_bits_mean'] is not None:
            self.assertEqual(res['expl_bits_mean'], res['expl_literals_mean'])


if __name__ == '__main__':
    unittest.main()
