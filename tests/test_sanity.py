"""utils/sanity.py: model-randomisation cascade.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_sanity.py -v
"""
import os
import sys
import unittest

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_logic import ATOMS  # noqa: E402
from models.model import GINTELL  # noqa: E402
from models_proto.model_proto import MODELS  # noqa: E402
from tests.test_interp_metrics import one_hot, undirected  # noqa: E402
from utils.sanity import RandomisationTest, cascade, compare, randomise  # noqa: E402


def graphs():
    x = one_hot(['C', 'N', 'O', 'O', 'C', 'H'])
    e = undirected([(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)])
    return [Data(x=x, edge_index=e, y=torch.tensor([i % 2])) for i in range(6)]


class TestCascade(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MODELS['node'](len(ATOMS), 2, num_layers=2, hidden_dim=8, num_prototypes=4, proto_mask=True).eval()

    def test_order_is_top_down(self):
        self.assertEqual([n for n, _ in cascade(self.model)], ['head', 'prototypes', 'conv1', 'conv0'])
        self.assertEqual([n for n, _ in cascade(GINTELL(len(ATOMS), 2, num_layers=2, hidden_dim=8))],
                         ['head', 'conv1', 'conv0'])

    def test_permute_keeps_values_touches_only_the_stage(self):
        before = {n: p.clone() for n, p in self.model.named_parameters()}
        randomise(cascade(self.model)[0][1], 'permute', torch.Generator().manual_seed(1))
        for n, p in self.model.named_parameters():
            if n.startswith('fc.'):
                torch.testing.assert_close(p.view(-1).sort().values, before[n].view(-1).sort().values)
            else:
                torch.testing.assert_close(p, before[n])
        self.assertFalse(torch.equal(self.model.fc.weight_sigma, before['fc.weight_sigma']))

    def test_compare(self):
        ref = [{'pred': 0, 'nodes': [0, 1]}, {'pred': 1, 'nodes': []}]
        self.assertEqual(compare(ref, ref), {'pred_agreement': 1.0, 'cites_nodes': 0.5, 'node_jaccard': 1.0,
                                             'expl_identical': 1.0})
        out = compare(ref, [{'pred': 0, 'nodes': [1, 2]}, {'pred': 0, 'nodes': [3]}])
        self.assertEqual((out['pred_agreement'], out['node_jaccard'], out['expl_identical']), (0.5, 1 / 3, 0.0))

    def test_run(self):
        test = RandomisationTest(self.model, graphs(), 'Mutagenicity')
        for order in ('cascade', 'independent'):
            rows = test.run('permute', order, draws=1, log=lambda s: None)
            self.assertEqual([r['stage'] for r in rows], ['trained', 'head', 'prototypes', 'conv1', 'conv0'])
            self.assertEqual(rows[0]['pred_agreement'], 1.0)
        rows = test.run('init', 'independent', draws=1, log=lambda s: None)
        self.assertEqual(rows[1]['head_literals_constant'], 1.0)    # Phi init w=1, b=0: always true


if __name__ == '__main__':
    unittest.main()
