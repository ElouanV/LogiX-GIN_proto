import unittest

import torch
from torch_geometric.data import Data

from models_proto.model_proto import get_model
from utils.viz_proto import (citations, explain_graphs, head_literal_text, literal_kind, plot_examples,
                             plot_prototypes, plot_vocabulary, prototype_rule)


def graphs(n=24, f=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = 4 + i % 5
        x = torch.nn.functional.one_hot(torch.randint(0, 5, (k,), generator=g), f).float()
        src = torch.arange(k - 1)
        ei = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])
        out.append(Data(x=x, edge_index=ei, y=torch.tensor([i % 2])))
    return out


class TestVizProto(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = get_model('node', num_features=14, num_classes=2, hidden_dim=8, num_layers=2,
                               num_prototypes=4, proto_mask=True).eval()
        self.data = graphs()

    def test_records(self):
        recs = explain_graphs(self.model, self.data)
        self.assertEqual(len(recs), len(self.data))
        self.assertEqual(recs[0]['s_node'].shape, (len(self.data[0].x), 4))
        for r in recs:
            self.assertEqual(r['backed'], r['expl'] is not None)

    def test_names(self):
        self.assertIn('L', prototype_rule(self.model, 0))
        for i in range(self.model.fc.in_features):
            self.assertIn(literal_kind(self.model, i), ('always', 'present', 'absent', 'band'))
            self.assertIn(f'P{i % 4}', head_literal_text(self.model, i))

    def test_figures(self):
        recs = explain_graphs(self.model, self.data)
        self.assertTrue(plot_examples(self.model, recs, ds='synthetic').axes)
        self.assertTrue(plot_vocabulary({'m': self.model}).axes)
        if citations(self.model, recs)[0]:
            self.assertTrue(plot_prototypes(self.model, recs, self.data, ds='synthetic').axes)


if __name__ == '__main__':
    unittest.main()
