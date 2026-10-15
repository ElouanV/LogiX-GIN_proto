"""binarise_ablation.py: each condition binarises exactly its part; 'rule' is the symbolic model.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_binarise_ablation.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.data import Batch
    from binarise_ablation import binarise_ablation, conditions, predict
    from interp_metrics import symbolic_head, trunk_states
    from tests.test_qat import batch, model
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'binarise_ablation not importable: {e}')


class TestBinariseAblation(unittest.TestCase):

    def test_conditions_cover_parts_and_layers(self):
        c = conditions(3)
        self.assertEqual(list(c), ['soft', 'head', 'trunk', 'trunk_lits', 'trunk_outs', 'L0', 'L1', 'L2', 'rule'])
        self.assertEqual(c['L1'], ([0, 1, 0], [0, 1, 0], 0))
        self.assertEqual(c['rule'], ([1, 1, 1], [1, 1, 1], 1))

    def test_rule_condition_is_the_symbolic_model_and_soft_is_the_network(self):
        m, b = model(), batch()
        loader = [b]
        p_rule, y = predict(m, loader, 'cpu', *conditions(2)['rule'])
        margin = symbolic_head(m, trunk_states(m, b.x.float(), b.edge_index), b.batch)[0]
        self.assertTrue((p_rule == margin.argmax(1).numpy()).all())
        p_soft, _ = predict(m, loader, 'cpu', *conditions(2)['soft'])
        with torch.no_grad():
            net = m(b.x.float(), b.edge_index, b.batch, margin=True).argmax(1).numpy()
        self.assertTrue((p_soft == net).all())

    def test_hooks_are_reset_and_result_has_every_condition(self):
        m, b = model(), batch()
        res = binarise_ablation(m, b.to_data_list(), 'cpu')
        self.assertEqual(res['soft']['agree'], 1.0)
        self.assertEqual(len(res['graded_outputs']), 2)
        self.assertTrue(all(c.nn[0].quant_rate == 0.0 for c in m.convs) and m.fc.quant_rate == 0.0)


if __name__ == '__main__':
    unittest.main()
