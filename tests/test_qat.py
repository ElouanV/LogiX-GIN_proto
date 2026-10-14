"""QAT by Quant-Noise: models_proto/tell.quant_noise, GINTELLProtoBase.set_quant_noise, qat_proto.py.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_qat.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.data import Batch, Data
    from interp_metrics import symbolic_head, trunk_states
    from models_proto.model_proto import get_model
    from models_proto.tell import quant_noise
    from qat_proto import _trunk, diagnostics, ramp_rate, rule_losses
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'qat modules not importable: {e}')


def batch(seed=0, graphs=6):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(graphs):
        n = 7
        x = (torch.rand(n, 3, generator=g) > 0.5).float()
        src = torch.randint(0, n, (12,), generator=g)
        dst = torch.randint(0, n, (12,), generator=g)
        out.append(Data(x=x, edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
                        y=torch.randint(0, 2, (1,), generator=g)))
    return Batch.from_data_list(out)


def model(level='node'):
    torch.manual_seed(0)
    return get_model(level, num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=3).eval()


class TestQuantNoise(unittest.TestCase):

    def test_rate_zero_is_identity_and_rate_one_binarises(self):
        p = torch.rand(50, 4)
        self.assertIs(quant_noise(p, 0.0), p)
        torch.testing.assert_close(quant_noise(p, 1.0), (p >= 0.5).float())
        torch.testing.assert_close(quant_noise(p, 1.0, training=False), (p >= 0.5).float())

    def test_partial_rate_mixes_in_training_and_is_soft_in_eval(self):
        torch.manual_seed(0)
        p = torch.rand(2000, 5) * 0.8 + 0.1                 # never exactly 0 / 1
        q = quant_noise(p, 0.3)
        share = ((q == 0) | (q == 1)).float().mean().item()
        self.assertAlmostEqual(share, 0.3, delta=0.03)
        self.assertTrue(torch.equal(quant_noise(p, 0.3, training=False), p))

    def test_straight_through_gradient(self):
        p = torch.rand(40, 3, requires_grad=True)
        quant_noise(p, 0.5).sum().backward()
        torch.testing.assert_close(p.grad, torch.ones_like(p))

    def test_ramp(self):
        self.assertEqual(ramp_rate(0, 300, 1.0, 2 / 3), 0.0)
        self.assertAlmostEqual(ramp_rate(100, 300, 1.0, 2 / 3), 0.5)
        self.assertEqual(ramp_rate(250, 300, 1.0, 2 / 3), 1.0)
        self.assertAlmostEqual(ramp_rate(250, 300, 0.5, 2 / 3), 0.5)


class TestModel(unittest.TestCase):

    def test_off_by_default_and_at_rate_zero(self):
        b = batch()
        m = model()
        ref = m(b.x, b.edge_index, b.batch)
        m.set_quant_noise(0.0)
        self.assertTrue(torch.equal(m(b.x, b.edge_index, b.batch), ref))

    def test_rate_one_is_the_rule_model_interp_metrics_reads(self):
        """Same unit states as interp_metrics.trunk_states, same head literals as symbolic_head
        (the head's graded output can saturate to ties, so predictions use the margins)."""
        b = batch()
        for level in ('node', 'graph'):
            m = model(level)
            m.set_quant_noise(1.0)
            with torch.no_grad():
                xs, scores = _trunk(m, b, 1.0)
                ref = trunk_states(m, b.x, b.edge_index)
                margin, lit, _ = symbolic_head(m, ref, b.batch)
                for a, r in zip(xs, ref):
                    self.assertTrue(torch.equal(a, r), level)
                self.assertTrue(torch.equal(scores, margin), level)
                self.assertTrue(torch.equal(m.fc.phi_in(m.head_input(m.readout(xs, b.batch))) >= 0.5, lit))

    def test_diagnostics_restore_the_rate_and_mode(self):
        m = model().train()
        m.set_quant_noise(0.4)
        d = diagnostics(m, [batch()], torch.device('cpu'))
        self.assertEqual((m.quant_rate, m.training), (0.4, True))
        self.assertTrue(0 <= d['val_fidelity'] <= 1)
        self.assertEqual({k for k in d if k.startswith('val_flip_L')}, {'val_flip_L0', 'val_flip_L1'})

    def test_margin_forward_at_rate_one_is_symbolic_head(self):
        """The rule-model loss is computed on forward(margin=True) at rate 1, in training
        mode: it must be the margins interp_metrics reads, with gradients."""
        b = batch()
        for level in ('node', 'graph'):
            m = model(level).train()
            m.set_quant_noise(1.0)
            out = m(b.x, b.edge_index, b.batch, margin=True)
            with torch.no_grad():
                margin, _, _ = symbolic_head(m, trunk_states(m, b.x, b.edge_index), b.batch)
            self.assertTrue(torch.allclose(out, margin, atol=1e-5), level)
            out.sum().backward()
            self.assertGreater(m.convs[0].nn[0].weight_sigma.grad.abs().sum().item(), 0, level)

    def test_rule_losses_restore_the_rate_and_are_zero_when_off(self):
        import copy
        b = batch()
        m = model().train()
        frozen = copy.deepcopy(m).eval()
        frozen.set_quant_noise(0.0)
        m.set_quant_noise(0.3)
        y = b.y.view(-1).long()
        self.assertEqual(rule_losses(m, frozen, 0.0, 0.0)(b.x, b.edge_index, b.batch, y), 0.0)
        loss = rule_losses(m, frozen, 1.0, 1.0)(b.x, b.edge_index, b.batch, y)
        self.assertEqual(m.quant_rate, 0.3)
        self.assertTrue(torch.isfinite(loss) and loss.requires_grad)


class TestRunPath(unittest.TestCase):

    def test_split_run_path_reads_runs_under_any_root(self):
        """hps trials and k-fold final runs (QAT fine-tunes them) live outside results_proto/."""
        import json
        import tempfile
        from sparsify_proto import split_run_path
        self.assertEqual(split_run_path('results_proto/MUTAG/node-x/base/2'), ('proto', 'MUTAG', 'node-x', 'base', '2'))
        with tempfile.TemporaryDirectory() as t:
            d = os.path.join(t, 'results_final', 'exp', 'MUTAG', 'nmp', 'fold3', 'run', 'MUTAG', 'node-x', 'base|split=kfold', '3')
            os.makedirs(d)
            with open(os.path.join(d, 'args.json'), 'w') as f:
                json.dump({'proto_level': 'node'}, f)
            self.assertEqual(split_run_path(d), ('proto', 'MUTAG', 'node-x', 'base|split=kfold', '3'))
            with self.assertRaises(ValueError):
                split_run_path(os.path.dirname(d))


if __name__ == '__main__':
    unittest.main()
