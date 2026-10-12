"""utils/optim.py: AdamW default, Muon for the conv weights.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_optim.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models_proto.model_proto import get_model
    from utils.optim import MuonWithAuxAdamW, build_optimizer, muon_params, zeropower_via_newtonschulz5
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'utils.optim not importable: {e}')


def model():
    torch.manual_seed(0)
    return get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=2)


class TestBuild(unittest.TestCase):

    def test_default_is_torch_adamw(self):
        for name in (None, 'adamw'):
            opt = build_optimizer(model(), name, 1e-3, 1e-5)
            self.assertIs(type(opt), torch.optim.AdamW)

    def test_muon_gets_exactly_the_conv_weight_matrices(self):
        m = model()
        opt = build_optimizer(m, 'muon', 1e-3, 0.0)
        muon, other = opt.param_groups
        self.assertTrue(muon['use_muon'] and not other['use_muon'])
        self.assertEqual({id(p) for p in muon['params']}, {id(c.nn[0].weight_sigma) for c in m.convs})
        self.assertEqual(len(muon['params']) + len(other['params']), len(list(m.parameters())))
        self.assertEqual(muon['lr'], 0.02)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            build_optimizer(model(), 'sgd', 1e-3, 0.0)


class TestUpdates(unittest.TestCase):

    def test_aux_adamw_matches_torch_adamw(self):
        torch.manual_seed(1)
        p1 = torch.nn.Parameter(torch.randn(5, 3))
        p2 = torch.nn.Parameter(p1.detach().clone())
        ref = torch.optim.AdamW([p1], lr=1e-2, weight_decay=0.1)
        ours = MuonWithAuxAdamW([dict(params=[p2], use_muon=False, lr=1e-2, weight_decay=0.1)])
        for _ in range(20):
            g = torch.randn(5, 3)
            for p, o in ((p1, ref), (p2, ours)):
                o.zero_grad()
                p.grad = g.clone()
                o.step()
        torch.testing.assert_close(p1, p2)

    def test_newton_schulz_is_roughly_orthogonal(self):
        torch.manual_seed(2)
        for shape in ((8, 28), (64, 16)):
            s = torch.linalg.svdvals(zeropower_via_newtonschulz5(torch.randn(*shape)))
            self.assertTrue(((s > 0.6) & (s < 1.25)).all(), s)     # the reference's loose band

    def test_muon_decreases_a_quadratic(self):
        torch.manual_seed(3)
        W = torch.nn.Parameter(torch.randn(6, 10))
        target = torch.randn(6, 10)
        opt = MuonWithAuxAdamW([dict(params=[W], use_muon=True, lr=0.05)])
        start = float(((W - target) ** 2).sum())
        for _ in range(300):              # fixed-size steps: ~lr * sqrt(rows) in Frobenius norm
            opt.zero_grad()
            ((W - target) ** 2).sum().backward()
            opt.step()
        self.assertLess(float(((W - target) ** 2).sum()), 0.01 * start)

    def test_lr_schedulers_see_both_groups(self):
        opt = build_optimizer(model(), 'muon', 1e-3, 0.0)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=0)
        sched.step(1.0)
        sched.step(0.0)
        self.assertEqual([g['lr'] for g in opt.param_groups], [0.01, 5e-4])


class TestCli(unittest.TestCase):

    def test_default_keeps_results_paths(self):
        from train_proto import parse_cli
        _, _, args = parse_cli(['--dataset', 'MUTAG'])
        self.assertNotIn('optimizer', args)
        self.assertNotIn('muon_lr', args)
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--optimizer', 'adamw', '--muon_lr', '0.05'])
        self.assertNotIn('optimizer', args)
        self.assertNotIn('muon_lr', args)

    def test_muon_enters_the_args(self):
        from train_proto import parse_cli
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--optimizer', 'muon'])
        self.assertEqual(args['optimizer'], 'muon')
        self.assertNotIn('muon_lr', args)
        _, _, args = parse_cli(['--dataset', 'MUTAG', '--optimizer', 'muon', '--muon_lr', '0.05'])
        self.assertEqual(args['muon_lr'], 0.05)


if __name__ == '__main__':
    unittest.main()
