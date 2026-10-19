"""Fixes of the student head saturation (train_logic.py / train_proto.py --head_init center
--logit_loss).

At the random LogicalLayer init the head's sigmoid(10 (W.lit + b)) rounds to exactly 1, and
F.binary_cross_entropy then gives the head no gradient: on MUTAG every NMP fold predicted
one class. --logit_loss computes the same BCE from the logits; --head_init center shifts the
head bias so each class's mean pre-activation on the training graphs is 0.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_head_saturation.py -v
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tell import LogicalLayer as UpstreamLogicalLayer  # noqa: E402
from models_proto.tell import LogicalLayer  # noqa: E402
from utils.training import class_loss  # noqa: E402

try:
    from torch_geometric.loader import DataLoader
    import train_logic
    import train_proto
    from models.model import GIN, GINTELL
    from models_proto.gintell import GINTELLPool
    from models_proto.model_proto import get_model
    from tests.test_val_loss import graphs
except Exception as e:                                   # torch_geometric missing etc.
    train_proto = None
    IMPORT_ERROR = str(e)


class TestLogitLoss(unittest.TestCase):

    def test_same_loss_and_gradient_when_not_saturated(self):
        torch.manual_seed(0)
        y = torch.tensor([0, 1, 1, 0])
        grads = []
        for use_logits in (False, True):
            z = torch.linspace(-0.3, 0.3, 8).reshape(4, 2).requires_grad_()
            out = torch.sigmoid(z)
            loss = class_loss(out, y, 2, z if use_logits else None)
            loss.backward()
            grads.append((loss.item(), z.grad.clone()))
        self.assertAlmostEqual(grads[0][0], grads[1][0], places=5)
        self.assertTrue(torch.allclose(grads[0][1], grads[1][1], atol=1e-6))

    def test_saturated_wrong_output_keeps_its_gradient(self):
        y = torch.tensor([0])
        z = torch.tensor([[30.0, 30.0]], requires_grad=True)       # sigmoid = exactly 1.0 for both
        F.binary_cross_entropy(torch.sigmoid(z).reshape(-1), F.one_hot(y, 2).float().reshape(-1)).backward()
        self.assertEqual(z.grad.abs().sum().item(), 0.0)            # what kept the heads stuck
        z.grad = None
        class_loss(torch.sigmoid(z), y, 2, z).backward()
        self.assertGreater(z.grad[0, 1].item(), 0.4)                # the wrong class is pushed down

    def test_logical_layer_logits(self):
        torch.manual_seed(0)
        x = torch.rand(5, 10)
        for fc in (LogicalLayer(10, 2), UpstreamLogicalLayer(10, 2)):
            out = fc(x)
            self.assertTrue(torch.allclose(torch.sigmoid(fc.logits), out, atol=1e-6))
            fc(x, discrete_output=True)
            self.assertIsNone(fc.logits)                             # binarised output: no logits


@unittest.skipIf(train_proto is None, 'training modules not importable')
class TestCenterHeadBias(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1)
        self.teacher = GIN(3, 2, num_layers=2, hidden_dim=8)
        self.loader = DataLoader(graphs(24, 0), batch_size=8, shuffle=False)

    def _margins(self, m):
        with torch.no_grad():
            zs = []
            for d in self.loader:
                _, ly = self.teacher.forward_e(d.x, d.edge_index, d.batch)
                zs.append(m.head(m.readout(ly[:-1], d.batch), margin=True))
            return torch.cat(zs)

    def test_mean_margin_is_zero_and_only_the_bias_moves(self):
        for level in ('node', 'graph'):
            torch.manual_seed(0)
            m = get_model(level, num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=3,
                          proto_mask=level == 'node')
            before = {k: v.clone() for k, v in m.state_dict().items()}
            torch.manual_seed(5)
            shift = train_proto.center_head_bias(self.teacher, m, self.loader, 'cpu')
            self.assertEqual(shift.shape, (2,))
            changed = [k for k, v in m.state_dict().items() if not torch.equal(v, before[k])]
            self.assertEqual(changed, ['fc.b'])
            # the teacher's Gumbel states are re-sampled here, so the mean is ~0, not exactly 0
            self.assertLess(self._margins(m).mean(0).abs().max().item(), 0.5 * shift.abs().max().item() + 1e-3)

    def test_classic_mean_margin_is_zero_and_only_the_bias_moves(self):
        for make in (lambda: GINTELL(3, 2, num_layers=2, hidden_dim=8),
                     lambda: GINTELLPool(3, 2, num_layers=2, hidden_dim=8, pool_ops=('mean', 'max'))):
            torch.manual_seed(0)
            m = make()
            before = {k: v.clone() for k, v in m.state_dict().items()}
            torch.manual_seed(5)
            shift = train_logic.center_head_bias(self.teacher, m, self.loader, 'cpu')
            changed = [k for k, v in m.state_dict().items() if not torch.equal(v, before[k])]
            self.assertEqual(changed, ['fc.b'])
            with torch.no_grad():                                    # Gumbel states re-sampled: ~0
                torch.manual_seed(5)
                z = []
                for d in self.loader:
                    lx, _ = self.teacher.forward_e(d.x, d.edge_index, d.batch)
                    m.fc(train_logic.head_input(m, lx))
                    z.append(m.fc.logits / 10)
            self.assertLess(torch.cat(z).mean(0).abs().max().item(), 0.5 * shift.abs().max().item() + 1e-3)

    def test_rng_untouched(self):
        m = get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=3)
        torch.manual_seed(7)
        train_proto.center_head_bias(self.teacher, m, self.loader, 'cpu')
        a = torch.rand(3)
        torch.manual_seed(7)
        self.assertTrue(torch.equal(a, torch.rand(3)))

    def test_cli_flags_only_enter_the_results_path_when_set(self):
        for script in (train_logic, train_proto):
            _, _, off = script.parse_cli(['--dataset', 'MUTAG'])
            self.assertNotIn('head_init', off)
            self.assertNotIn('logit_loss', off)
            _, _, on = script.parse_cli(['--dataset', 'MUTAG', '--head_init', 'center', '--logit_loss'])
            self.assertEqual((on['head_init'], on['logit_loss']), ('center', True))
        base = ['--dataset', 'MUTAG', '--proto_level', 'node']
        _, _, on = train_proto.parse_cli(base + ['--head_init', 'center', '--logit_loss'])
        self.assertEqual((on['head_init'], on['logit_loss']), ('center', True))
        self.assertTrue(train_proto.loss_weights({**on, 'conv_reg': 0, 'fc_reg': 0, 'proto_div_reg': 0,
                                                  'proto_ent_reg': 0})['logit_loss'])



@unittest.skipIf(train_proto is None, 'training modules not importable')
class TestProtoInit(unittest.TestCase):
    """--proto_init data (a searched hyper-parameter): prototypes start on training rows."""

    def setUp(self):
        torch.manual_seed(1)
        self.teacher = GIN(3, 2, num_layers=2, hidden_dim=8)
        self.loader = DataLoader(graphs(24, 0), batch_size=8, shuffle=False)

    def test_farthest_point_rows(self):
        from models_proto.proto import PrototypeLayer
        p = PrototypeLayer(4, 3)
        x = torch.tensor([[0., 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 0, 0]])
        idx = p.init_from_rows(x, torch.Generator().manual_seed(0)).tolist()
        self.assertEqual(len(set(idx)), 3)
        self.assertTrue(torch.equal(p.prototypes.detach(), x[idx]))            # binary prototypes = the rows
        first = idx[0]
        far = (x != x[first]).sum(1)
        self.assertEqual(far[idx[1]].item(), far.max().item())                  # second = farthest from first

    def test_data_init_every_level_deterministic_and_rng_untouched(self):
        for level in ('node', 'graph', 'both'):
            out = []
            for _ in range(2):
                torch.manual_seed(0)
                m = get_model(level, num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=3)
                torch.manual_seed(7)
                idxs = train_proto.init_prototypes(self.teacher, m, self.loader, 'cpu', seed=3)
                after = torch.rand(3)
                out.append((idxs, [p.proto_logits.clone() for p in m.proto_layers], after))
            torch.manual_seed(7)
            self.assertTrue(torch.equal(out[0][2], torch.rand(3)), level)       # global RNG not consumed
            self.assertEqual(len(out[0][0]), len(m.proto_layers))
            for l in out[0][1]:
                self.assertTrue(bool((l.abs() == 3.0).all()), level)             # set to +-magnitude

    def test_cli_and_space(self):
        from utils import hps
        _, _, off = train_proto.parse_cli(['--dataset', 'MUTAG'])
        self.assertNotIn('proto_init', off)                                     # default random: path unchanged
        _, _, on = train_proto.parse_cli(['--dataset', 'MUTAG', '--proto_init', 'data'])
        self.assertEqual(on['proto_init'], 'data')
        for model in ('node_mask_push', 'graph', 'node_mask_push_qat'):
            self.assertEqual(hps.MODELS[model]['space']['proto_init']['choices'], ['random', 'data'])
        self.assertNotIn('proto_init', hps.MODELS['classic']['space'])


if __name__ == '__main__':
    unittest.main()
