"""The val loss every training loop logs (train_epoch(..., optimizer=None)) leaves training
unchanged: same weights after training with and without the val passes in between, because
the pass runs without gradient and inside a forked RNG (utils.evaluation.held_out), while
the teacher's Gumbel sampling and the input dropout do consume random numbers.

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_val_loss.py -v
"""
import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import train_baseline
    import train_logic
    import train_proto
    from models.model import GIN, GINTELL
    from models_proto.model_proto import get_model
    from sparsify_proto import notebook_sparsity_epoch
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'training modules not importable: {e}')


def graphs(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        k = 6
        src, dst = torch.randint(0, k, (10,), generator=g), torch.randint(0, k, (10,), generator=g)
        out.append(Data(x=(torch.rand(k, 3, generator=g) > 0.5).float(),
                        edge_index=torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
                        y=torch.randint(0, 2, (1,), generator=g)))
    return out


def same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


class TestValLossLeavesTrainingUnchanged(unittest.TestCase):

    def setUp(self):
        self.train = DataLoader(graphs(24, 0), batch_size=8, shuffle=True)
        self.val = DataLoader(graphs(12, 1), batch_size=8, shuffle=False)

    def _twice(self, make, step, val_step, epochs=3):
        """Train two copies from the same seed, the second with a val pass after each epoch."""
        models, losses = [], []
        for with_val in (False, True):
            torch.manual_seed(0)
            m = make()
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.01)
            for e in range(epochs):
                step(m, opt, e)
                if with_val:
                    losses.append(val_step(m, e))
            models.append(m)
        return models, losses

    def test_teacher(self):
        make = lambda: GIN(3, 2, num_layers=2, hidden_dim=8)
        (a, b), losses = self._twice(
            make, lambda m, opt, e: train_baseline.train_epoch(m, self.train, 'cpu', opt, 2),
            lambda m, e: train_baseline.train_epoch(m, self.val, 'cpu', None, 2)[0])
        self.assertTrue(same(a, b))
        self.assertTrue(all(l > 0 for l in losses))

    def _teacher(self):
        torch.manual_seed(1)
        return GIN(3, 2, num_layers=2, hidden_dim=8)

    def test_logic_student(self):
        teacher = self._teacher()
        make = lambda: GINTELL(3, 2, num_layers=2, hidden_dim=8)
        (a, b), losses = self._twice(
            make, lambda m, opt, e: train_logic.train_epoch(teacher, m, self.train, 'cpu', opt, 2, train_full=e > 0),
            lambda m, e: train_logic.train_epoch(teacher, m, self.val, 'cpu', None, 2, train_full=e > 0)[0][-1])
        self.assertTrue(same(a, b))
        self.assertTrue(all(l > 0 for l in losses))

    def test_proto_student_and_qat_extra_loss(self):
        teacher = self._teacher()
        make = lambda: get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=3)
        kw = dict(proto_div_reg=0.01, proto_ent_reg=0.01)
        (a, b), losses = self._twice(
            make, lambda m, opt, e: train_proto.train_epoch(teacher, m, self.train, 'cpu', opt, 2, train_full=e > 0, **kw),
            lambda m, e: train_proto.train_epoch(teacher, m, self.val, 'cpu', None, 2, train_full=e > 0, **kw)[0][-1])
        self.assertTrue(same(a, b))
        self.assertTrue(all(l > 0 for l in losses))

    def test_val_pass_changes_no_weight(self):
        teacher, m = self._teacher(), get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8,
                                                 num_prototypes=3)
        before = copy.deepcopy(m)
        train_proto.train_epoch(teacher, m, self.val, 'cpu', None, 2)
        self.assertTrue(same(before, m))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_layerwise_hoyer(self):
        make = lambda: get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=3)
        (a, b), losses = self._twice(
            make, lambda m, opt, e: notebook_sparsity_epoch(m, self.train, 'cpu', opt, 2, 0.1, 0.1),
            lambda m, e: notebook_sparsity_epoch(m, self.val, 'cpu', None, 2, 0.1, 0.1)[0])
        self.assertTrue(same(a, b))
        self.assertTrue(all(l > 0 for l in losses))


if __name__ == '__main__':
    unittest.main()
