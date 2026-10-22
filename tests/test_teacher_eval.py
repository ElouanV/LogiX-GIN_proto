"""--teacher_eval (train_proto.py, train_logic.py): the frozen teacher distills in eval mode,
so the layer inputs the student is trained on stay binary; upstream (train mode) lets the
teacher's dropout scale them to 1 / (1 - 0.15).

    ~/miniconda3/envs/logix-gin/bin/python -m unittest tests/test_teacher_eval.py -v
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.loader import DataLoader
    import train_logic
    import train_proto
    from models.model import GIN, GINTELL
    from models_proto.model_proto import get_model
    from tests.test_val_loss import graphs
except Exception as e:                                   # torch_geometric missing etc.
    raise unittest.SkipTest(f'training modules not importable: {e}')


class RecordingTeacher(GIN):
    """A GIN that records the layer inputs (layers_x) of every forward_e."""

    def forward_e(self, *args, **kwargs):
        layers_x, layers_y = super().forward_e(*args, **kwargs)
        self.seen = getattr(self, 'seen', []) + [torch.cat([x.flatten() for x in layers_x[1:-1]])]
        return layers_x, layers_y


class TestTeacherEval(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.loader = DataLoader(graphs(16, 0), batch_size=8, shuffle=True)

    def _values(self, epoch, teacher_eval):
        teacher = RecordingTeacher(3, 2, num_layers=2, hidden_dim=8)
        epoch(teacher, teacher_eval)
        return torch.unique(torch.cat(teacher.seen)), teacher.training

    def check(self, epoch):
        values, training = self._values(epoch, teacher_eval=True)
        self.assertFalse(training)
        self.assertTrue(set(values.tolist()) <= {0.0, 1.0})
        values, training = self._values(epoch, teacher_eval=False)       # upstream
        self.assertTrue(training)
        self.assertTrue((values > 1.0).any())

    def test_proto_student(self):
        student = get_model('node', num_features=3, num_classes=2, num_layers=2, hidden_dim=8, num_prototypes=4)
        opt = torch.optim.Adam(student.parameters(), lr=0.01)
        self.check(lambda t, te: train_proto.train_epoch(t, student, self.loader, 'cpu', opt, 2, teacher_eval=te))

    def test_logic_student(self):
        student = GINTELL(3, 2, num_layers=2, hidden_dim=8)
        opt = torch.optim.Adam(student.parameters(), lr=0.01)
        self.check(lambda t, te: train_logic.train_epoch(t, student, self.loader, 'cpu', opt, 2, teacher_eval=te))

    def test_cli(self):
        _, _, args = train_proto.parse_cli(['--dataset', 'MUTAG', '--teacher_eval'])
        self.assertTrue(args['teacher_eval'])
        _, _, args = train_proto.parse_cli(['--dataset', 'MUTAG'])
        self.assertNotIn('teacher_eval', args)                  # off: results paths unchanged
        _, _, args = train_logic.parse_cli(['--dataset', 'MUTAG'])
        self.assertNotIn('teacher_eval', args)


if __name__ == '__main__':
    unittest.main()
