"""Classic LogiX-GIN (models/model.py GINTELL) with a configurable graph readout.

Upstream GINTELL pools the trunk units with mean ‖ max ‖ sum and feeds the head
hstack([p, 1 - p]). The sum is an unbounded count, so its ``1 - p`` literal has no
meaning, and it adds a third of the head's inputs. ``GINTELLPool`` takes ``pool_ops``
(a subset of mean / max / sum, in that order) to measure what the sum readout buys.
With ``pool_ops=('mean', 'max', 'sum')`` it computes exactly GINTELL.

The trunk (``.convs``) is GINTELL's, so the layer-wise distillation of train_logic.py
applies unchanged. ``readout(xs, batch)`` is the hook latent_logic.py and
interp_metrics.py use for non-upstream readouts.
"""
import torch

from models.model import GINTELL
from models.tell import LogicalLayer
from models_proto.model_proto import POOLS

UPSTREAM_POOLS = ('mean', 'max', 'sum')


class GINTELLPool(GINTELL):
    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1,
                 pool_ops=UPSTREAM_POOLS):
        super().__init__(num_features, num_classes, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
        unknown = set(pool_ops) - set(UPSTREAM_POOLS)
        if unknown or not pool_ops:
            raise ValueError(f'pool_ops must be a non-empty subset of {UPSTREAM_POOLS}, got {pool_ops}')
        self.pool_ops = tuple(op for op in UPSTREAM_POOLS if op in pool_ops)
        self.fc = LogicalLayer(2 * num_layers * len(self.pool_ops) * hidden_dim, num_classes, dummy_phi_in=False)

    def readout(self, xs, batch):
        h = torch.hstack(xs)
        return torch.hstack([POOLS[op](h, batch) for op in self.pool_ops])

    def select_pooled(self, pooled):
        """Columns of an upstream mean ‖ max ‖ sum readout (e.g. the teacher's) that this
        model's readout keeps."""
        n = pooled.shape[1] // len(UPSTREAM_POOLS)
        return torch.hstack([pooled[:, UPSTREAM_POOLS.index(op) * n:(UPSTREAM_POOLS.index(op) + 1) * n]
                             for op in self.pool_ops])

    def forward(self, x, edge_index, batch, discrete=False, *args, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            if i != 0: self.dropout(x)          # upstream no-op, kept for identical behaviour
            x = conv(torch.hstack([x, 1 - x]), edge_index)
            xs.append(x)
        s = self.readout(xs, batch)
        return self.fc(torch.hstack([s, 1 - s]))
