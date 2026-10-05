"""LogiX-GIN with prototypes (graph-level tasks).

Three variants share the GINTELL trunk (GINConv layers whose nn is a LogicalLayer,
jumping-knowledge concatenation) and the GINTELL head. They differ only in the
readout, where extracted features are compared to learned binary prototypes and
the similarity vector s replaces the pooled embedding as input of the head:

    GINTELLProtoNode   node embeddings  [N × L·h]  → K_n prototypes → per-node similarities
                       pooled per graph with mean ‖ max                  → s ∈ [0,1]^{2·K_n}
    GINTELLProtoGraph  pooled embedding [G × 2·L·h] (mean ‖ max) → K_g prototypes
                                                                        → s ∈ [0,1]^{K_g}
    GINTELLProtoBoth   concatenation of the two                          → s ∈ [0,1]^{2·K_n + K_g}

Head, unchanged from GINTELL:
    task='classification' → fc = LogicalLayer(2·dim(s) → C) applied to [s, 1−s]
    task='regression'     → fc = nn.Linear(dim(s) → out)

Sum pooling (``pool_ops`` / ``node_pool_ops`` / ``graph_pool_ops`` containing 'sum', off
by default) is supported as in PLEX/model_proto.py:
    node level   the summed similarity counts the nodes matching a prototype. It is
                 unbounded, so the head negates only the bounded columns
                 (``bounded_mask``): fc reads [s, 1 − s_bounded], and its phi_in turns a
                 count into the literal "count ≥ t".
    graph level  the summed trunk units are counts, which a binary prototype cannot be
                 compared to by Hamming agreement. They go through a learned threshold
                 first (``phi_sum``, the Phi of the LogicalLayers), so a prototype bit on
                 that block reads "unit u holds on ≥ t_u nodes".
With mean/max only, both are no-ops and the model is the one of earlier checkpoints.

The trunk is shape-identical to models_proto.model.GINTELL (same state-dict keys
under `convs`), so the layer-wise distillation of train_logic.py applies to it
unchanged; the prototype layer(s) have no teacher counterpart.
"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool

from models_proto.tell import LogicalLayer, Phi
from models_proto.proto import PrototypeLayer

POOLS = {'mean': global_mean_pool, 'max': global_max_pool, 'sum': global_add_pool}
BOUNDED_POOLS = ('mean', 'max')        # pooled values of [0,1] inputs stay in [0,1]


class GINTELLProtoBase(nn.Module):
    """Shared trunk and head. Subclasses define the prototype readout."""

    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1, task='classification',
                 proto_mask=False):
        super().__init__()
        self.proto_kwargs = {'mask': proto_mask}          # see PrototypeLayer
        self.num_features, self.num_classes = num_features, num_classes
        self.num_layers, self.hidden_dim = num_layers, hidden_dim
        self.task = task

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = GINConv(
                nn.Sequential(
                    LogicalLayer(2 * num_features if i == 0 else 2 * hidden_dim, hidden_dim, dummy_phi_in=False)
                ),
            )
            self.convs.append(conv)
        self.dropout = nn.Dropout(dropout)

        self._build_readout()
        if task == 'classification':
            bounded = torch.tensor(self.bounded_mask, dtype=torch.bool)
            assert len(bounded) == self.readout_dim, (len(bounded), self.readout_dim)
            self.register_buffer('_bounded', bounded, persistent=False)
            self.fc = LogicalLayer(self.readout_dim + int(bounded.sum()), num_classes, dummy_phi_in=False)
        elif task == 'regression':
            self.fc = nn.Linear(self.readout_dim, num_classes)
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    # ---- provided by subclasses ----
    def _build_readout(self):
        raise NotImplementedError

    @property
    def readout_dim(self):
        raise NotImplementedError

    @property
    def proto_layers(self):
        raise NotImplementedError

    @property
    def bounded_mask(self):
        """Per readout column: True if it stays in [0,1] (and so has a negation literal)."""
        raise NotImplementedError

    def prototype_inputs(self, xs, batch):
        """Tensors each prototype layer is compared against, in proto_layers order."""
        raise NotImplementedError

    def prototype_bit_layers(self):
        """Per prototype layer, the conv layer every input bit is read from [d]."""
        raise NotImplementedError

    def readout_groups(self):
        """Prototype (numbered across proto_layers) behind every readout column [readout_dim]."""
        raise NotImplementedError

    def readout(self, xs, batch):
        """xs: per-layer node states [N × h]; returns s ∈ [0,1]^{G × readout_dim}."""
        raise NotImplementedError

    # ---- shared ----
    @staticmethod
    def _pool(x, batch, ops):
        return torch.hstack([POOLS[op](x, batch) for op in ops])

    @staticmethod
    def _bounded_for(ops, k):
        """Column mask of k columns pooled with ops, in _pool's order."""
        return [op in BOUNDED_POOLS for op in ops for _ in range(k)]

    def _make_phi_sum(self, ops):
        """Thresholds of the summed trunk units (graph-level prototypes with 'sum')."""
        return Phi(self.num_layers * self.hidden_dim) if 'sum' in ops else None

    def _pool_graph(self, h, batch, ops, phi_sum, symbolic=False):
        """Pooled trunk units in [0,1] for graph prototypes: a summed block goes through
        phi_sum, binarised (straight-through) in hard mode or for the symbolic reading."""
        blocks = []
        for op in ops:
            z = POOLS[op](h, batch)
            if op == 'sum':
                z = phi_sum(z)
                if symbolic or getattr(self, 'hard', False):
                    z = z + ((z >= 0.5).float() - z).detach()
            blocks.append(z)
        return torch.hstack(blocks)

    def _bit_layers(self, n_ops=None):
        """Conv layer of each bit of hstack(xs) (n_ops=None) or of its n_ops pooled copies."""
        layers = torch.arange(self.num_layers * self.hidden_dim) // self.hidden_dim
        return layers if n_ops is None else layers.repeat(n_ops)

    def _unit_care(self):
        """Per trunk unit (hstack(xs) layout), the care bits of every prototype and pooled
        copy reading it: [n_units × (prototypes · copies)], straight-through."""
        n = self.num_layers * self.hidden_dim
        return torch.cat([p.care.view(p.num_prototypes, -1, n).flatten(0, 1) for p in self.proto_layers]).t()

    def vocab_size(self):
        """Fraction of trunk units that at least one prototype cares about (1 unmasked)."""
        return (self._unit_care() > 0.5).any(1).float().mean()

    def vocab_penalty(self):
        """Group-sparsity surrogate of vocab_size: mean over units of sqrt(share of the
        prototype bits reading the unit that care about it). Its gradient exists from the
        all-cared start (the exact count has none there) and is smaller for a unit that is
        already widely cared about, so prototypes are pushed to share few units."""
        return torch.sqrt(self._unit_care().mean(1) + 1e-8).mean()

    def set_hard(self, hard=True):
        """Compute exactly the rules: every literal binarised, every conv unit binarised
        (straight-through gradients). The head keeps a graded output so its argmax is the
        argmax of the rule margins W.lit + b, which is how the rules predict."""
        for c in self.convs:
            c.nn[0].hard_in = c.nn[0].hard_out = hard
        self.fc.hard_in = hard
        self.hard = hard                    # binarises phi_sum too (graph-level sum)

    def set_mask_layer_cost(self, cost):
        """Weight the cared-bit penalty of every prototype layer by the conv layer each bit
        reads: cost[l] for bits from layer l (see PrototypeLayer.set_bit_cost)."""
        cost = torch.as_tensor(cost, dtype=torch.float)
        if len(cost) != self.num_layers:
            raise ValueError(f'one cost per conv layer ({self.num_layers}), got {len(cost)}')
        for p, bl in zip(self.proto_layers, self.prototype_bit_layers()):
            p.set_bit_cost(cost[bl])

    def head_input(self, s):
        """What fc reads: [s, 1 - s] over the bounded columns (all of them without sum;
        pickles from before bounded_mask have no _bounded and are all bounded)."""
        b = getattr(self, '_bounded', None)
        return torch.hstack([s, 1 - s if b is None or bool(b.all()) else 1 - s[:, b]])

    @property
    def has_unbounded_readout(self):
        """True with sum pooling at node level: the head is then not [s, 1 - s]."""
        b = getattr(self, '_bounded', None)
        return b is not None and not bool(b.all())

    def head_columns(self):
        """Per head literal: (readout column, positive?), in head_input's order."""
        R = self.readout_dim
        b = getattr(self, '_bounded', None)
        neg = range(R) if b is None else torch.nonzero(b.cpu()).flatten().tolist()
        return [(i, True) for i in range(R)] + [(i, False) for i in neg]

    def head(self, s, discrete_output=False):
        if self.task == 'classification':
            return self.fc(self.head_input(s), discrete_output=discrete_output)
        return self.fc(s)

    def forward(self, x, edge_index, batch, discrete=False, *args, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            if i != 0: self.dropout(x)
            x = conv(torch.hstack([x, 1 - x]), edge_index)
            xs.append(x)
        s = self.readout(xs, batch)
        return self.head(s)

    def forward_e(self, x, edge_index, batch, discrete=False, *args, **kwargs):
        ret_x = []
        ret_y = []
        xs = []
        for i, conv in enumerate(self.convs):
            ret_x.append(x)
            if i != 0: self.dropout(x)
            x = conv(torch.hstack([x, 1 - x]), edge_index)
            xs.append(x)
            ret_y.append(x)
        s = self.readout(xs, batch)
        ret_x.append(s)
        x = self.head(s, discrete_output=False)
        ret_y.append(x)
        return ret_x, ret_y

    def forward_from_layers(self, xs, batch):
        """Readout + head only, from per-layer node states (e.g. a frozen teacher's
        layers_y[:-1]); lets the prototypes and head train while the trunk is
        being distilled."""
        return self.head(self.readout(xs, batch))

    @torch.no_grad()
    def project_prototypes(self, loader, device=None):
        """ProtoPNet push: snap each prototype to its closest training node
        (node-level) / graph (graph-level). Returns, per prototype layer, the
        chosen row indices in loader order."""
        self.eval()
        device = device or next(self.parameters()).device
        feats = [[] for _ in self.proto_layers]
        for data in loader:
            data = data.to(device)
            x = data.x.float() if data.x is not None else torch.ones((data.num_nodes, self.num_features), device=device)
            xs = []
            for conv in self.convs:
                x = conv(torch.hstack([x, 1 - x]), data.edge_index)
                xs.append(x)
            for i, f in enumerate(self.prototype_inputs(xs, data.batch)):
                feats[i].append(f)
        return [layer.project(torch.cat(feats[i])) for i, layer in enumerate(self.proto_layers)]


class GINTELLProtoNode(GINTELLProtoBase):
    """Prototypes over node embeddings; per-graph features by pooling node similarities.

    A head literal then reads "mean / max over the nodes of the similarity to
    prototype k": max → some node resembles k, mean → average resemblance.

    By default only pooling operators whose output stays in [0,1] are used. With
    'sum', the summed similarity grows with graph size (measured range 3.7-94 on
    Mutagenicity), so it gets no negation literal (bounded_mask) and the head's phi_in
    thresholds it as a count.
    """

    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1,
                 task='classification', num_prototypes=16, pool_ops=('mean', 'max'), proto_mask=False):
        self.num_prototypes = num_prototypes
        self.pool_ops = tuple(pool_ops)
        super().__init__(num_features, num_classes, num_layers, hidden_dim, dropout, task, proto_mask)

    def _build_readout(self):
        self.proto = PrototypeLayer(self.num_layers * self.hidden_dim, self.num_prototypes, **self.proto_kwargs)

    @property
    def readout_dim(self):
        return len(self.pool_ops) * self.num_prototypes

    @property
    def bounded_mask(self):
        return self._bounded_for(self.pool_ops, self.num_prototypes)

    @property
    def proto_layers(self):
        return [self.proto]

    def prototype_inputs(self, xs, batch):
        return [torch.hstack(xs)]

    def prototype_bit_layers(self):
        return [self._bit_layers()]

    def readout_groups(self):
        return torch.arange(self.readout_dim) % self.num_prototypes

    def readout(self, xs, batch):
        h = torch.hstack(xs)                              # [N × L·h]
        s_node = self.proto(h)                            # [N × K]
        return self._pool(s_node, batch, self.pool_ops)   # [G × 2K]


class GINTELLProtoGraph(GINTELLProtoBase):
    """Prototypes over the pooled graph embedding.

    Hamming agreement needs inputs in [0,1], so a 'sum' block (off by default) is
    thresholded by phi_sum before the prototypes see it.
    """

    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1,
                 task='classification', num_prototypes=16, pool_ops=('mean', 'max'), proto_mask=False):
        self.num_prototypes = num_prototypes
        self.pool_ops = tuple(pool_ops)
        super().__init__(num_features, num_classes, num_layers, hidden_dim, dropout, task, proto_mask)

    def _build_readout(self):
        self.proto = PrototypeLayer(len(self.pool_ops) * self.num_layers * self.hidden_dim, self.num_prototypes,
                                    **self.proto_kwargs)
        self.phi_sum = self._make_phi_sum(self.pool_ops)

    @property
    def readout_dim(self):
        return self.num_prototypes

    @property
    def bounded_mask(self):
        return [True] * self.num_prototypes

    def graph_input(self, h, batch, symbolic=False):
        return self._pool_graph(h, batch, self.pool_ops, getattr(self, 'phi_sum', None), symbolic)

    @property
    def proto_layers(self):
        return [self.proto]

    def prototype_inputs(self, xs, batch):
        return [self.graph_input(torch.hstack(xs), batch)]

    def prototype_bit_layers(self):
        return [self._bit_layers(len(self.pool_ops))]

    def readout_groups(self):
        return torch.arange(self.num_prototypes)

    def readout(self, xs, batch):
        z = self.graph_input(torch.hstack(xs), batch)            # [G × 2·L·h]
        return self.proto(z)                                     # [G × K]


class GINTELLProtoBoth(GINTELLProtoBase):
    """Node-level pooled similarities concatenated with graph-level similarities."""

    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1,
                 task='classification', num_prototypes=16, num_graph_prototypes=None,
                 node_pool_ops=('mean', 'max'), graph_pool_ops=('mean', 'max'), proto_mask=False):
        self.num_prototypes = num_prototypes
        self.num_graph_prototypes = num_graph_prototypes or num_prototypes
        self.node_pool_ops, self.graph_pool_ops = tuple(node_pool_ops), tuple(graph_pool_ops)
        super().__init__(num_features, num_classes, num_layers, hidden_dim, dropout, task, proto_mask)

    def _build_readout(self):
        self.proto_node = PrototypeLayer(self.num_layers * self.hidden_dim, self.num_prototypes, **self.proto_kwargs)
        self.proto_graph = PrototypeLayer(len(self.graph_pool_ops) * self.num_layers * self.hidden_dim,
                                          self.num_graph_prototypes, **self.proto_kwargs)
        self.phi_sum = self._make_phi_sum(self.graph_pool_ops)

    @property
    def readout_dim(self):
        return len(self.node_pool_ops) * self.num_prototypes + self.num_graph_prototypes

    @property
    def bounded_mask(self):
        return self._bounded_for(self.node_pool_ops, self.num_prototypes) + [True] * self.num_graph_prototypes

    def graph_input(self, h, batch, symbolic=False):
        return self._pool_graph(h, batch, self.graph_pool_ops, getattr(self, 'phi_sum', None), symbolic)

    @property
    def proto_layers(self):
        return [self.proto_node, self.proto_graph]

    def prototype_inputs(self, xs, batch):
        h = torch.hstack(xs)
        return [h, self.graph_input(h, batch)]

    def prototype_bit_layers(self):
        return [self._bit_layers(), self._bit_layers(len(self.graph_pool_ops))]

    def readout_groups(self):
        node = torch.arange(len(self.node_pool_ops) * self.num_prototypes) % self.num_prototypes
        return torch.cat([node, self.num_prototypes + torch.arange(self.num_graph_prototypes)])

    def readout(self, xs, batch):
        h = torch.hstack(xs)
        s_node = self._pool(self.proto_node(h), batch, self.node_pool_ops)       # [G × 2K_n]
        s_graph = self.proto_graph(self.graph_input(h, batch))                   # [G × K_g]
        return torch.hstack([s_node, s_graph])


MODELS = {'node': GINTELLProtoNode, 'graph': GINTELLProtoGraph, 'both': GINTELLProtoBoth}


def get_model(level, *args, **kwargs):
    """level ∈ {'node', 'graph', 'both'}; remaining arguments go to the class."""
    return MODELS[level](*args, **kwargs)
