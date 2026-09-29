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

The trunk is shape-identical to models_proto.model.GINTELL (same state-dict keys
under `convs`), so the layer-wise distillation of train_logic.py applies to it
unchanged; the prototype layer(s) have no teacher counterpart.
"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool

from models_proto.tell import LogicalLayer
from models_proto.proto import PrototypeLayer

POOLS = {'mean': global_mean_pool, 'max': global_max_pool, 'sum': global_add_pool}


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
            self.fc = LogicalLayer(2 * self.readout_dim, num_classes, dummy_phi_in=False)
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

    def prototype_inputs(self, xs, batch):
        """Tensors each prototype layer is compared against, in proto_layers order."""
        raise NotImplementedError

    def readout(self, xs, batch):
        """xs: per-layer node states [N × h]; returns s ∈ [0,1]^{G × readout_dim}."""
        raise NotImplementedError

    # ---- shared ----
    @staticmethod
    def _pool(x, batch, ops):
        return torch.hstack([POOLS[op](x, batch) for op in ops])

    def head(self, s, discrete_output=False):
        if self.task == 'classification':
            return self.fc(torch.hstack([s, 1 - s]), discrete_output=discrete_output)
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

    Like GINTELLProtoGraph, only pooling operators whose output stays in [0,1] are
    used. sum is excluded: a summed similarity grows with graph size (measured range
    3.7-94 on Mutagenicity), so the head's negation literal 1-s leaves [0,1] entirely
    and stops being a negation.
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
    def proto_layers(self):
        return [self.proto]

    def prototype_inputs(self, xs, batch):
        return [torch.hstack(xs)]

    def readout(self, xs, batch):
        h = torch.hstack(xs)                              # [N × L·h]
        s_node = self.proto(h)                            # [N × K]
        return self._pool(s_node, batch, self.pool_ops)   # [G × 2K]


class GINTELLProtoGraph(GINTELLProtoBase):
    """Prototypes over the pooled graph embedding.

    Only pooling operators whose output stays in [0,1] are used (mean, max), since
    Hamming agreement needs inputs in [0,1]; sum is excluded by default.
    """

    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1,
                 task='classification', num_prototypes=16, pool_ops=('mean', 'max'), proto_mask=False):
        self.num_prototypes = num_prototypes
        self.pool_ops = tuple(pool_ops)
        super().__init__(num_features, num_classes, num_layers, hidden_dim, dropout, task, proto_mask)

    def _build_readout(self):
        self.proto = PrototypeLayer(len(self.pool_ops) * self.num_layers * self.hidden_dim, self.num_prototypes,
                                    **self.proto_kwargs)

    @property
    def readout_dim(self):
        return self.num_prototypes

    @property
    def proto_layers(self):
        return [self.proto]

    def prototype_inputs(self, xs, batch):
        return [self._pool(torch.hstack(xs), batch, self.pool_ops)]

    def readout(self, xs, batch):
        z = self._pool(torch.hstack(xs), batch, self.pool_ops)   # [G × 2·L·h]
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

    @property
    def readout_dim(self):
        return len(self.node_pool_ops) * self.num_prototypes + self.num_graph_prototypes

    @property
    def proto_layers(self):
        return [self.proto_node, self.proto_graph]

    def prototype_inputs(self, xs, batch):
        h = torch.hstack(xs)
        return [h, self._pool(h, batch, self.graph_pool_ops)]

    def readout(self, xs, batch):
        h = torch.hstack(xs)
        s_node = self._pool(self.proto_node(h), batch, self.node_pool_ops)       # [G × 2K_n]
        s_graph = self.proto_graph(self._pool(h, batch, self.graph_pool_ops))    # [G × K_g]
        return torch.hstack([s_node, s_graph])


MODELS = {'node': GINTELLProtoNode, 'graph': GINTELLProtoGraph, 'both': GINTELLProtoBoth}


def get_model(level, *args, **kwargs):
    """level ∈ {'node', 'graph', 'both'}; remaining arguments go to the class."""
    return MODELS[level](*args, **kwargs)
