"""LogiX-GIN with node prototypes whose head reads the prototype *rule*, not a similarity
degree (docs/INTERPRETABILITY_ISSUES.md, issue 5).

Notation, for a graph G with nodes V and K masked node prototypes (models_proto/proto.py):

    l_v ∈ {0,1}^d      the trunk bits of node v (all conv layers, hstack(xs))
    p_k, c_k           prototype k's pattern and care mask, d_k = Σ_j c_kj cared bits
    m_k(v)             = Σ_j c_kj · [l_vj ≠ p_kj], node v's mismatches with prototype k
    P_k(v)             = ∧_{j: c_kj = 1} (l_vj = p_kj), the prototype rule (m_k(v) = 0)
    s_k(v)             = exp(−m_k(v) / T) ∈ (0, 1], the similarity (T annealed to 1)

GINTELLProtoNode pools s_k(v) (mean, max). A mean literal mixes how many nodes match with
how well each one does, so it is not a statement about P_k. This model pools a crisp
predicate instead:

    q_k(v) = [ s_k(v) ≥ τ_k ] = [ m_k(v) ≤ μ_k ],  μ_k = −T ln τ_k
           "node v satisfies P_k with at most μ_k of its d_k conditions violated"
           (τ_k learned per prototype: tell.RampPhi on s, binarised like every literal)
    n_k(G) = Σ_v q_k(v)          ∈ {0, …, |V|}   how many nodes satisfy it (count)
    r_k(G) = n_k(G) / |V|        ∈ [0, 1]        which share of the nodes do (share)

The readout is s = [n_1..n_K | r_1..r_K] (pool_ops = ('count', 'share')), and the head
reads both directions of each, all through tell.RampPhi (monotone, exactly x ≥ t):

    head input [n, r, −n, 1 − r]   literals  n_k ≥ a,  r_k ≥ t,  n_k ≤ b,  r_k ≤ 1 − t'

Max pooling is the count literal n_k ≥ 1. A positive literal cites exactly the nodes with
q_k(v) = 1; a negative one ("no node satisfies P_k") cites none. Needs masked prototypes
(proto_mask), so that the tolerance reads as mismatches. The trunk is GINTELLProtoNode's.
"""
import math

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool

from models_proto.model_proto import GINTELLProtoNode
from models_proto.tell import RampPhi

COUNT_OPS = ('count', 'share')


class GINTELLProtoCount(GINTELLProtoNode):
    """Node prototypes; per graph, the count and share of nodes satisfying each prototype rule."""

    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1,
                 task='classification', num_prototypes=16, proto_mask=True, predicate_slope=4.0, head_slope=4.0):
        if not proto_mask:
            raise ValueError('GINTELLProtoCount needs masked prototypes (proto_mask): q_k reads mismatches')
        self.predicate_slope, self.head_slope = predicate_slope, head_slope
        super().__init__(num_features, num_classes, num_layers, hidden_dim, dropout, task, num_prototypes,
                         pool_ops=COUNT_OPS, proto_mask=proto_mask)
        if task == 'classification':
            self.bound_head()

    def _build_readout(self):
        super()._build_readout()
        self.predicate_phi = RampPhi(self.num_prototypes, bounded=True, slope=self.predicate_slope)

    @property
    def bounded_mask(self):
        K = self.num_prototypes
        return [False] * K + [True] * K                       # counts, then shares

    def predicate(self, h, symbolic=False):
        """q [N × K]: node v satisfies prototype k's rule up to its tolerance. Soft while
        training, binarised in hard mode / for the symbolic reading, Quant-Noise under QAT."""
        s = self.proto(h) if not symbolic else self.proto.similarity(h)
        return self._as_bits(self.predicate_phi(s), symbolic)

    @staticmethod
    def pool_predicate(q, batch):
        return torch.hstack([global_add_pool(q, batch), global_mean_pool(q, batch)])

    def readout(self, xs, batch, symbolic=False):
        return self.pool_predicate(self.predicate(torch.hstack(xs), symbolic), batch)    # [G × 2K]

    def tolerance(self):
        """μ_k = −T ln τ_k: the mismatches q_k tolerates at the current temperature."""
        return -self.proto.mask_temp * torch.log(self.predicate_phi.t)

    # ---- head: both directions of every column, monotone threshold literals ----
    def head_input(self, s):
        K = self.num_prototypes
        n, r = s[:, :K], s[:, K:]
        return torch.hstack([n, r, -n, 1 - r])

    def head_columns(self):
        R = self.readout_dim
        return [(i, True) for i in range(R)] + [(i, False) for i in range(R)]

    @property
    def has_unbounded_readout(self):
        return True

    def bound_head(self, max_slope=math.pi):
        """The head's literals are tell.RampPhi thresholds; count thresholds sit between two
        integers once centred (offset +-1/2: n >= med + 1/2, n <= med + 1/2)."""
        if self.task != 'classification':
            raise ValueError('threshold head literals: classification heads only')
        K = self.num_prototypes
        bounded = [False] * K + [True] * K + [False] * K + [True] * K
        offset = [0.5] * K + [0.0] * K + [-0.5] * K + [0.0] * K
        self.fc.phi_in = RampPhi(4 * K, bounded=bounded, offset=offset, slope=self.head_slope).to(self.fc.b.device)
