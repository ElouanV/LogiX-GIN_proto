"""Binary prototype layer for LogiX-GIN with prototypes.

The layer only learns prototypes and computes similarities; it contains no logic.
Its output is a feature vector s ∈ [0,1]^K that the existing classifier/regressor
(LogicalLayer fc, or a linear layer for regression) consumes.

A prototype is a binary pattern p ∈ {0,1}^d over a (near-)binary feature vector
l ∈ [0,1]^d. The similarity is the normalised Hamming agreement

    s_k = (1/d) · Σ_j  p_kj·l_j + (1 − p_kj)·(1 − l_j)     ∈ [0, 1]

i.e. the fraction of literals on which l and p_k agree.

Optional binary mask (``mask=True``): each prototype also learns which bits it cares
about, c_k ∈ {0,1}^d (straight-through like p_k), and the similarity becomes a soft
conjunction over the cared bits only

    s_k = exp(−(1/T) · Σ_j  c_kj · [l_j ≠ p_kj])        (mismatches on cared bits)

A node matching every cared bit scores 1, each mismatch multiplies the score by e^{-1/T}.
At T = 1 a prototype reads as a short rule "L0u3 ∧ ¬L1u12 ∧ L2u5". The trainer anneals
T (``mask_temp``) from ~d/4, where the score behaves like the Hamming agreement and
every prototype gets gradient, down to 1, and penalises ``mask_size`` (fraction of
cared bits) to keep few bits per prototype. ``set_bit_cost`` weights that penalty per input bit, e.g. to
make bits read from conv layer 0 (whose units decode to short rules) cheaper to care about.
"""
import torch
from torch import nn

from models_proto.tell import hard_sigmoid


def binary_entropy(p, eps=1e-8):
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))


class PrototypeLayer(nn.Module):
    """K binary prototypes compared to inputs in [0,1]^d by Hamming agreement.

    Args:
        in_features:    d
        num_prototypes: K

    Attributes set at each forward, for optional regularisation:
        reg_loss:      mean pairwise agreement between (soft) prototypes; minimise to
                       keep prototypes diverse
        proto_entropy: mean binary entropy of the soft prototypes; minimise to make
                       them decisive
    """

    def __init__(self, in_features, num_prototypes, mask=False, mask_init=2.0):
        super().__init__()
        self.in_features = in_features
        self.num_prototypes = num_prototypes
        self.proto_logits = nn.Parameter(torch.Tensor(num_prototypes, in_features))
        self.mask = mask
        self.mask_temp = 1.0
        if mask:                                  # starts with every bit cared
            self.mask_logits = nn.Parameter(torch.full((num_prototypes, in_features), float(mask_init)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.proto_logits, -1.0, 1.0)

    @property
    def masked(self):
        return getattr(self, 'mask', False)       # models pickled before the option existed

    @property
    def care(self):
        """Binary care mask {0,1}^{K×d} (all ones without the option), straight-through."""
        if not self.masked:
            return torch.ones_like(self.proto_logits)
        return hard_sigmoid(self.mask_logits)

    @property
    def prototypes_soft(self):
        return torch.sigmoid(self.proto_logits)

    @property
    def prototypes(self):
        """Binary prototypes {0,1}^{K×d}, straight-through in autograd."""
        return hard_sigmoid(self.proto_logits)

    def similarity(self, x):
        """x: [n × d] in [0,1]. Returns [n × K] normalised agreement in [0,1]."""
        p = self.prototypes
        if self.masked:
            c = self.care
            mismatch = x @ (c * (1 - p)).t() + (1 - x) @ (c * p).t()
            return torch.exp(-mismatch / self.mask_temp)
        agree = x @ p.t() + (1 - x) @ (1 - p).t()
        return agree / self.in_features

    def forward(self, x):
        s = self.similarity(x)

        ps = self.prototypes_soft
        pair = (ps @ ps.t() + (1 - ps) @ (1 - ps).t()) / self.in_features
        off = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=pair.device)
        self.reg_loss = pair[off].mean() if self.num_prototypes > 1 else pair.new_zeros(())
        self.proto_entropy = binary_entropy(ps).mean()
        if self.masked:
            cost = getattr(self, 'bit_cost', None)          # pickles from before set_bit_cost
            self.mask_size = (self.care * cost).mean() if cost is not None else self.care.mean()
        else:
            self.mask_size = ps.new_ones(())
        return s

    def set_bit_cost(self, cost):
        """Per-input-bit weight [d] of the cared-bit penalty mask_size (default 1 everywhere)."""
        self.register_buffer('bit_cost', torch.as_tensor(cost, dtype=torch.float, device=self.proto_logits.device))

    @torch.no_grad()
    def project(self, x, magnitude=3.0):
        """ProtoPNet push: snap every prototype to the input row it agrees with most.

        x: [n × d] feature rows (nodes or graphs) from the training set.
        Returns the chosen row index per prototype so it can be shown.
        """
        l = (x > 0.5).float()
        idx = self.similarity(l).argmax(0)              # [K]
        self.proto_logits.copy_((2 * l[idx] - 1) * magnitude)
        return idx

    def extra_repr(self):
        return f'in_features={self.in_features}, num_prototypes={self.num_prototypes}, mask={self.masked}'
