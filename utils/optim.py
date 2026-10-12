"""Optimizers of the student trainings: AdamW (the default, upstream) or Muon.

Muon (Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/) replaces the update
of a 2-D weight matrix by the orthogonalised (Newton-Schulz) Nesterov momentum, so every
direction of the matrix moves at the same rate. It is meant for hidden linear maps; the
other parameters (head, embeddings, vectors) stay on AdamW. Here Muon gets the conv
units' ``weight_sigma`` (the logits of the TELL weights, ``weight = sigmoid(ws) *
exp(we)``); the head, the prototype and mask logits, thresholds, ``weight_exp`` and the
literal parameters ``phi_in`` keep AdamW. This is a single-device port of the reference
``MuonWithAuxAdam`` (github.com/KellerJordan/Muon, MIT): torch 2.5 has no
``torch.optim.Muon``, and the PyPI package ``muon`` is an unrelated project.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate U V^T of G = U S V^T with the quintic Newton-Schulz iteration of the
    reference implementation (coefficients tuned for speed, not exact convergence)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """One optimizer, two kinds of param groups: ``use_muon=True`` groups get Muon
    (momentum, nesterov, ns_steps, decoupled weight decay), the others AdamW (betas, eps).
    A single object, so ReduceLROnPlateau and the lr logging work unchanged."""

    def __init__(self, param_groups):
        for g in param_groups:
            if g['use_muon']:
                g.setdefault('momentum', 0.95)
                g.setdefault('nesterov', True)
                g.setdefault('ns_steps', 5)
            else:
                g.setdefault('betas', (0.9, 0.999))
                g.setdefault('eps', 1e-8)
            g.setdefault('weight_decay', 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if g['use_muon']:
                    if not state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.lerp_(p.grad, 1 - g['momentum'])
                    upd = p.grad.lerp(buf, g['momentum']) if g['nesterov'] else buf
                    upd = zeropower_via_newtonschulz5(upd, g['ns_steps'])
                    upd *= max(1, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1 - g['lr'] * g['weight_decay'])
                    p.add_(upd, alpha=-g['lr'])
                else:
                    if not state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1
                    b1, b2 = g['betas']
                    state['exp_avg'].lerp_(p.grad, 1 - b1)
                    state['exp_avg_sq'].lerp_(p.grad.square(), 1 - b2)
                    m = state['exp_avg'] / (1 - b1 ** state['step'])
                    v = state['exp_avg_sq'] / (1 - b2 ** state['step'])
                    p.mul_(1 - g['lr'] * g['weight_decay'])
                    p.addcdiv_(m, v.sqrt().add_(g['eps']), value=-g['lr'])
        return loss


def muon_params(model):
    """The conv units' weight matrices (``convs[i].nn[0].weight_sigma``)."""
    return [c.nn[0].weight_sigma for c in model.convs]


def build_optimizer(model, name, lr, weight_decay, muon_lr=0.02):
    """``adamw``: torch AdamW on every parameter (the upstream optimizer, bit for bit).
    ``muon``: Muon at ``muon_lr`` on muon_params(model), AdamW at ``lr`` on the rest."""
    if name in (None, 'adamw'):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name != 'muon':
        raise ValueError(f'unknown optimizer {name!r}')
    hidden = {id(p) for p in muon_params(model)}
    other = [p for p in model.parameters() if id(p) not in hidden]
    return MuonWithAuxAdamW([
        dict(params=muon_params(model), use_muon=True, lr=muon_lr, weight_decay=weight_decay),
        dict(params=other, use_muon=False, lr=lr, weight_decay=weight_decay),
    ])
