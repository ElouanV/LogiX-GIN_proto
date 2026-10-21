import math

from torch import nn
import torch
import torch.nn.functional as F
def logit(x, tau=10):
    x = x
    return torch.log((x / (1 - x + 1e-8))+1e-8)/tau

def sigmoid(x, tau=10):
    return 1/(1+torch.exp(-tau*x))


def quant_noise(p, rate, training=True):
    """QAT by Quant-Noise (Fan et al., ICLR 2021, arXiv:2004.07320) for values in [0, 1] read
    as bits: in training, each entry is binarised at 0.5 with probability ``rate`` (forward:
    the bit, backward: straight-through, Bengio et al. 2013) and the others stay soft, so
    exact gradients still flow through them. rate = 1 (or eval mode with rate = 1): every
    entry binarised, the layer computes exactly its rules; eval with rate < 1: soft."""
    if rate <= 0:
        return p
    hard = p + ((p >= 0.5).float() - p).detach()
    if rate >= 1:
        return hard
    if not training:
        return p
    return torch.where(torch.rand_like(p) < rate, hard, p)


def sharpen(p, beta):
    """p in [0, 1] pushed towards {0, 1} around 0.5: sigmoid(beta * logit(p)). beta = 1 is
    the identity, beta -> inf the step p >= 0.5 the rules read; gradients stay real."""
    return torch.sigmoid(beta * torch.logit(p, eps=1e-6))


def hard_sigmoid(logits, threshold=0.5):
    y_soft = logits.sigmoid()
    indices = (y_soft > threshold).nonzero(as_tuple=True)
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
    y_hard[indices[0], indices[1]] = 1.0
    ret = y_hard - y_soft.detach() + y_soft
    return ret


from torch import Tensor




def step(x, tau=0):
    def fourier(inp, tau): 
        r = 0
        for i in range(tau+1):
            r+=torch.sin((2*i+1)*inp)/(2*i+1)
        return r
    return torch.clamp(((fourier(x, tau)/fourier(torch.tensor(torch.pi/2), tau))+1)/2, 0, 1)


def sin(x, tau=10):
    return torch.sin(tau*x)

class Phi(nn.Module):
    def __init__(self, features, calculate_entropy=True):
        super().__init__()
        self.calculate_entropy = calculate_entropy
        self.w_ = nn.Parameter(torch.Tensor(features))
        self.b = nn.Parameter(torch.Tensor(features))
        self.reset_parameters()
        self.tau = None
        self.entropy = None

    @property
    def w(self):
        return torch.exp(self.w_)
    
    @property
    def t(self):
        return -self.b/self.w

    def reset_parameters(self):
        nn.init.constant_(self.w_, 1)
        # nn.init.uniform_(self.w_, 0.1, 0.9)
        with torch.no_grad():
            self.w_.copy_(torch.log(self.w_+1e-8))
        
        nn.init.constant_(self.b, 0)
        # nn.init.uniform_(self.b, 0.1, 0.9)
        # nn.init.uniform_(self.b, -0.9, -0.1)

    def forward(self, x):
        # output = sigmoid(self.w*x+self.b)
        output = step(self.w*x+self.b)
        if self.tau is not None:
            # output = sigmoid(self.w*x+self.b, tau=self.tau)
            output = step(self.w*x+self.b, tau=self.tau)
        if self.calculate_entropy:
            self.entropy = -(output*torch.log(output+1e-8) + (1-output)*torch.log(1-output + 1e-8)).mean()
        return output
    
class BoundedPhi(Phi):
    """Head literals as thresholds (train_proto.py --head_phi bounded): step(w (x - t)) with
    a bounded slope w = max_slope * sigmoid(w_) and a threshold t = sigmoid(t_) in (0, 1).

    Phi reads a literal as true where (w x + b) mod 2 pi is in [0, pi]
    (min_covers.literal_true_intervals): with w > pi a literal on [0, 1] can be a band or a
    union of intervals, and at Phi's init (w = 1, b = 0) it is true on all of [0, 1], a
    hidden bias. Here w <= pi and 0 < t < 1, so z = w (x - t) stays in [-pi, pi]: the
    literal is exactly x >= t (its negation, read on 1 - x, x <= 1 - t), never a band, and
    never true on every input. The trunk keeps Phi, whose interval unions are wanted.
    w, b and t keep Phi's meaning, so every reader of phi_in.w / .b works unchanged."""

    def __init__(self, features, max_slope=math.pi, slope=1.0, threshold=0.5, calculate_entropy=True):
        nn.Module.__init__(self)
        if not 0 < slope < max_slope:
            raise ValueError(f'slope {slope} not in (0, max_slope={max_slope})')
        self.calculate_entropy = calculate_entropy
        self.max_slope, self.slope_init, self.threshold_init = float(max_slope), float(slope), float(threshold)
        self.w_ = nn.Parameter(torch.empty(features))
        self.t_ = nn.Parameter(torch.empty(features))
        self.reset_parameters()
        self.tau = None
        self.entropy = None

    @property
    def w(self):
        return self.max_slope * torch.sigmoid(self.w_)

    @property
    def t(self):
        return torch.sigmoid(self.t_)

    @property
    def b(self):
        return -self.w * self.t

    def reset_parameters(self):
        with torch.no_grad():
            self.w_.fill_(math.log(self.slope_init / (self.max_slope - self.slope_init)))
            self.t_.fill_(math.log(self.threshold_init / (1 - self.threshold_init)))

    @torch.no_grad()
    def set_thresholds(self, t, eps=0.02):
        """Thresholds set to t (clamped to [eps, 1 - eps], so t_ stays finite)."""
        self.t_.copy_(torch.logit(torch.as_tensor(t, dtype=self.t_.dtype, device=self.t_.device).clamp(eps, 1 - eps)))


class DummyPhi(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.w = torch.ones(features)
        self.b = -torch.ones(features)*0.5
        self.entropy = None
    @property
    def t(self):
        return -self.b/self.w

    def forward(self, x):
        return x
    
class LogicalLayer(nn.Module):
    def __init__(self, in_features, out_features, dummy_phi_in=False, use_weight_sigma=True, use_weight_exp=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.use_weight_sigma = use_weight_sigma
        self.use_weight_exp = use_weight_exp
        #print(dummy_phi_in)
        if dummy_phi_in:
            self.phi_in = DummyPhi(in_features)
        else:
            self.phi_in = Phi(in_features)
        #print(self.phi_in)
        if use_weight_sigma:
            self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        else: 
            self.weight_sigma = torch.ones((out_features, in_features)).float()*0.9999 # sigma(logit(0.0.9999)) = 0.9999
        
        if use_weight_exp and use_weight_sigma:
            self.weight_exp = nn.Parameter(torch.Tensor(out_features, 1))
        elif use_weight_exp and not use_weight_sigma:
            self.weight_exp = nn.Parameter(torch.Tensor(out_features, in_features))
        else: 
            self.weight_exp = torch.zeros((out_features, 1)).float() # e^0 = 1
                
        self.b = nn.Parameter(torch.Tensor(out_features))
        
        self.prune_ = nn.Parameter(torch.ones((out_features, in_features))) 
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.constant_(self.weight, 0)
        # nn.init.uniform_(self.b, 0.1, 0.9)
        nn.init.uniform_(self.b, -0.9, -0.1)
        if self.use_weight_sigma:
            nn.init.uniform_(self.weight_sigma, 0.1, 0.9)
            with torch.no_grad():
                self.weight_sigma.copy_(torch.logit(self.weight_sigma))
        if self.use_weight_exp:
            nn.init.uniform_(self.weight_exp, 0.1, 0.9)
            with torch.no_grad():
                self.weight_exp.copy_(torch.log(self.weight_exp+1e-8))

        self.set_prune(torch.ones((self.out_features, self.in_features)))

    @property
    def prune(self):
        self.prune_.requires_grad = False
        return self.prune_

    def set_prune(self, prune):
        with torch.no_grad():
            self.prune_.copy_(prune)
    
    @property
    def weight(self):
        ws = self.weight_sigma
        we = self.weight_exp
        w = sigmoid(ws)*torch.exp(we)
        w = w * self.prune
        return w

    @property
    def weight_s(self):
        ws = self.weight_sigma
        w = sigmoid(ws)
        w = w * self.prune
        return w

    @property
    def weight_e(self):
        we = self.weight_exp
        w = torch.exp(we)
        w = w * self.prune
        return w
        

    @staticmethod
    @torch.no_grad()
    def find_logic_rules(w, t_in, t_out, max_rule_len=float('inf'), max_rules=float('inf')):
        w = w.clone()
        t_in = t_in.clone()
        t_out = t_out.clone()
        t_out = t_out.item()
        ordering_scores = w
        # ordering_scores = x_train[(x_train>t_in).float()@w > t_out, :].sum(0)
        sorted_idxs = torch.argsort(ordering_scores, 0, descending=True)
        t_out -= w[t_in < 0].sum()
        # mask = (t_in >= 0) & (t_in <= 1) & (w > 0)
        mask = (t_in >= 0) & (w > 0)
        

        total_result = set()
        

        def find_logic_rules_recursive(index, current_sum):
            if len(result) > max_rules:
                return
            
            if len(current_combination) > max_rule_len:
                return
            
            if current_sum >= t_out:
                c = idxs_to_visit[current_combination].cpu().detach().tolist()
                c = tuple(sorted(c))
                result.add(c)
                # print(current_combination, 'rules', len(result))
                return


            for i in range(index, idxs_to_visit.shape[0]):
                current_combination.append(i)
                find_logic_rules_recursive(i + 1, current_sum + w[idxs_to_visit[i]])
                current_combination.pop()

        idxs_to_visit = sorted_idxs[mask[sorted_idxs]]
        current_combination = []
        result = set()
        find_logic_rules_recursive(0, 0)
        return result

    def extract_rules(self):
        ws = self.weight
        t_in = self.phi_in.t
        t_out = -self.b

        rules = []
        for i in range(self.out_features):
            w = ws[i].to('cpu')
            ti = t_in.to('cpu')
            to = t_out[i].to('cpu')
            rules.append(self.find_logic_rules(w, ti, to))

        return rules

    
    def forward(self, x, discrete_output=False, margin=False):
        # x = self.phi_in(torch.hstack([x, 1-x]))
        x = self.phi_in(x)
        # sharpness / sharpness_out (set by GINTELLProtoBase.set_sharpness): gradual
        # hardening of the literals (sharpen) and of the output (a steeper sigmoid); 1 = off
        beta_in, beta_out = getattr(self, 'sharpness', 1.0), getattr(self, 'sharpness_out', 1.0)
        if beta_in != 1.0:
            x = sharpen(x, beta_in)
        # hard_in / hard_out (set by GINTELLProtoBase.set_hard): binarise the literals / the
        # output at 0.5 in the forward pass, straight-through in the backward pass, so the
        # layer computes exactly its rules (pickles from before have neither attribute)
        if getattr(self, 'hard_in', False):
            x = x + ((x >= 0.5).float() - x).detach()
        # QAT (GINTELLProtoBase.set_quant_noise): a random share quant_rate of the literals
        # binarised; the unit outputs are binarised by the caller, between layers, so the
        # per-layer distillation loss stays on the soft output (pickles: no attribute = off)
        x = quant_noise(x, getattr(self, 'quant_rate', 0.0), self.training)
        self.max_in, _ = x.max(0)
        reg_loss = 0
        entropy_loss = 0
        if self.use_weight_sigma:
            reg_loss += torch.clamp(self.weight_s, min=1e-5).sum(-1).mean()
        else:
            reg_loss += torch.clamp(self.weight, min=1e-5).sum(-1).mean()
        if self.phi_in.entropy is not None:
            entropy_loss += self.phi_in.entropy
        # print('b', reg_loss, entropy_loss)
        self.reg_loss = reg_loss
        
        w = self.weight
        if margin:          # QAT (qat_proto.py): the pre-activation W.lit + b, the rule margins at rate 1
            return x @ w.t() + self.b
        z = x @ w.t() + self.b
        o = sigmoid(z, tau=10 * beta_out)
        # logits: o = sigmoid(logits), for a loss that keeps its gradient where o rounds to
        # exactly 0 or 1 (utils.training.class_loss); None when the output is binarised
        self.logits = 10 * beta_out * z
        if discrete_output or getattr(self, 'hard_out', False):
            o = hard_sigmoid(x @ w.t() + self.b)
            self.logits = None
        
        self.entropy_loss = entropy_loss + -(o*torch.log(o+1e-8) + (1-o)*torch.log(1-o + 1e-8)).mean()
        return o