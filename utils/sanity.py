"""Model-randomisation sanity check of the explanations (Adebayo et al., "Sanity checks for
saliency maps", NeurIPS 2018), cascading from the head down to conv layer 0.

An explanation method passes when its explanations change as the model's learned weights
are destroyed. Ours are read off the weights exactly, so they change by construction; the
test is there for the *metrics*: the ground-truth precision and the explanation size of a
model whose learned structure is gone are the baseline the trained model must beat (a GT
precision a scrambled model also reaches measures the dataset, e.g. how many nodes lie near
an NO2 group, not what the model learned).

Stages, top-down: head (fc, and phi_sum when the graph-level sum is thresholded),
prototypes, conv layer L-1, ..., conv layer 0. Two orders, as in the paper:

    cascade       cumulative: each stage randomised on top of the ones above it
    independent   one stage at a time, every other stage trained. Needed here: on the
                  trained NMP / QAT NMP Mutagenicity students a permuted head alone already
                  predicts one class for every graph, so the cascade's lower stages cite
                  nothing and say nothing more

Two ways to randomise:

    permute   each trained parameter tensor's entries shuffled (default): same values,
              same scale and sparsity (masks, pruning), learned assignment destroyed
    init      the modules' own reset_parameters, as in the original test. Phi resets to
              w = 1, b = 0, which makes every literal constant true on [0, 1], so a
              re-initialised head cites nothing: an artefact of the init, not a finding

Per stage, against the trained model's records (interp_metrics(..., records=True)):

    pred_agreement     share of graphs where the rule model predicts the same class
    cites_nodes        share of graphs whose evidence cites at least one node
    node_jaccard       mean Jaccard of the cited node sets, over graphs where both cite one
    expl_identical     share of graphs with the same cited node set (empty included)

next to the stage's own interp metrics (logic_acc, expl_*_evidence, gt_*_evidence).
"""
import copy

import numpy as np
import torch

from interp_metrics import interp_metrics

MODES = ('permute', 'init')
ORDERS = ('cascade', 'independent')


def cascade(model):
    """[(stage name, [modules])] from the head down to conv layer 0."""
    head = [model.fc] + ([model.phi_sum] if getattr(model, 'phi_sum', None) is not None else [])
    stages = [('head', head)]
    if hasattr(model, 'proto_layers'):
        stages.append(('prototypes', list(model.proto_layers)))
    stages += [(f'conv{l}', [model.convs[l]]) for l in reversed(range(len(model.convs)))]
    return stages


@torch.no_grad()
def randomise(modules, mode, gen):
    """Randomise the parameters of `modules` in place (see the module docstring)."""
    if mode == 'permute':
        for m in modules:
            for p in m.parameters():
                flat = p.view(-1)
                flat.copy_(flat[torch.randperm(flat.numel(), generator=gen)])
    elif mode == 'init':
        torch.manual_seed(int(torch.randint(2 ** 31, (1,), generator=gen)))
        for m in modules:
            for sub in m.modules():
                if hasattr(sub, 'reset_parameters'):
                    sub.reset_parameters()
    else:
        raise ValueError(f'mode {mode!r} not in {MODES}')


def compare(ref, recs):
    """Agreement of per-graph records `recs` with the trained model's `ref` (same graphs)."""
    pred = np.mean([a['pred'] == b['pred'] for a, b in zip(ref, recs)])
    if ref and ref[0]['nodes'] is None:                    # graph prototypes cite no node
        return {'pred_agreement': float(pred)}
    sets = [(set(a['nodes']), set(b['nodes'])) for a, b in zip(ref, recs)]
    both = [len(s & t) / len(s | t) for s, t in sets if s and t]
    return {'pred_agreement': float(pred),
            'cites_nodes': float(np.mean([bool(t) for _, t in sets])),
            'node_jaccard': float(np.mean(both)) if both else None,
            'expl_identical': float(np.mean([s == t for s, t in sets]))}


KEEP = ('logic_acc', 'logic_fidelity', 'head_literals_constant', 'expl_literals_evidence_mean',
        'expl_bits_evidence_mean', 'protos_cited_evidence', 'gt_precision_evidence',
        'gt_chance_evidence', 'gt_n_evidence', 'gt_n')


class RandomisationTest:
    """The cascade on one trained model and split, `draws` random draws per stage."""

    def __init__(self, model, dataset, ds_name, device='cpu'):
        self.model, self.dataset, self.ds_name, self.device = model, dataset, ds_name, device

    def evaluate(self, model):
        res, recs = interp_metrics(model, self.dataset, self.device, self.ds_name, records=True)
        return {k: res.get(k) for k in KEEP}, recs

    def run(self, mode='permute', order='cascade', draws=3, seed=0, log=print):
        if order not in ORDERS:
            raise ValueError(f'order {order!r} not in {ORDERS}')
        base, ref = self.evaluate(self.model)
        tag = {'mode': mode, 'order': order}
        rows = [{'stage': 'trained', **tag, 'draw': None, **base, **compare(ref, ref)}]
        log(f"trained: {rows[0]}")
        for draw in range(draws):
            gen = torch.Generator().manual_seed(seed * 1000 + draw)
            model = copy.deepcopy(self.model)
            for s, (name, _) in enumerate(cascade(model)):
                if order == 'independent':
                    model = copy.deepcopy(self.model)
                randomise(cascade(model)[s][1], mode, gen)
                res, recs = self.evaluate(model)
                rows.append({'stage': name, **tag, 'draw': draw, **res, **compare(ref, recs)})
                log(f"draw {draw} {name}: {rows[-1]}")
        return rows
