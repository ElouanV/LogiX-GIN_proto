# Quantization-aware training (QAT) of prototype LogiX-GIN

Branch `feat/qat`. Entry point `qat_proto.py`. Tests `tests/test_qat.py`.

## Problem

A trained LogiX-GIN student is *read* as rules by binarising at 0.5:

- every literal (the `phi_in` threshold of every conv unit and of the head);
- every conv unit output.

See `interp_metrics.trunk_states` and `symbolic_head`. The network itself computes with soft
values, and fidelity is the agreement between the two.

### Where fidelity is lost

On the `sum_ablation` final folds (2026-10-12), the trunk accounts for most of the
infidelity. The head accounts for only 0-5 points.

### What we tried before

On Mutagenicity NMP (seeds 0-2, test split), every earlier way of closing the gap cost about
11 points of accuracy.

| method | network acc | fidelity | rule-model acc |
|---|---|---|---|
| none (trained model) | 0.812 | 0.80 | 0.727 |
| hardening during training (sharpen, β 30) | 0.769 | 0.88 | 0.716 |
| hardening fine-tune (harden_proto.py, β 1 → 20) | 0.701 | 0.88 | 0.672 |
| hard fine-tune (all bits binarised at once, straight-through) | 0.697 | 1.00 | 0.697 |

**Why sharpening fails.** It makes every sigmoid steep, so its gradient vanishes. At β = 30,
a TELL unit only passes gradient for |z| < 0.034.

**Why binarising everything at once fails.** It changes the whole network in one step. In
`set_hard`, the binarisation also happens inside each layer, so the per-layer distillation
BCE is computed on exact 0/1 values. This is a hypothesis for why that run degraded; it has
not been tested in isolation.

## Literature

The binarisation steps are an instance of quantisation (1 bit per activation).

1. **Straight-through estimator (STE).** Bengio, Léonard, Courville (2013), *Estimating or
   propagating gradients through stochastic neurons for conditional computation*,
   [arXiv:1308.3432](https://arxiv.org/abs/1308.3432). The forward pass uses the binary
   value and the backward pass treats it as the identity. Yin et al. (ICLR 2019),
   *Understanding straight-through estimator in training activation quantized neural nets*,
   [arXiv:1903.05662](https://arxiv.org/abs/1903.05662), show which STE choices give descent
   directions.
2. **Binarized networks.** Hubara, Courbariaux, Soudry, El-Yaniv, Bengio (NeurIPS 2016),
   *Binarized Neural Networks*, [arXiv:1602.02830](https://arxiv.org/abs/1602.02830). They
   train with binary activations in the forward pass and STE in the backward pass; the loss
   stays on real-valued outputs.
3. **Quant-Noise.** Fan, Stock, Graham, Grave, Gribonval, Jégou, Joulin (ICLR 2021),
   *Training with Quantization Noise for Extreme Model Compression*,
   [arXiv:2004.07320](https://arxiv.org/abs/2004.07320). Only a random subset of the values
   is quantised at each forward pass, so unbiased gradients still flow through the others.
   Quantising everything with STE ("QAT") is the special case rate = 1, which they show
   degrades at extreme compression. **This is the method implemented here.**
4. **Progressive quantisation.** Zhou, Yao, Guo, Xu, Chen (ICLR 2017), *Incremental Network
   Quantization*, [arXiv:1702.03044](https://arxiv.org/abs/1702.03044). They quantise a
   growing share of the network (50 % → 75 % → 87.5 % → 100 %) and retrain the rest in
   between. This motivates ramping the Quant-Noise rate from 0 to 1.
5. **Gradient estimators that approach the step gradually.** Qin et al. (CVPR 2020),
   *IR-Net*, [arXiv:1909.10788](https://arxiv.org/abs/1909.10788), Error Decay Estimator.
   The forward pass is binary, and the backward pass uses a tanh surrogate that sharpens
   during training. This is a candidate second step: our sharpening schedule moved to the
   backward pass only.
6. **Distillation into the quantised model.** Polino, Pascanu, Alistarh (ICLR 2018), *Model
   compression via distillation and quantization*,
   [arXiv:1802.05668](https://arxiv.org/abs/1802.05668). They distil a full-precision teacher
   into the quantised student. Here the student is already distilled from the GIN teacher
   layer by layer. A candidate third step is distilling from the frozen soft student, so the
   rule model keeps its predictions.

## What `qat_proto.py` does (step 1: Quant-Noise + STE + progressive rate)

It starts from a trained run, like harden_proto.py, and trains on the same objective as the
run (`train_full` distillation + task loss + its regularisers).

- **Ramp.** At epoch e, the rate is r = rate_to · min(1, e / (ramp_frac · epochs)).
- **What is binarised.** Every bit the rules read is binarised at 0.5 with probability r, at
  every forward pass. Binarised bits pass straight-through gradients; the others stay soft
  with exact gradients.
- **Where unit outputs are binarised.** Unit outputs are binarised between layers, where the
  next layer and the readout read them. Each layer's own distillation loss stays on its soft
  output, as in the binarized-network training of ref. 2.
- **Model selection.** At rate = rate_to, the kept epoch is the one with the best validation
  accuracy *of the rule model* (all bits binary, predictions from the rule margins W·lit + b).
  With rate_to = 1, the saved network computes exactly its rules.

### Code changed (all marked `QAT` in comments; off by default, existing models and pickles unchanged)

| file | change |
|---|---|
| `models_proto/tell.py` | `quant_noise(p, rate, training)`; `LogicalLayer.forward` applies it to the literals (`quant_rate` attribute) |
| `models_proto/model_proto.py` | `set_quant_noise(rate)`, `quant_out(x)`; `forward` binarises conv outputs between layers; `_pool_graph` binarises phi_sum |
| `train_proto.py` | `train_epoch(..., quant=None)`: the chained input of the next layer goes through `quant` |
| `harden_proto.py` | `load_run` / `write_run` extracted (behaviour unchanged), shared with qat_proto.py |
| `qat_proto.py` | the fine-tune, its diagnostics and logging |

### MLflow (experiment `qat/<dataset>`)

Logged per epoch:

| metric | meaning |
|---|---|
| `qat/rate` | current binarisation rate |
| `qat/train_loss` | training loss |
| `qat/val_acc_soft` | network accuracy with every bit soft |
| `qat/val_acc_rules` | rule-model accuracy |
| `qat/val_fidelity` | agreement of the soft network with the rule model |
| `qat/val_flip_L<l>` | share of conv-l unit states that differ between the two |
| `qat/grad_L<l>`, `qat/grad_head` | gradient norm of the weight logits (shows vanishing gradients) |
| `qat/best_val_acc_rules` | best rule-model val accuracy so far |

`before/*` and `after/*` hold the same quantities for val and test, for the soft network and
the rule model. `qat.json` in the output directory keeps the whole history.

### How to read a result

The number to beat is the **rule-model test accuracy**: 0.727 for the trained model, 0.697
for the hard fine-tune, both at fidelity 1 once the network is the rule model.
`interp_metrics.py --run_path <out_dir>` gives the full report (fidelity, explanation size,
NO2/NH2 precision).

**Evaluate QAT runs with this branch's code.** The checkpoints are full pickles. If
`interp_metrics.py` is run from a checkout without the QAT hooks, the network runs soft and
the reported network accuracy and fidelity are wrong.

## Result of step 1 (2026-10-12)

Mutagenicity NMP, seeds 0-2, test split, 300 epochs, rate 0 → 1.

| | network acc | fidelity | rule-model acc | NO2/NH2 precision |
|---|---|---|---|---|
| trained model | 0.812 | 0.80 | 0.727 | 0.71 |
| hard fine-tune | 0.697 | 1.00 | 0.697 | |
| QAT (Quant-Noise) | 0.742 ± 0.057 | 1.00 | 0.742 ± 0.057 | 0.57 |

Per seed, rule-model accuracy goes from 0.726 / 0.774 / 0.680 to 0.767 / 0.783 / 0.677.
