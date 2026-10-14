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

### Why step 1 stays far below the network (diagnosis, 2026-10-13)

The bar for an interesting result is a rule model at ≥ 0.80 test accuracy on Mutagenicity,
about what the soft network reaches.

- **It underfits; it does not overfit.** The step-1 rule model reaches 0.759 / 0.754 / 0.681
  on the *training* split, about its val and test accuracy. The binarised model does not fit
  the task.
- **The head is not saturated on Mutagenicity.** The median o(1 − o) of the graded head output
  is 0.16–0.23 before and after, with no ties, so the task gradient gets through the head.
- **The gradients do not vanish** (`qat/grad_*` stay at 1–4).
- **The objective is mostly not about the task, and not about the rule model.** It has six
  per-layer distillation terms (each conv, alone and chained) and a head term trained on the
  teacher's states. The only end-to-end term is computed on the graded head output, at the
  current Quant-Noise rate.
- **Training length is not the main cause.** The val rule accuracy reaches its level by epoch
  25 (seed 0: 0.687 → 0.747) and then drifts by about +2 points over the last 100 epochs.

## Step 2: losses on the rule model and keep-start selection

- `--task_weight λ`: cross-entropy of the **rule margins** W·lit + b against the labels. The
  margins come from a forward pass at rate 1 (every bit binarised, straight-through
  gradients), at every batch and whatever the current rate. `forward(..., margin=True)` returns
  the head pre-activation, which at rate 1 equals `interp_metrics.symbolic_head`'s margins
  (tested).
- `--distill_weight μ` (`--distill_T`): self-distillation, T²·KL from the frozen starting
  student's soft margins to the rule margins (Polino 2018, ref. 6). The 0.81 soft network is the
  target, not the labels.
- `--base_weight`: weight of the run's own objective; 0 drops the distillation to the teacher.
- **Selection** (all step-2 runs): the starting model is a candidate (epoch −1), so the
  fine-tune never returns a rule model worse on val than the one it starts from. In step 1,
  seed 2 would have kept its start (val 0.726 vs 0.710 for the best QAT epoch).
- `train_acc_rules` is logged every 25 epochs (underfitting check).

Step-2 directories carry `base_weight`, `task_weight`, `distill_weight` and `select` in their
name, so they never overwrite the step-1 runs.

### Ablation plan

Same source runs (Mutagenicity NMP `node-facbe6d14f72`, seeds 0-2), rate 0 → 1 over 2/3 of
the epochs, lr of the run. Go/no-go: rule-model **test** accuracy ≥ 0.80 (mean of 3 seeds),
with the epoch selected on val.

| arm | base | task λ | distill μ | question |
|---|---|---|---|---|
| A0 | 1 | 0 | 0 | step 1 (exists, keep-start applied post hoc) |
| A1 | 1 | 1 | 0 | does a task loss on the rule model help? |
| A2 | 1 | 0 | 1 | does matching the soft student help? |
| A3 | 1 | 1 | 1 | both |
| A4 | 0 | 1 | 0 | capacity probe: can the binary model fit train ≥ 0.80 at all? |
| A5 | 0 | 1 | 1 | rule model trained only toward labels + soft student |

- Stage A: A1–A5 at 300 epochs, 15 runs of about 50 min each, 5 in parallel, so about 3 h of
  wall time.
- Stage B: the best arm at 1000 epochs, 3 runs of about 2.8 h. This tests whether length matters
  once the loss is right.
- Stage C, only if Stage B reaches 0.80: λ / μ ∈ {0.3, 3}, then 10 k-fold folds on Mutagenicity
  and PROTEINS.

If A4 cannot fit the training set beyond ~0.78, the limit is the binary model's capacity
(trunk width, 16 prototypes), not the training, and the next step is architectural.

## Result of step 2, stage A + lr check (2026-10-13)

Mutagenicity NMP, seeds 0-2, 300 epochs; rule-model accuracy (epoch selected on val).

| arm | lr | val | test | test per seed | max train |
|---|---|---|---|---|---|
| start | | | 0.727 | 0.726 / 0.774 / 0.680 | |
| A0 step 1 | 1e-3 | 0.751 | 0.743 ± 0.057 | 0.767 / 0.783 / 0.677 | |
| **A1 base + task** | 1e-3 | 0.789 | **0.772 ± 0.049** | 0.790 / 0.809 / 0.717 | 0.776 |
| A1 | 3e-4 | 0.778 | 0.759 ± 0.025 | 0.786 / 0.756 / 0.735 | 0.772 |
| A1 | 1e-4 | 0.783 | 0.753 ± 0.036 | 0.772 / 0.776 / 0.712 | 0.769 |
| A2 base + distill | 1e-3 | 0.772 | 0.764 ± 0.062 | 0.811 / 0.788 / 0.694 | 0.775 |
| A3 base + task + distill | 1e-3 | 0.768 | 0.758 ± 0.026 | 0.774 / 0.772 / 0.728 | 0.774 |
| A4 task only | 1e-3 | 0.756 | 0.725 ± 0.031 | 0.742 / 0.744 / 0.689 | 0.747 |
| A4 | 1e-4 | 0.762 | 0.747 ± 0.068 | 0.800 / 0.770 / 0.671 | 0.757 |
| A5 task + distill | 1e-3 | 0.763 | 0.709 ± 0.061 | 0.756 / 0.730 / 0.641 | 0.754 |

No arm reaches the 0.80 bar. "Max train" is the training rule accuracy (every 25 epochs): no
arm fits the training split beyond 0.78, at any lr, so neither training length nor lr is the
limit. The keep-start candidate was never selected.

### Where the ceiling is

The teacher GIN (binary node states) reaches 0.823 / 0.802 / 0.790 test on these seeds.
Scratch diagnostics, rule-model accuracy:

- **The rule head on the teacher's own binary states** (a perfect trunk): train 0.790 / 0.763 /
  0.770 and test 0.800 / 0.776 / 0.772 for the starting models. A1's own-trunk rules reach
  about this (test 0.790 / 0.809 / 0.717). QAT has removed most of the trunk's loss, and the
  head is the remaining ceiling.
- **Retraining only the head as a rule model on teacher states** (STE, lr 1e-3 / 1e-2, 200
  epochs): test 0.751-0.786, train at most 0.805.
- **A fresh binary node-prototype head** (no mask) with K prototypes on teacher states:
  - K 16: train 0.753, test 0.748.
  - K 32: train 0.764, test 0.738.
  - K 64: train 0.783, test 0.768.

  More prototypes help slowly. The binary prototype readout (Hamming similarity → mean/max
  pooling → thresholded literals) loses information that the teacher's head keeps from the
  same states (train 0.82-0.84).

## What limits the rule model (diagnostics, 2026-10-14)

Mutagenicity NMP seed runs 0-9 (the A1 runs), test split, scratch probes; a probe head is
selected on val over weight decay {0, 1e-4, 1e-3} and epochs. Correction to the section
above: `GIN.forward_e` ignores `deterministic`, so those probes read *sampled* (stochastic
Gumbel) teacher states. Here the teacher's states are deterministic (bit = logit > 0), which
is what a student thresholded at 0.5 imitates.

**1. Binarising the trunk is what costs accuracy, not binarising the head.** Quant rate 0/1
set separately on the trunk and on the head (soft = S, hard = H):

| model | trunk S, head S | trunk S, head H | trunk H, head S | trunk H, head H (rule model) |
|---|---|---|---|---|
| start | 0.820 ± 0.017 | 0.812 ± 0.014 | 0.699 ± 0.062 | 0.687 ± 0.067 |
| A1 | 0.692 ± 0.040 | 0.685 ± 0.038 | 0.761 ± 0.035 | 0.763 ± 0.031 |

The teacher is affected in the same way: 0.801 ± 0.028 on sampled states and 0.772 ± 0.022 on
deterministic states. A1's rule model (0.763) is at the deterministic teacher's level.

**2. The hard bits still carry the information; the readout literals do not.** These are
fresh heads trained on the pooled hard node bits (all 192 trunk units, no prototypes). The
literals are:

- `mm05`: [mean ≥ 0.5, max], the literal type of the current head.
- `thermo`: count ≥ t for t in {1, 2, 3, 4, 6, 8, 12}, and mean ≥ f for f in {.1, .25, .5, .75}.

`rule1` is one threshold layer (one LogicalLayer over [b, 1 − b]). `rule2` adds a hidden
layer of binarised threshold units. Test accuracy:

| head on bits of | teacher (det.) | start, hard trunk | A1, hard trunk |
|---|---|---|---|
| MLP, mean/max/log-sum (real) | 0.815 ± 0.016 | 0.810 ± 0.020 | 0.814 ± 0.013 |
| linear, mean/max/log-sum (real) | 0.806 ± 0.015 | 0.782 ± 0.018 | 0.801 ± 0.023 |
| rule1, mm05 | 0.773 ± 0.024 | 0.767 ± 0.027 | 0.772 ± 0.015 |
| **rule1, thermo** | 0.793 ± 0.023 | 0.792 ± 0.016 | **0.797 ± 0.022** |
| rule2 h32, mm05 | 0.771 ± 0.015 | 0.781 ± 0.021 | 0.775 ± 0.017 |
| **rule2 h32, thermo** | 0.799 ± 0.017 | 0.792 ± 0.022 | **0.802 ± 0.021** |
| rule2 h128, thermo | 0.802 ± 0.018 | 0.793 ± 0.021 | 0.799 ± 0.016 |

Paired against A1's rule model on A1's own bits:

- rule1 thermo: +0.035 (9/10 seeds, Wilcoxon p = 0.006).
- rule2 h32 thermo: +0.040 (10/10, p = 0.002).
- rule1 mm05: +0.009 (p = 0.38).

So with [mean ≥ 0.5, max] literals, one threshold layer is capped at about 0.77 whatever the
trunk. Count thresholds ("at least t nodes have bit j") lift a one-layer rule head to about
0.80. A second rule layer helps only with count literals. The node-prototype heads measured
above (0.74-0.77) sit at or below the mm05 cap.

Caveats:

- One split per seed (not the k-fold).
- 192 × 11 thermometer literals: explanation size is not controlled yet, and neither the
  prototypes nor the mask are in these heads.
- Selection over wd × epochs on val is mildly optimistic.

Scripts are in the session scratchpad (not kept). They should become a proper head option
if they are kept.
