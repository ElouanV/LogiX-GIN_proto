# The Gumbel-sigmoid teacher: how it samples, why, and what it does to the students

Audit issue 1 (2026-10-22). The GIN teacher (`models/model.py`, `GIN`) makes its node states
binary with `gumbel_sigmoid(..., hard=True)`. That function **samples**: it adds random noise
before thresholding, at training **and** at evaluation. This document explains the function, why
the noise is there, and measures what it does to the teacher's reported accuracy and to the
students distilled from it. Numbers: the k-fold teachers of every dataset (10 folds, test split,
20 draws per fold), `teacher_noise.py` (commands in §6).

## 1. Summary

- Every evaluation of the same teacher checkpoint is a new random sample. Within one fold the
  test accuracy moves by **0.8 to 3.0 points (std)** over repeated evaluations, up to **16 points
  (range) on MUTAG** and 19 on BA2Motifs. The teacher row of `paper/table_main.tex` is one draw.
- The noise is **biased**: it is one Gumbel(0, 1) draw, whose mean is 0.577, not a symmetric
  logistic noise. A unit with logit 0 fires with probability 0.63, not 0.5.
- The teacher **needs** the noise to be accurate. With every bit at its most likely value it
  drops from 99.1 to 85.8 % on BAMultiShapes, 81.9 to 62.5 % on BA2Motifs and 85.8 to 69.2 % on
  MUTAG. Its head reads mean / sum pooled *samples*, which behave like probabilities.
- 22 % (Mutagenicity) to 86 % (AIDS) of the teacher's node bits are uncertain (0.05 < P(1) < 0.95).
  A student is distilled on a new sample of them every epoch, so the best bit accuracy a
  deterministic student can reach is 71-94 %.
- The students learn those probabilities. On the bits the teacher is unsure about, 80-88 % of a
  student's unit outputs are graded (in (0.1, 0.9)), against 13-26 % on the bits it is sure about,
  and they correlate with the teacher's P(1) (r = 0.63-0.83). Reading the student as rules
  binarises exactly these units at 0.5: that is a large part of the network-vs-rules infidelity
  that QAT then has to remove.

## 2. How `gumbel_sigmoid` works

```python
gumbels = -torch.empty_like(logits).exponential_().log()   # g ~ Gumbel(0, 1)
if deterministic: gumbels = 0
y_soft = ((logits + gumbels) / tau).sigmoid()
y_hard = (y_soft > 0.5)                                    # hard=True
return y_hard - y_soft.detach() + y_soft                   # straight-through
```

For a unit with pre-activation (logit) `l`:

- **Forward:** the bit is `[l + g > 0]` with `g ~ Gumbel(0, 1)` (`-log E`, `E ~ Exp(1)`). Dividing by
  `tau` rescales `l + g` but never changes its sign, so the hard bit has the **same distribution
  for every tau**. `tau = 1000` at evaluation (`test_epoch`, `evaluate`) therefore does not switch
  the noise off. Only `deterministic=True` does, and no evaluation path passes it.
- **Backward:** straight-through: the gradient is that of `sigmoid((l + g) / tau)`, with `tau = 1` in
  training.
- **Probability of a 1:** `P(l + g > 0) = P(g > -l) = 1 - exp(-e^l)`, since the Gumbel CDF is
  `exp(-exp(-x))`. This is **not** `sigmoid(l)`:

  | logit l | -2 | -1 | -0.367 | 0 | 1 | 2 |
  |---|---|---|---|---|---|---|
  | P(bit = 1) | 0.13 | 0.31 | 0.50 | **0.63** | 0.93 | 0.9994 |
  | sigmoid(l) | 0.12 | 0.27 | 0.41 | 0.50 | 0.73 | 0.88 |

  The binary Concrete / Gumbel-sigmoid of Maddison et al. (2017) and Jang et al. (2017) adds the
  *difference* of two Gumbels (a logistic noise, mean 0), which gives `P = sigmoid(l)`. Upstream adds
  a single Gumbel (mean γ = 0.577, checked by Monte Carlo: P(g > 0) = 0.6325, mean 0.5772), so
  every unit is biased towards 1.
- **Most likely bit:** `P >= 1/2` iff `l >= ln ln 2 = -0.367`. `deterministic=True` thresholds at 0
  instead, which is not the mode: units with `l` in [-0.367, 0) are 1 more often than not, but read
  0 in deterministic mode.

## 3. Why the noise is there

LogiX-GIN distils the teacher's hidden states into logical layers, which need **binary** targets.
A hard threshold has no gradient, so the teacher is trained through a stochastic binary unit with a
straight-through estimator (Bengio et al. 2013), in the Gumbel-softmax family. The noise is what
makes this work in training:

- it explores both values of a unit whose logit is near the threshold, so the gradient sees the
  effect of flipping it;
- it penalises small margins: a unit whose logit stays near 0 gives a random bit, which hurts the
  loss, so training pushes logits away from 0 and most bits become near-deterministic;
- it regularises like dropout on the binary code.

What it was presumably *not* meant to do is stay on at evaluation: with `tau` large the soft value
is 0.5 for every unit, which suggests the intent was a sharp, deterministic sample. Because the hard
threshold ignores `tau`, the sample stays random.

## 4. Measurements (test, 10 folds, 20 evaluations per fold)

All in %. *reported*: the accuracy in the teacher's `total_results.csv` (one draw, the number the
paper uses). *mean*: over the 20 draws. *eval sd / range*: within a fold, mean over folds (*max
range*: the worst fold). *deterministic*: `deterministic=True`. *most likely*: every bit at its
mode `[l >= -0.367]`. *uncertain*: node bits with 0.05 < P(1) < 0.95. *ceiling*: mean max(P, 1 - P),
the best bit accuracy a deterministic student can have against the sampled targets. *agree2*:
agreement of two independent draws.

| dataset | reported | mean | eval sd | eval range | max range | deterministic | most likely | uncertain | ceiling | agree2 |
|---|---|---|---|---|---|---|---|---|---|---|
| MUTAG | 85.6 | 85.8 | 3.0 | 9.6 | 15.8 | 69.1 | 69.2 | 56.5 | 85.8 | 79.6 |
| Mutagenicity | 79.8 | 79.9 | 0.9 | 3.2 | 3.9 | 76.6 | 78.0 | 22.4 | 94.1 | 91.6 |
| AIDS | 99.8 | 99.7 | 0.0 | 0.1 | 0.5 | 99.3 | 96.3 | 85.7 | 70.6 | 63.1 |
| NCI1 | 77.1 | 77.4 | 0.9 | 3.6 | 4.9 | 73.8 | 76.4 | 25.8 | 93.0 | 90.2 |
| BBBP | 86.5 | 86.3 | 0.8 | 2.8 | 3.4 | 82.1 | 82.7 | 42.2 | 88.5 | 84.0 |
| PROTEINS | 74.9 | 74.4 | 1.1 | 4.3 | 5.4 | 71.5 | 71.8 | 39.7 | 88.9 | 84.7 |
| BA2Motifs | 81.7 | 81.9 | 1.8 | 6.1 | 19.0 | 60.4 | 62.5 | 61.0 | 81.4 | 75.5 |
| BAMultiShapes | 98.8 | 99.1 | 0.8 | 2.4 | 4.0 | 62.6 | 85.8 | 56.5 | 83.9 | 78.3 |

Reading:

- The *reported* number is an unbiased draw (it matches *mean*), but with a fold-level noise of
  about 1 point (3 on MUTAG) that the paper's ± over folds mixes with the fold-to-fold variance.
  The checkpoint itself was selected on a noisy validation accuracy, so it is partly selected for a
  lucky validation draw.
- *Deterministic* and *most likely* are 3-37 points below *mean*: the teacher's decision function
  is the noisy one. The pooled readout (mean | max | sum over nodes) of sampled bits estimates
  `E[bit] = P(1)` per unit and graph, a graded quantity, and the head (`fc1`, `fc2`) was trained on
  those noisy pools. Thresholding every unit removes that graded information. AIDS is the exception
  that shows it: 86 % of its bits are uncertain, yet the teacher is at 99.7 % in every mode, because
  the atom types alone separate its classes.
- BA2Motifs: folds 2, 4, 7, 9 are at chance in every mode (the known trap of these teachers,
  `paper/table_main.tex` TODO); on the other folds the teacher is ~99 % noisy, 52-100 % most likely.

## 5. Effect on the students

The students (`train_logic.py`, `train_proto.py`) distil each conv layer by BCE against
`teacher.forward_e`, which samples new bits at every batch (`utils/training.py: distill_trunk`).

1. **Targets are Bernoulli samples.** The BCE against a target that is 1 with probability p is
   minimised by the output p. A student unit that copies a teacher unit therefore learns its
   probability, not a bit. Measured on test, fold 0 (40 teacher draws; *graded*: student unit output
   in (0.1, 0.9); the student's own soft states, `interp_metrics.trunk_states(symbolic=False)`):

   | student | teacher bits uncertain (0.1 < P < 0.9) | corr(student output, teacher P) on them | graded on uncertain bits | graded on certain bits |
   |---|---|---|---|---|
   | NMP, Mutagenicity | 22 % | 0.63 | 80 % | 17 % |
   | classic, PROTEINS | 47 % | 0.67 | 83 % | 13 % |
   | NMP, BAMultiShapes | 55 % | 0.83 | 88 % | 26 % |
   | NMP + QAT, Mutagenicity | 22 % | 0.57 | 0 % | 0 % |

   QAT binarises every unit, so its outputs are bits; the others are graded exactly where the
   teacher is random.
2. **Bit agreement has a ceiling.** Against a fresh sample, a deterministic student can agree on at
   most the *ceiling* of §4 (on fold 0's test nodes: 93 % of the bits on Mutagenicity, 84 % on
   PROTEINS, 82 % on BAMultiShapes). The symbolic students of fold 0 agree with a fresh draw on
   88-89 % (Mutagenicity, NMP and QAT), 79 % (PROTEINS classic) and 78 % (BAMultiShapes NMP), 4-5
   points under the ceiling: most of the distillation error is the teacher's noise, the rest is
   the student. Against the teacher's most likely bit they agree on 92-93 %, 86 % and 87 %. The
   `distill/L*_acc` curves in MLflow are bounded the same way.
3. **The rule reading binarises the graded units.** The rules read every unit at 0.5. Where the
   student outputs a probability, binarising it at 0.5 is the student's version of the teacher's
   *most likely* mode, which §4 shows loses accuracy. This is one mechanism behind the low fidelity
   of the non-QAT students (71 % NMP Mutagenicity, 57 % classic PROTEINS on fold 0) and behind the
   accuracy QAT gives up to make the network equal to its rules.
4. **Teacher vs student comparisons.** The student is evaluated deterministically, the teacher by
   one random draw. A "student within x points of the teacher" statement compares a deterministic
   model with one noisy sample of a stochastic one whose deterministic versions are 3-37 points
   worse.
5. **Not affected:** the splits, the search and the student evaluation (no noise at student test
   time). The teacher's eval-mode switch of `--teacher_eval` (audit issue 4) removes its dropout, not
   this sampling: Gumbel draws happen in both modes.

## 6. Options

| option | cost | what changes |
|---|---|---|
| Report the teacher as the mean over K draws (e.g. 20) per fold, and say it is stochastic | none (CPU, minutes) | the teacher row only; honest about what the teacher is |
| Also report the most-likely-bit teacher | none | shows how much the teacher relies on sampling |
| Distil the students on P(1) (soft targets, the expectation of the sampled bit) instead of samples | student reruns | removes the target noise; the student's units are then explicitly probabilities, which says plainly why its rules need QAT |
| Retrain the teachers with a symmetric logistic noise annealed to 0, so that the deterministic teacher is the trained one | every teacher and student retrained | the teacher becomes a deterministic binary network; a different method from LogiX-GIN upstream |

The first two need no training and can go in the paper now (`paper/table_main.tex`, TEACHER ROW
note). The last two change the method and are decisions for the authors.

## 7. Reproduce

```bash
# §4 (CPU, ~15 min): every k-fold teacher, 20 draws per fold
python teacher_noise.py --datasets MUTAG Mutagenicity AIDS NCI1 BBBP PROTEINS BA2Motifs BaMultiShapes \
    --draws 20 --out teacher_noise.csv
# §5.2: a student against its teacher's samples
python teacher_noise.py --student results_final/<exp>/<dataset>/<model>/fold0/run/.../0 --draws 20
```

Measured on commit `fix/audit` (2026-10-22) with the teachers of `results/<dataset>/
batch_size=128|dropout=0.15|epochs=500|hidden_dim=64|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3|split=kfold`
and the final runs of `sum_ablation`, `sum_ablation_ctrl` and `qat_nmp` (fold 0); §5 with
`--student <those four runs> --draws 40` (columns `teacher_unsure`, `corr_soft_p1_unsure`,
`graded_on_unsure`, `graded_on_sure`, `vs_one_draw`, `vs_majority`, `ceiling`).
