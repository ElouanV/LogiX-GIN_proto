# Interpretability claims: open issues

Written 2026-10-21. A working list of what is missing, incomplete or wrong in the
interpretability claims of the paper. We address the points one at a time. Each issue has
a status, the problem, the evidence (numbers, with their source), where it lives in the
code, a proposed fix with its cost, and the decision it needs.

All numbers are **preliminary**: test split, runs from before the head-saturation fix of
2026-10-19 (`sum_ablation`, `sum_ablation_ctrl`, `qat_nmp*`). They are 10-fold unless
marked "3 folds". Explanation metrics are the *evidence* versions (`interp_metrics.py`
`*_evidence`: head literals constant on every input count as a bias and are not cited)
unless marked "plain". Code references are to branch `feat/paper-figures` (worktree
`../LogiX-GIN-paper`), not yet merged.

## Summary

| # | Issue | Kind | Status |
|---|---|---|---|
| 1 | GT precision is reported without its coverage | metric | agreed, to do (CPU) |
| 2 | Our "fidelity" is network-vs-rules, not explanation-vs-input (no Fid+/Fid−) | metric | agreed, to do (CPU) |
| 3 | Explanation size counts opaque trunk bits, not readable literals | metric | agreed, to do (CPU) |
| 4 | Head literals are non-monotone (bands, unions) and mostly constant | model | decided 2026-10-21; implemented (c76095a), to train |
| 5 | The head reads prototype *similarity degrees*, not whether the prototype rule holds | model / claim | implemented (8ade359), running 2026-10-22 |
| 6 | QAT explanations rarely point at nodes | finding | open, follows from 4-5 |
| 7 | No competitors, no post-hoc reference for GT precision | experiment | waiting on model design |
| 8 | No evidence that humans understand the explanations (simulatability) | experiment | to decide, last |
| 9 | Model-randomisation sanity check | experiment | code done, full run to do |
| 10 | Stability of prototypes and explanations across folds / seeds | experiment | not started |
| 11 | Protocol gaps (pre-fix runs, missing datasets, 3-fold rows, no tests) | protocol | waits for E1 |
| 12 | Accuracy / fidelity of the non-QAT students | performance | context |

What we can claim today: an exactly faithful logical GNN (QAT: the network *is* its rule
model) at a small accuracy cost, with explanations that are short in head literals and
exact. What we cannot claim yet: "human-understandable", or "localises the known cause".

---

## Where the model stands (context for every issue)

| Model | Dataset | Network acc. | Rule acc. | Fidelity | Teacher |
|---|---|---|---|---|---|
| NMP+QAT | Mutagenicity | 76.6 | 76.6 | 100 | 79.8 |
| NMP+QAT | BAMultiShapes (3 folds) | 98.7 | 98.7 | 100 | ~99 |
| NMP+QAT | PROTEINS (3 folds) | 75.3 | 75.3 | 100 | 75 |
| NMP | Mutagenicity | 80.4 | 66.8 | 73.6 | 79.8 |
| NMP | MUTAG | — | 39.8 | 100 | — (collapsed, constant output) |
| classic LogiX-GIN | MUTAG / PROTEINS / BAMultiShapes | — | 58-77 | 63-81 | — |

Sources: `paper/table_main.tex`, `paper/table_interp.tex`, `paper/table_binarisation.tex`.
AIDS, BBBP, NCI1 and BA2Motifs have no student results yet.

---

## 1. GT precision is reported without its coverage

**Status:** agreed, to do.

**Problem.** GT precision is the share of cited nodes that fall in the ground-truth region,
averaged over the GT-scored predictions *that cite a node*. A prediction whose evidence
cites no node (it is decided by constant literals or by graph-level averages) is left out.
So a model that rarely points at nodes can still show a decent GT precision.

**Evidence** (scored predictions that still cite a node, plain → evidence):

| Model | Coverage | GT prec. (chance) |
|---|---|---|
| NMP+QAT Mutagenicity | 523 / 1493 = **35%**; folds 0 and 1: none; fold 3: GT prec. 0 | 43 ± 41 (22) |
| NMP Mutagenicity | 1137 / 1382 = 82% | 50 ± 26 (23) |
| NMP+QAT BAMultiShapes (3 folds) | 759 / 1171 = 65% | 87 ± 17 (33) |
| NMP+QAT K≤8 BAMultiShapes (3 folds) | 196 / 244 = 80% | 99 ± 1 (33) |

Mutagenicity GT precision is about 2× chance with a large spread. NO₂/NH₂ is also a
debated ground truth for Mutagenicity.

**Where.** `interp_metrics.py`, `interp_metrics()`: `gt_n` / `gt_n_evidence` are already
stored; the ratio is not.

**Fix.** Add `gt_coverage_evidence = gt_n_evidence / gt_n` to `interp_metrics`, rerun the
interp stage (CPU, ~20 min for all finished experiments), and add a coverage column (or a
sentence) next to GT prec. in `table_interp.tex`. Report a fold with no cited node as
coverage 0, not as a missing value.

**Decision needed.** Column in the table, or in the text only.

---

## 2. Our "fidelity" is not the explanation fidelity reviewers expect

**Status:** agreed, to do.

**Problem.** `logic_fidelity` measures whether the rule model agrees with the network. That
is the faithfulness of the *model reading*. The GNN-explanation literature reports
fidelity of the *explanation on the input* (Yuan et al. 2022 survey; GraphFramEx, Amara et
al. 2022):

- **Fid+** (necessity): how much the prediction changes when the explanation is removed
  from the graph.
- **Fid−** (sufficiency): how much it changes when only the explanation is kept.

We report neither, so our numbers can't be compared with any competitor.

**Where.** New metric next to `interp_metrics.py`. The per-graph cited nodes now exist
(`interp_metrics(..., records=True)`, commit 1d75d24).

**Fix.** For each test graph, take the cited nodes, run the rule model (a) on the graph
without those nodes and (b) on the subgraph of those nodes alone, and report the change in
predicted class / probability. Also report the same with a random node set of the same size
as a baseline.

**Caveats.**
- Removing nodes creates out-of-distribution graphs; report both deletion and insertion.
- On QAT, the evidence cites no node in most graphs (issue 6), so Fid+ / Fid− is undefined
  there. Report on how many graphs it is defined.

**Cost.** CPU, about the cost of one interp stage.

---

## 3. Explanation size counts opaque trunk bits, not readable literals

**Status:** agreed, to do.

**Problem.** "Bits" counts the trunk bits the cited prototypes care about. Only conv layer 0
decodes into short atom-count rules (~3 literals). A layer 1-2 bit needs about 10-14
literals and has about 1e10 minimal covers (CLAUDE.md, rule-extraction limits). So an
explanation of "43 bits" is mostly 43 opaque features such as `L2u4`. The worked-example
figure shows it: the layer-2 boxes read `n_NOT L1u29 = 4`. The size we report undercounts
what a human must read.

**Evidence.** QAT Mutagenicity: 2.5 head literals, 43 bits per prediction. The share of
cited units in layer 0 (`units_cited_L0_evidence`) is ~0.20 on QAT fold 0.

**Where.** `interp_metrics.py` `explanation_metrics()`; decoding in `min_covers.py`
(greedy minimum-cardinality bound, `count_min_covers`).

**Fix.** Report, per prediction:
1. The explanation unfolded down to layer 0: for every cited bit, the shortest sufficient
   set of true literals at the node where it is evaluated, recursively (what
   `viz_proto.plot_unfolding` already draws for one graph). That is the length a human
   reads.
2. The share of the explanation made of layer-0 conditions.

The greedy bound is instant and exact, so this costs no rule enumeration.

**Decision needed.** Which number becomes the headline size: head literals, bits, or
unfolded literals. Recommendation: unfolded literals as the main number, the others in the
appendix.

---

## 4. Head literals are non-monotone and mostly constant

**Status:** decided 2026-10-21. Implemented behind a flag (feat/paper-figures c76095a), not
yet trained.

**Decision.**
- **Head:** bound the slope, and initialise the thresholds inside the data, so that a
  literal that does not discriminate ends up unused (no head weight) rather than acting as
  a hidden bias.
- **Trunk:** keep the periodic `Phi`. A decoded condition such as
  `n_NOT Cl ∈ [1.00, 2.11] ∪ [4.86, 5.00]` is extra expressiveness, which we want there.
- **Open:** classic LogiX-GIN. Recommendation: keep it upstream, as published, and report
  its band and constant rates next to ours.

**Where it lives.** Every literal goes through `Phi` (`models_proto/tell.py`, identical to
upstream `models/tell.py`; the upstream head is `LogicalLayer(..., dummy_phi_in=False)`,
`models/model.py:141`):

```python
z = w * x + b                     # w = exp(w_) > 0, one (w, b) per input column; init w = 1, b = 0
step(z) = clamp(((Σ_{i=0..τ} sin((2i+1) z)/(2i+1)) / norm + 1) / 2, 0, 1)
```

`step` is a truncated Fourier *series* of a square wave (not a transform), period 2π.
τ = 0 by default, i.e. `(1 + sin z)/2`. Training sets τ = 10 in the end-to-end phase
(`train_logic.py:99`, `train_proto.py`, `utils/training.py:60`), close to a hard square
wave.

`Phi` does not compute intervals. Each input column gets one literal. The intervals appear
when the literal is read as a rule: binarised at 0.5, it is true when `sin z ≥ 0`, i.e.
`(w·x + b) mod 2π ∈ [0, π]`, the same set for every τ. In x that is the union of
`[(2kπ − b)/w, ((2k+1)π − b)/w]` (`min_covers.literal_true_intervals`, used only for the
readout).

| z over the input range | Literal |
|---|---|
| inside one half-period (e.g. the init: z ∈ [0, 1] ⊂ [0, π]) | constant (here true on every input) |
| crosses one boundary | threshold `x ≥ t` or `x ≤ t` |
| crosses two or more (only if w > π on [0, 1]) | band or union of intervals |

In the head, x is a pooled similarity or mean / max of bits, in [0, 1]. The head reads
`[s, 1 − s]`, so each pooled value already has a positive and a negated literal: a band can
be built from two thresholds, so the periodicity adds nothing needed there. The entropy term
of the loss (`fc.phi_in.entropy`) and τ = 10 both reward a large w, which is when bands
appear (hypothesis, not tested).

**Problem 1: bands and unions.** E.g. `mean match(P15) ∈ [0, .16] ∪ [.31, .46] ∪ [.62, .77]
∪ [.92, 1]`: exact, but not readable as "resembles P15" or its negation.

**Problem 2: constant literals.** At init every literal is true on every input. A literal
only becomes input-dependent if training moves it enough; many never do, and a constant-true
literal with a head weight is a hidden bias.

**Evidence: our students** (shape of the trained head literals on [0, 1], all folds, among
literals with a non-zero head weight):

| Model | Threshold | Band or union | Constant true | w > π (all literals) |
|---|---|---|---|---|
| NMP Mutagenicity | 78 | 20 | 10 | 8% |
| QAT Mutagenicity | 59 | **44** | 32 | **20%** |
| QAT BAMultiShapes | 103 | 24 | 218 | 10% |

**Evidence: classic LogiX-GIN has the same problem** (`sum_ablation`, 10 folds, head
literals on mean / max columns; sum columns left out, see below):

| Model | Dataset | Constant on [0, 1] | Bands / unions among the input-dependent literals the head uses |
|---|---|---|---|
| classic (mean, max, sum) | MUTAG | 85% | 0% (58 thresholds) |
| | PROTEINS | 99% | **43%** (21 of 49) |
| | BAMultiShapes | 93% | 15% |
| classic (mean, max) | MUTAG | 90% | **38%** (26 of 68) |
| | PROTEINS | 96% | 15% |
| | BAMultiShapes | 93% | 26% |

- Classic has *more* constant literals than ours (85-99% against ~50%). Few literals ever
  leave the init, consistent with its head staying at init (the saturation fixed for the
  prototype students, not upstream).
- Fewer than 2% of classic literals have w > π, but those are over-represented among the
  literals the head weighs.
- Classic also reads sum-pooled counts and negates them as `1 − count` upstream. Those
  inputs reach tens, so the same φ crosses many periods even with a moderate w. Likely
  mostly unions; not measured.

So this is a property of the TELL head, not of our prototypes or of QAT (QAT makes it worse
on Mutagenicity). The paper can present it as an unreported limitation of LogiX-GIN's
literals and the bound as the fix.

**Implemented** (feat/paper-figures c76095a; `tests/test_bounded_phi.py`):
- `models_proto/tell.py` `BoundedPhi(Phi)`: slope `w = π·sigmoid(w_)` and threshold
  `t = sigmoid(t_) ∈ (0, 1)`, `z = w·(x − t)`. Since w ≤ π and 0 < t < 1, z stays in
  [−π, π]: the literal read at 0.5 is **exactly `x ≥ t`** (its negation on `1 − x`:
  `x ≤ 1 − t`), never a band, never true on every input. `w`, `b = −w·t` and `t` keep
  `Phi`'s meaning, so the explorer, `unpack_rules`, `interp_metrics` and the figures read
  it unchanged.
- `GINTELLProtoBase.bound_head()`: swaps the head's `phi_in`. Refuses a node-level sum
  readout (unbounded counts in the head).
- `utils.training.center_head_thresholds`: each threshold starts at the median of its
  column over the training set (teacher states), so every literal first splits the training
  graphs in two. Runs before the bias centring (`--head_init center`), which reads the
  literals.
- `train_proto.py --head_phi bounded [--head_threshold center|mid]`. Off by default and out
  of the results path, so earlier runs reproduce. The trunk keeps `Phi`.
- Checks: 2000 random (w_, t_) all give exactly `[t, 1]`; the binarised output matches
  `x ≥ t` with τ = 10; a bounded head has no literal constant on [0, 1]; pickling round
  trip; full suite (206 tests) passes. CPU smoke run (MUTAG fold 0, 60 epochs, scratch
  output): trains end to end, all 64 head literals are single thresholds, thresholds
  centred at a median t of 0.20 (range 0.11-0.88).

**Notes.**
- With w ≤ π the soft (τ = 0) literal `(1 + sin z)/2` is monotone only for |z| ≤ π/2; near
  the ends of [0, 1] it bends back while staying on the same side of 0.5. The rule reading
  is exact; only the soft network's value is affected. A bound of π/2 would make it
  monotone too, at the price of a softer step. Keep π unless training shows a problem.
- "Unused rather than a hidden bias" relies on the head weight penalty (`fc_reg`) shrinking
  the weight of a literal that does not help. A threshold can still drift outside the range
  the data actually covers and become constant *on the data*; the evidence metrics would
  then still count it as input-dependent. Measure the constant-on-data rate after training.

**To do.**
1. Add model entries to `configs/models.yaml` (e.g. `node_mask_push_qat` + `head_phi:
   bounded`), smoke test on one fold on GPU, estimate the sweep.
2. Measure: band rate (should be 0), constant-on-data rate, explanation size, coverage
   (issue 1), accuracy against the unbounded head.
3. Decide classic: upstream as published (recommended) or an extra "classic + bound" row.

---

## 5. The head reads prototype similarity degrees, not the prototype rule

**Status:** implemented 2026-10-22 on `feat/predicate-readout` (worktree
`../LogiX-GIN-predicate`, 8ade359): `models_proto/model_proto_count.py` `GINTELLProtoCount`,
`train_proto.py --node_readout predicate`. Running: `configs/experiments/predicate_readout_3fold.yaml`
(see "Decisions taken" below). Position agreed: a prototype is a predicate (a rule over
trunk bits), not a graph; we do not ground it on an example molecule.

### Notation

For a graph G with nodes V, node v has binary trunk bits `l_v ∈ {0,1}^d` (all conv
layers, read as rules). Prototype k has a pattern `p_k ∈ {0,1}^d` and a care mask
`c_k ∈ {0,1}^d` with `d_k = Σ_j c_kj` cared bits (`models_proto/proto.py`).

- Mismatches of node v with prototype k: `m_k(v) = Σ_j c_kj · [l_vj ≠ p_kj] ∈ {0, …, d_k}`
- The prototype rule: `P_k(v) := ∧_{j : c_kj = 1} (l_vj = p_kj)`, i.e. `m_k(v) = 0`.
  E.g. `P4 = ¬L0u56 ∧ ¬L1u43 ∧ … ∧ L2u59` (13 conditions).
- Similarity: `s_k(v) = exp(−m_k(v) / T)` with T annealed to 1 (`PrototypeLayer.similarity`).

### What the model computes today

The node model pools the node similarities per graph (`GINTELLProtoNode`, `pool_ops = mean, max`):

```
x_k^max(G)  = max_v  s_k(v)                    = exp(−min_v m_k(v) / T)
x_k^mean(G) = (1/|V|) · Σ_v s_k(v)             = (1/|V|) · Σ_v exp(−m_k(v) / T)
```

and the head reads literals `φ(x) ≥ 0.5` on `[x, 1 − x]`.

- **max is a rule.** `x_k^max ≥ t ⇔ min_v m_k(v) ≤ −T·ln t`: "some node satisfies P_k up to
  μ exceptions", μ = ⌊−T ln t⌋. That is an M-of-N rule (at least d_k − μ of the d_k
  conditions hold) with an existential quantifier. Readable.
- **mean is not.** It mixes *how many* nodes match with *how well* each matches, through
  the exponential. Example, two graphs of 20 nodes and a prototype of 13 cared bits:

  | | nodes | `m_k(v)` | `x^mean` | P_k holds somewhere? |
  |---|---|---|---|---|
  | A | 1 exact match, 19 far | 0, then 8 each | 1/20 + 19·e⁻⁸/20 = **0.0503** | yes, at one node |
  | B | 20 near misses | 3 each | e⁻³ = **0.0498** | no |

  A literal `x^mean ≥ 0.05` says the same about A and B. So "mean over nodes of
  match(P28) ≤ 0.73" is not a statement about the rule P28, and P15 (satisfied by no
  training node) can still be cited.
- **sum is not either, today.** Sum pooling exists (`pool_ops` with sum,
  `node_mask_push_sum`), but it sums the *soft* similarities, so it has the same mixture
  as mean, scaled by |V|. And the head reads it with the periodic `Phi` on values up to
  ~94 (Mutagenicity), so its literals are unions of many intervals (issue 4).

**Why max alone is not enough.** Max only says whether the pattern exists. The models rely
heavily on the mean literals (the "how widespread" information). Share of the literals
cited in the evidence explanations (test, all folds):

| Model | max | ¬max | mean | ¬mean |
|---|---|---|---|---|
| NMP Mutagenicity | 28% | 45% | 15% | 12% |
| QAT Mutagenicity | 12% | 35% | 20% | 32% |
| QAT BAMultiShapes (3 folds) | 3% | 1% | 43% | 53% |

Dropping mean would remove half of QAT's Mutagenicity explanations and nearly all of its
BAMultiShapes ones. The fix is not to change *which* pooling, but *what* is pooled: pool a
crisp predicate instead of a soft degree.

### Proposal: pool the rule, not the similarity

**Step 1, node predicate with a tolerance.** Per prototype, a learned tolerance μ_k:

```
q_k(v) = [ m_k(v) ≤ μ_k ]  ∈ {0, 1}         "P_k holds at v up to μ_k exceptions"
```

Implemented as a bounded literal on the similarity (issue 4's `BoundedPhi`, since
`s_k(v) ∈ [0, 1]`): `q_k(v) = [s_k(v) ≥ τ_k] ⇔ m_k(v) ≤ −T·ln τ_k`, so μ_k = ⌊−T ln τ_k⌋.
Exact, single threshold, learned per prototype. (Option: μ_k = 0, the strict conjunction.)

**Step 2, pool the predicate.** Two readouts per prototype:

```
n_k(G) = Σ_v q_k(v)        ∈ {0, 1, …, |V|}   count  (sum pooling)
r_k(G) = n_k(G) / |V|      ∈ [0, 1]           share  (mean pooling)
```

Max pooling becomes `[n_k ≥ 1]`, a special case of a count literal, so it is not needed
as a separate readout. Unlike today, every value is a statement about the rule: in the
table above, A has `n_k = 1, r_k = 0.05` and B has `n_k = 0, r_k = 0`.

**Step 3, head literals.** The two readouts need different literals:

- **Share `r_k ∈ [0, 1]`:** issue 4's bounded head literal, unchanged: `r_k ≥ t` and, on
  `1 − r_k`, `r_k ≤ 1 − t'`.
- **Count `n_k`, unbounded:** issue 4's bound does not apply. Bounding the slope by π over
  the count range (up to ~100 nodes) would make the step so soft that 0 and 1 node can't be
  told apart, and `1 − n` is not a negation of a count. A new count literal instead, with
  two independently parametrised directions:

  ```
  at least:  [ n_k ≥ a_k ]     z = w·(n_k − a_k),   value = (1 + sin(clamp(z, −π/2, π/2))) / 2
  at most:   [ n_k ≤ b_k ]     z = w·(b_k − n_k),   same squash
  ```

  The clamp keeps the squash monotone for any slope w > 0 (no periodicity, no bands), so
  the literal binarised at 0.5 is exactly `n_k ≥ a_k`. Counts are integers, so it reads
  `n_k ≥ ⌈a_k⌉`. Init: a_k, b_k at the median count over the training graphs (like the
  centred thresholds of issue 4), or a_k = 0.5 (existence).

**Training.** `q_k(v)` is soft during training (the bounded literal's value), so `n_k`,
`r_k` and their gradients are those of soft counts; QAT binarises `q_k` progressively
(Quant-Noise, as it does the other bits), and at rate 1 the counts are exact. The
straight-through estimator is the same as everywhere else in the model.

### What the explanations become

A class rule is a weighted threshold over literals such as:

| Literal | Reads | Cites |
|---|---|---|
| `n_4 ≥ 2` (μ_4 = 1) | "at least 2 nodes satisfy P4 with at most 1 of its 13 conditions violated" | those nodes |
| `n_9 ≤ 0` | "no node satisfies P9" | nothing (evidence of absence) |
| `r_28 ≤ 0.3` | "at most 30% of the nodes satisfy P28" | nothing for ≤; the satisfying nodes for ≥ |

(Illustrative values.)

- **Every literal is about the rule P_k**, with an explicit tolerance and an explicit count
  or share. A prototype that no node satisfies at its tolerance gives a constant-false
  literal, which the evidence already discounts; support and confidence of `q_k` on the
  training set are properties of the rule (rule mining), reported per prototype.
- **Cited nodes are exact.** A positive count or share literal cites the nodes with
  `q_k(v) = 1`, a set defined by the rule, instead of today's "best-matching nodes" (the
  argmax of a similarity). GT precision is then the share of those nodes in the
  ground-truth region, well defined.
- **Absence is explicit.** Negated literals ("no node satisfies P9") are a large part of the
  current explanations (¬max: 35-45% on Mutagenicity) and cite no node by nature. Issue 1's
  coverage should count them separately: an explanation by absence is not a failure to
  localise. This also accounts for part of issue 6.
- **Size.** One literal = one prototype rule (its d_k conditions) + μ_k + a count / share
  threshold. The bits count of issue 3 is unchanged.

### Expected effects and risks (hypotheses, to measure)

- Readability: no mean-of-exponentials literal left; with issue 4, no band.
- Accuracy: unknown. Counts may help where the label depends on how many times a pattern
  occurs (several NO₂ groups; BAMultiShapes combinations of motifs). Hard predicates may
  cost accuracy before QAT; QAT already showed it recovers from binarisation.
- Gradient: a clamped count literal has zero gradient when the count is far from its
  threshold (|z| > π/2). Median init keeps it active at start; a sigmoid squash is the
  fallback (also monotone).
- Graph size: counts grow with |V|; the share column normalises it. Keep both and let the
  head choose.

### Where in the code

| Part | File | Change |
|---|---|---|
| Node predicate q_k | `models_proto/model_proto.py` `GINTELLProtoNode.readout` | `BoundedPhi` per prototype on `s_k(v)` |
| Pooling of q | `models_proto/model_proto.py` `_pool`, `POOLS` | `count` (sum of q), `share` (mean of q) |
| Count literal | `models_proto/tell.py` | new `CountPhi` (two directions, clamped squash) |
| Head input | `head_input`, `head_columns` | `[r, 1 − r]` through `BoundedPhi`, `n` through `CountPhi` |
| Symbolic reading | `interp_metrics.py` `readout_values`, `head_literal_map`, `head_literal_constants`, `cited_nodes` | q, n, r binarised; cite the nodes with q = 1 |
| Text | `utils/viz_proto.py`, `explorer/reader.py` | "at least a nodes satisfy P_k (≤ μ exceptions)" |
| Flag | `train_proto.py` | e.g. `--node_readout predicate` (off by default, out of the results path) |

Graph-level prototypes are not concerned: the head reads one similarity per graph there,
already through issue 4's bounded literal.

### Decisions needed

1. Learned tolerance μ_k, or strict (μ_k = 0)?
2. Readout: count + share (recommended), count only, or count + share + exists?
3. Count squash: clamped step (TELL style, recommended) or sigmoid?
4. Train together with issue 4 as one new model (`node_mask_push_qat` + bounded head +
   predicate readout), one smoke test, one sweep (recommended), or separately as an
   ablation (more GPU)?

### Decisions taken (2026-10-22, the recommended options; revisit with the results)

1. **Learned tolerance.** `q_k(v) = [s_k(v) ≥ τ_k]` through `tell.RampPhi` on the
   similarity, τ_k learned; τ_k starts at the median similarity of the training nodes
   (teacher states), so each q_k first holds on half the nodes. Since T anneals from d/4 to
   1 while τ_k is a similarity, the tolerance μ_k = −T ln τ_k shrinks with T (Mutagenicity
   fold 0 smoke run: median μ = 95 of 192 cared bits at T = 48, i.e. about 2 at T = 1).
2. **Count + share.** Readout `[n_1..n_K | r_1..r_K]`; head input `[n, r, −n, 1 − r]`.
3. **Clamped step.** `tell.RampPhi`: `(1 + sin(clamp(w (x − t), −π/2, π/2))) / 2`, slope
   w = exp(w_) unbounded, monotone; exactly `x ≥ t` once binarised. Counts' thresholds
   start at the training median ± 1/2 (between two integers).
4. **One model with issue 4, plus issue 4 alone as the control.** At the K = 8 QAT NMP
   best sets (no new search), folds 0-2 of Mutagenicity and BaMultiShapes:
   `node_mask_push_qat_bounded` (issue 4) and `node_mask_push_qat_predicate` (4 + 5); the
   reference is `qat_nmp_p16_3fold` / `qat_nmp_p8_3fold` (periodic head, mean / max).

Readers: `interp_metrics` reads the binarised q and cites exactly the nodes with q = 1;
`min_covers.phi_true_intervals` gives a RampPhi literal's truth set `[t, hi]` (used by
interp_metrics, viz_proto, the explorer). The explorer and viz_proto texts still show a
negated count literal on `−n` (e.g. `count sim(P3) ∈ [−4, −0.5]` for `n_3 ≤ 0.5`); to
reword once the results say the model is worth keeping.

---

## 6. QAT explanations rarely point at nodes

**Status:** open finding. It probably follows from issues 4 and 5.

**Problem.** Once constant literals are removed, QAT's evidence cites a node in only 35% of
the GT-scored Mutagenicity predictions (NMP: 82%), and in about 10% of all test graphs on
fold 3. Most of its decisions are made by constant literals (the bias) plus graph-level
averages. Its high accuracy at fidelity 1 is real, but its explanations are mostly "the
bias, plus the average similarity to Pk is below t", not "this part of the molecule".

**Evidence.** Issue 1 table. Effect of the evidence change, fold 3: 5.1 → 1.6 literals,
GT prec. 0.68 → 0.00 (all its NO₂/NH₂ hits came from constant literals). Randomisation
smoke test (issue 9): inconclusive on QAT for the same reason.

**Fix.** Re-measure after issues 4 and 5. If it persists, it is a negative result of QAT for
localisation and goes in the paper as such.

---

## 7. No competitors and no post-hoc reference

**Status:** waiting on the model design.

**Problem.** GT precision and explanation size have no reference point. There are no
self-explainable GNN baselines and no post-hoc explainers on the teacher (GNNExplainer,
PGExplainer). Fidelity is 1 by construction for most self-explainable models, so the
comparable columns are size, GT precision and Fid+/Fid− (issue 2).

**Where.** `paper/table_main.tex`, `paper/table_interp.tex` (competitor rows, `\tbd`).

**Fix.** Define the explanation size of each competitor, run them on the same folds
(`splits/<dataset>_kfold.json`).

---

## 8. No evidence that humans understand the explanations

**Status:** to decide; last, once the model is frozen.

**What it is.** A simulatability study (Doshi-Velez & Kim 2017; Hase & Bansal, ACL 2020):
participants see some graphs with the model's explanations, then predict the model's output
on new graphs. The metric is how often they predict the model correctly, compared with no
explanation and with another method's explanation. A small study is 10-20 participants and
about 20 graphs. Our rules are exact, so a program simulates them perfectly; the study
measures whether a *human* can.

**Why.** It is the only direct evidence for "understandable". Without it, the claim must be
"faithful and compact". Issue 3 (readable size) is the proxy until then.

**Decision needed.** Run it (participants, maybe ethics approval), or scope the claim down.

---

## 9. Model-randomisation sanity check

**Status:** code done and committed; full run to do.

**What it is.** Adebayo et al., "Sanity checks for saliency maps" (NeurIPS 2018): scramble a
stage's learned weights and check that the explanations change. Ours are read off the
weights, so they change by construction. The test matters for the *metrics*: the GT
precision of a scrambled model is the baseline the trained one must beat. A GT precision
that a scrambled model also reaches measures the dataset, not the model.

**Code.** `utils/sanity.py`, `sanity_randomisation.py`, `tests/test_sanity.py`
(feat/paper-figures 82edfda; per-graph records 1d75d24).

```bash
python sanity_randomisation.py --run_path results_final/<exp>/<ds>/<model>/fold* \
    [--order independent|cascade] [--mode permute|init] [--draws 3]
```

Writes `<run dir>/sanity_<order>_<mode>_<split>.json`.

**Design choices.**
- **Order: independent by default** (one stage at a time, the rest trained). In the cascade
  order, a permuted head alone already predicts one class on every graph, so the lower
  stages say nothing more.
- **Randomise = permute each trained parameter tensor:** same values, scale and sparsity;
  the learned assignment is destroyed. Re-initialising (`--mode init`) makes every head
  literal constant true (Phi init w = 1, b = 0), so a re-initialised head cites nothing:
  an artefact of our init.

**Smoke run** (Mutagenicity fold 3, 3 draws, 15 s on CPU):

| Model | Scrambled stage | Rule acc. | Same class | Node overlap (Jaccard) | GT prec. (chance 24) |
|---|---|---|---|---|---|
| NMP | none (trained) | 63.1 | 100 | 1 | **78** |
| NMP | head | 47.3 | 39 | 0.54 | 20 |
| NMP | prototypes | 55.3 | 36 | 0.19 | 37 |
| NMP | conv 2 | 45.7 | 65 | 0.15 | 13 (n=3) |
| NMP | conv 1 | 55.4 | 36 | 0.19 | 22 |
| NMP | conv 0 | 57.1 | 43 | 0.16 | 22 |

NMP passes: explanations change with every stage and GT precision falls to chance. QAT is
inconclusive: its trained evidence cites a node in only 10% of test graphs (issue 6).

**To do.**
- Run on the final E1 folds of every node-prototype and classic model with a GT region
  (Mutagenicity, BA2Motifs, BAMultiShapes). ~10 min of CPU per dataset.
- Fill `paper/table_sanity.tex`.
- Decide on the data-randomisation test (retrain on permuted labels, one GPU retraining
  per fold).

---

## 10. Stability of prototypes and explanations

**Status:** not started.

**Problem.** We don't know whether the same prototypes (as rules) and the same explanations
come back across folds or seeds. If every fold learns different prototypes, "the model
learned the NO₂ prototype" is a statement about one run.

**Fix.**
- Across folds: match prototypes by their decoded layer-0 conditions, or by their support
  sets on a shared graph set, and report how many recur.
- Across seeds on one fold: the Jaccard of the cited node sets per test graph
  (`interp_metrics(..., records=True)` gives them).

**Cost.** CPU for folds. Seeds need retraining (GPU).

---

## 11. Protocol gaps

**Status:** waits for the final E1 experiment.

- Every number predates the head-saturation fix (2026-10-19).
- AIDS, BBBP, NCI1 and BA2Motifs have no student results; the BA2Motifs teacher dies at
  init (fix options pending).
- Several rows are on folds 0-2 only (QAT PROTEINS and BAMultiShapes, the K study).
- No significance tests except the paired Wilcoxon of `table_sum_ablation.tex`.
- The search ran on fold 0 only.

---

## 12. Accuracy and fidelity of the students without QAT

**Status:** context for the method choice.

- NMP without QAT: the rules reach 66.8% on Mutagenicity, 13 points below the teacher, at
  74% fidelity. A reader gets explanations of a different model a quarter of the time.
- Classic LogiX-GIN: 63-81% fidelity.
- NMP on MUTAG collapses to a constant output (rule acc. 39.8%); its evidence is the bias
  alone (0.1 literals), which the metric now shows correctly.

So the faithful-explanation claim holds for QAT only. That is an argument for QAT as the
headline model, which then makes issues 4-6 the priority.
