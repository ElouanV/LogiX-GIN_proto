# Symbolic readings of a LogiX-GIN model: which one each number uses

Audit issue 8 (2026-10-22). The code reads a trained student as rules in **three different ways**.
They agree on a QAT student (whose network computes its rules) and differ widely on every other
model. The authors' notebook uses the first; the explanation metrics of our paper use the third.
This document defines the three, lists which tool and which table uses which, measures the gap,
and says what to write.

## 1. The three readings

All three binarise a literal as `phi_in(input) >= 0.5` and a unit as `W.lit + b >= 0`
(`sigmoid(10 z) >= 0.5`). They differ in **what the literals of a layer are computed from**.

| | name | literals of layer l computed from | the prediction | code |
|---|---|---|---|---|
| R1 | **layer-local** | the *network's* soft outputs of layer l-1 | made by the network | `latent_logic.forward_with_activations` / `collect_activations` (port of the authors' notebook, cell 12) |
| R2 | **head only** | the network's trunk (soft, or binarised between layers for QAT), head literals binarised | the head's rules on the network's trunk | `interp_metrics.symbolic_head(model, trunk_states(..., symbolic=False))`: `head_fidelity`, `head_acc`; explorer `head_only` |
| R3 | **chained rule model** | the *binary rule outputs* of layer l-1, from the input features up; prototypes compared on these binary states; head literals binarised | made by the rules alone | `interp_metrics.trunk_states(symbolic=True)` + `symbolic_head`: `logic_acc`, `logic_fidelity`, `trunk_bit_agreement`, every `expl_*`, `protos_cited`, `gt_*`; `qat_proto.diagnostics` at rate 1; `utils/agreement.py`; the explorer |

What each reading claims:

- **R1** checks, layer by layer, that a unit's threshold rule reproduces the network's unit *when
  fed the network's own inputs*. It says "this unit computes this rule of its inputs". It does not
  say that the rules, composed, compute the prediction: an error in a layer-0 rule never reaches
  layer 1, which is fed the network's layer-0 states instead.
- **R2** composes the head's rules with the network's trunk: "given these trunk states, the
  head's rules give this class".
- **R3** composes every rule: the explanation *is* a classifier, and its accuracy (`logic_acc`)
  and agreement with the network (`logic_fidelity`) are the price of reading the network as rules.
  This is what the paper calls the rule model.

Smaller differences:

- **Aggregation `eps`.** `latent_logic.scatter_sum` assumes GIN's `eps = 0`;
  `interp_metrics.aggregate` uses `(1 + eps)`. Every checkpoint measured has `eps = 0` (it is a
  buffer, not trained), so this changes nothing today.
- **QAT.** R1 does not apply `quant_out` between layers. At quant rate 1 every `LogicalLayer`
  binarises its own literals, so R1 still agrees with the network on every unit (measured 1.000 on
  the four layers of a QAT Mutagenicity student, §3). The three readings coincide on QAT models.
- **Literal truth sets.** R1 tools describe a literal by the intervals where it holds over the
  *observed* range of its input (`find_step_intervals`, notebook cell 48); R3 tools use the exact
  intervals on the input's domain (`min_covers.literal_true_intervals`). This changes the text of a
  rule, not its truth on data.

## 2. Which reading the authors used

`nbs/LayerWiseRules.ipynb`, the canonical reading of the LogiX-GIN code:

- Activations come from `forward_with_activations` (cell 12): **R1**. Every layer's literals are
  computed from the network's soft states (`ret['x_bin'] = phi_in(scatter_sum([x, 1 - x])) >= 0.5`
  with `x` the network's output of the previous layer).
- Rules are extracted per unit from the weights (`find_logic_rules`, cell 9), restricted to
  literals with enough support on the validation activations (`min_support`, cells 9 and 32), and
  literals true on every validation graph are dropped from the printed rules (cells 34 and 52).
- The accuracy the notebook reports is the **network's** (`train_logic.test_epoch`, cells 19-28),
  also after its Hoyer sparsification. No cell evaluates the extracted rules as a classifier: there
  is no rule-model accuracy and no fidelity in the notebook.

So the upstream explanation is "the network, each of whose units is described by a rule", with the
prediction made by the network. That is R1 (R2 for the head). Whether the LogiX-GIN paper itself
reports a rule-model number is to be checked against the paper, not the code.

## 3. Measured gap (test split, fold 0)

`net`: network accuracy. R1: per-layer agreement of the unit rules with the network
(`rule_eval.rule_stats` `threshold_vs_network`, conv layers 0-2 then head). R2: accuracy and
agreement with the network. R3: chained bit agreement with the network's states (layers 0-2),
rule-model accuracy and fidelity.

| model | net | R1 per-layer agreement | R2 acc | R2 fid | R3 bits L0/L1/L2 | R3 acc | R3 fid |
|---|---|---|---|---|---|---|---|
| classic PROTEINS | 71.4 | 91.6 / 99.0 / 99.5 / 98.7 | 70.5 | 99.1 | 91.8 / 93.2 / 97.3 | 50.0 | 57.1 |
| classic MUTAG | 68.4 | 98.4 / 98.8 / 98.1 / 100 | 68.4 | 100 | 98.4 / 78.9 / 85.1 | 84.2 | 63.2 |
| classic BAMultiShapes | 99.0 | 97.2 / 97.2 / 94.9 / 100 | 99.0 | 100 | 97.1 / 87.7 / 88.4 | 84.0 | 85.0 |
| graph PROTEINS | 74.1 | 99.0 / 99.2 / 98.9 / 98.2 | 73.2 | 99.1 | 99.0 / 94.7 / 97.4 | 66.1 | 88.4 |
| NMP Mutagenicity | 80.6 | 97.3 / 99.2 / 98.4 / 86.2 | 75.3 | 76.7 | 97.4 / 94.6 / 95.6 | 69.1 | 71.0 |
| NMP + QAT Mutagenicity | 78.3 | 100 / 100 / 100 / 100 | 78.3 | 100 | 100 / 100 / 100 | 78.3 | 100 |

Reading:

- Under R1 every model looks faithful (92-100 % per layer; the classic head 99-100 %). Under R3
  the same classic models agree with their network on 57-85 % of the graphs. Small per-layer
  disagreements compound: on MUTAG, 98 % agreement per layer becomes 79 % of the layer-1 bits once
  layer 1 reads layer 0's rules instead of its network states.
- The NMP head is the one place where R1/R2 already lose: 86 % unit agreement, 77 % R2 fidelity.
  Its literals read prototype similarities pooled over nodes, graded values that the binarisation
  changes (see docs/GUMBEL_TEACHER.md §5 for why the trunk states are graded).
- R3 accuracy can exceed the network's (classic MUTAG, 84 vs 68): the rule model is a different
  classifier, not a degraded copy.
- QAT closes all three gaps at once, which is its point.

## 4. What our numbers use

| number | reading |
|---|---|
| `paper/table_interp.tex`: Rule acc., Fid., Lit., Bits, Protos, GT prec. (all models, classic included) | R3 |
| `paper/table_main*.tex`: accuracy, balanced accuracy, AUC | the network (for NMP + QAT = R3) |
| `paper/table_binarisation.tex`, `fig_binarisation`: fidelity, rule-model accuracy | R3 (`interp_metrics`, `qat_proto.diagnostics`) |
| `paper/table_hoyer.tex`: nonzero, covers/unit, shortest rule | the weights (reading-free) for nonzero and covers; R1 inputs for shortest/node and threshold_vs_network (`rule_eval.rule_stats`); fidelity and bits from R3 |
| MLflow `final/rules/*` of every run | R1 (`rule_eval.rule_stats` in `utils.training.register_final`) |
| QAT epoch selection (`qat/val_acc_rules`, `qat/val_balanced_acc_rules`) | R3 |
| `unpack_rules.py`, `explain_proto.py`, the component explanations of `utils/viz_proto.py` | R1 |
| explorer: forward pass, reasons, evidence | R3 (and R2 for "head only") |
| `utils/agreement.py`, prototype cards | R3 |

## 5. What to do

1. **Paper text.** Define the rule model as R3 once, in the metrics paragraph, and say that every
   explanation metric is computed on it, for every model. Say explicitly that this is stricter
   than the layer-local reading of the LogiX-GIN code (R1), in which each unit's rule is checked on
   the network's own inputs and the prediction stays the network's. Otherwise a reader may take the
   classic rows of `table_interp` (rule accuracy 50-61 % on PROTEINS) as a claim the LogiX-GIN
   authors made.
2. **Show both readings once.** `interp_metrics` now reports `head_acc` (R2) next to `head_fidelity`:
   a column or an appendix table with network / R2 / R3 accuracy per model shows how much of the
   gap is the head and how much the composition of the trunk rules.
3. **`table_hoyer`.** State that shortest/node is measured on the network's inputs (R1), or compute
   it on the R3 states for consistency with the other columns (a change to `rule_eval.rule_stats`:
   take `x_bin` from `interp_metrics.trunk_states` instead of `collect_activations`). Decision.
4. **Text tools (`unpack_rules.py`, `explain_proto.py`).** They print R1 rules, which is right for
   "what does this unit compute"; a caption that says "the model's rules" for an R1 printout of a
   non-QAT model should add that the composition of these rules is the R3 model, whose fidelity is
   in `table_interp`.

## 6. Reproduce

```bash
# R2 and R3 (head_acc, head_fidelity, logic_acc, logic_fidelity, trunk_bit_agreement)
python interp_metrics.py --run_path <final run dir> --split test
# R1 per-layer agreement: rule_eval.rule_stats(model, test_loader, device)[i]['threshold_vs_network'],
# also logged on val as MLflow final/rules/L<i>/threshold_vs_network (head: final/rules/head/...) by every training run
```

The §3 table was computed on the fold-0 final runs of `sum_ablation` (classic, graph),
`sum_ablation_ctrl` (NMP Mutagenicity) and `qat_nmp` (NMP + QAT), test split.
