# Experiments: what is run, what remains

An index of the paper's experiments. The **paper is the source of truth** for what remains
to run: every table and figure in `paper/` (`table_<name>.tex`, `figures/fig_<name>.tex`)
starts with a comment block giving its question, protocol, source runs, `STATUS` and
`TODO`. This file summarises those blocks and maps them to the experiment configs; when the
two disagree, the `.tex` block wins. Update both when a run finishes.

```bash
grep -rn "STATUS:" paper/          # where each table / figure stands
grep -rn '\\prelim\|\\tbd' paper/  # cells still preliminary (grey) or not run (red dash)
```

How to run anything here: `README.md` and, on any machine other than the main one,
**`docs/EXPERIMENT_PROTOCOL.md` (binding)**. Conventions: `CLAUDE.md`.

## 1. Protocol shared by every experiment

- **Datasets** (`configs/datasets.yaml`, `paper/table_datasets.tex`): MUTAG, Mutagenicity,
  AIDS, NCI1, BBBP, PROTEINS, BA2Motifs, BAMultiShapes. Ground truth for the explanations:
  Mutagenicity (NO2 / NH2), BA2Motifs and BAMultiShapes (planted motifs).
- **Splits:** stratified 10-fold CV, fold indices committed in `splits/<dataset>_kfold.json`,
  fold k = seed k. Shared by every model, teacher and baseline.
- **Teachers:** one GIN (3 × 64, 500 epochs) per fold, trained once on the main machine and
  shared through the teacher bundle (`python -m utils.teachers pack|unpack|verify`). Never
  retrained on another machine.
- **Students** (`configs/models.yaml`): `classic`, `classic_nosum` (LogiX-GIN head),
  `graph` (graph-level prototypes), `node_mask_push` (NMP: node prototypes with care mask
  and push), `node_mask_push_qat` (NMP + Quant-Noise QAT fine-tune, `docs/QAT.md`), and
  variants (`_sum`, `_p8`, `_p16`, `_bounded`, `_predicate`).
- **Search:** Optuna, one study per (dataset, model) on fold 0, objective = val balanced
  accuracy (of the rule model for QAT models), test never used. Spaces in
  `configs/search_spaces.yaml`, studies under `results_hps/`. Controlled ablations skip the
  search: `params_from` (a model takes another model's best set) or `best_sets` (a hand set).
- **Selection:** every run keeps the epoch with the best validation balanced accuracy
  (`--select balanced_acc`; QAT: `select_metric`, on the rule model), the objective of the
  search; tables report accuracy and balanced accuracy. Competitors must select the same way.
- **Distillation:** every student, classic included, distills from the frozen teacher in eval
  mode (`--teacher_eval`). Both are fixed in `configs/models.yaml` since 2026-10-22; runs
  from before kept the best accuracy and distilled through the teacher's dropout.
- **Evaluation:** `interp_metrics.py` on every final fold (`interp.json`): rule-model
  accuracy, fidelity, explanation size (evidence literals and bits), prototypes cited,
  ground-truth precision and its chance level. Test split, mean ± std over folds. Paired
  comparisons: two-sided Wilcoxon over folds.
- **Cost:** codecarbon logs `co2/*` per run to MLflow. Estimate before launching, report
  after.

One experiment = one YAML in `configs/experiments/`:

```bash
python run_experiment.py configs/experiments/<exp>.yaml run    --datasets <D...>
python run_experiment.py configs/experiments/<exp>.yaml status --datasets <D...>
```

Outputs: `results_final/<exp>/<dataset>/<model>/fold<k>/` and `summary.csv` per dataset.

## 2. Experiments run so far

| Config | Question | Datasets (folds) | State | Feeds |
|---|---|---|---|---|
| `sum_ablation` | Accuracy of every student, with and without a sum readout, each tuned | MUTAG, PROTEINS, BAMultiShapes (10) | done, **before the head-saturation fix** (2026-10-19) | `table_main*`, `table_interp` (grey) |
| `sum_ablation_ctrl` | Same pairs at equal hyper-parameters (`params_from`): does sum help? | PROTEINS, BAMultiShapes, Mutagenicity (10) | done, before the fix | `table_sum_ablation`, `fig_pareto`, `fig_rule_complexity` |
| `qat_nmp` | NMP + QAT tuned jointly: does the rule model keep the accuracy? | Mutagenicity (10) | done | `table_main`, `table_interp`, `table_binarisation`, `fig_binarisation` |
| `qat_nmp_3fold` | Same on more datasets, cheaper (15 trials, folds 0-2) | PROTEINS, BAMultiShapes (3) | done | `table_main` (^3) |
| `qat_nmp_p8_3fold` | How many prototypes are needed? K ∈ {4, 8} | BAMultiShapes (3) | done (picked K = 8) | `table_num_prototypes` |
| `qat_nmp_p16_3fold` | K ∈ {8, 16} | Mutagenicity (3) | done (picked K = 8) | `table_num_prototypes` |
| `predicate_readout_3fold` | Interpretability issues 4 + 5 (`docs/INTERPRETABILITY_ISSUES.md`): threshold head literals (`_bounded`), and a head reading how many nodes satisfy each prototype rule (`_predicate`), at the K = 8 best sets | Mutagenicity, BAMultiShapes (3) | **running** since 2026-10-22 on the main machine | not in the paper yet |

## 3. Experiments that remain (from the paper's STATUS / TODO blocks)

Rough priority: E1 first, since most tables are filled from its final folds.

| Id | Table / figure | What to run | Needs |
|---|---|---|---|
| **E1** | `table_main`, `table_main_full`, `table_interp`, every `fig_*` | Final main experiment (`configs/experiments/main.yaml`, hps root `results_hps_e1`): `classic`, `classic_nosum`, `graph`, `node_mask_push` (+ the QAT model once decided) with the current `configs/models.yaml` (head fix, teacher in eval mode, balanced-accuracy selection) on all 8 datasets, 25 trials on fold 0, then 10 folds + interp. | A new YAML (e.g. `configs/experiments/main.yaml`); a decision on K and on the readout (`table_num_prototypes`, `predicate_readout_3fold`); BA2Motifs teacher fix |
| E1-teacher | `table_main` | BA2Motifs teacher: 4/10 folds at chance (Gumbel init trap). Retrain BA2Motifs only (restarts with `--init_seed`, or a per-dataset teacher config). | Decision |
| E1b | `table_main` | Competitors (3-4 of ProtGNN, PIGNN, GIB, GELATO, ...; post-hoc explainers on the teacher for GT) on the same folds and the same search budget | Choice of baselines, code |
| E5 | `table_nmp_components` | Cumulative ablation of NMP: dense node prototypes → + push → + care mask (= NMP) → + vocab_reg → + QAT; plus head fix on/off. One E1 best set per dataset, 10 folds. MUTAG, PROTEINS, Mutagenicity (+ BAMultiShapes for GT). ~150 runs. | Model entries + YAML (`params_from`) |
| K | `table_num_prototypes` | Decide the K policy (small fixed K, or K in the search); if kept, 10 folds for the chosen rows | Decision |
| E6 | `table_binarisation`, `fig_binarisation` | Every binarisation arm (none, sharpen during training, sharpen FT, hard FT, QAT fixed / tuned) on the 10 folds of the final E1 NMP students (post-hoc fine-tunes from the checkpoints, no retraining). Mutagenicity, PROTEINS, BAMultiShapes. | E1 |
| E7 | `table_hoyer`, `fig_rule_complexity` | Layer-wise Hoyer sparsification (`sparsify_proto.py`) as a post-hoc stage on the final E1 NMP and NMP + QAT folds, with rule counts. Guard `unpack_rules` with the min-cover bound first (63 GB blow-up on Mutagenicity fold 5). | E1, post-hoc stage in the YAML |
| Sanity | `table_sanity` | `sanity_randomisation.py --run_path results_final/<exp>/<ds>/<model>/fold*` on the final E1 node-prototype and classic folds with a GT region (Mutagenicity, BA2Motifs, BAMultiShapes). CPU, ~10 min per model. Decide on the data-randomisation test. | E1 |
| E2 | `table_sum_ablation` | Decide: keep (pairs are fair, both arms pre-fix) or rerun with the fixed models (~80 runs, ≤ 1 kgCO2e) | Decision |
| E9 / E10 | `table_sensitivity` (appendix) | E10: vocab_reg ∈ {0, 0.01, 0.1, 0.3} on the E1 NMP set, 10 folds, Mutagenicity + PROTEINS. E9: trunk width 32 (needs width-32 teachers). E10 first. | E1, YAML with per-row overrides |
| Q4 | `fig_prototypes`, `fig_explanation`, `fig_agreement` | Redraw from the final E1 folds (compact prototype cards; evidence-only head box) | E1 |
| Cost | `4_experiments.tex` (Implementation) | Total GPU-hours and kgCO2e of all experiments, from MLflow `co2/*` | All of the above |
| Audit-2 | `table_interp` | Explanations are sufficient for the predicted class's rule to fire, not for the argmax decision (both rules fire on 4-54% of test graphs): report the share or make them contrastive | Decision |
| Audit-7 | `table_hoyer` | Rule counts are upper bounds (unsatisfiable covers): drop literals never active on the training set before counting | Code |
| Audit-10 | `4_experiments.tex` | Text: teacher and students use AdamW, not Adam | Later |
| Teacher | `table_main` | Teacher accuracy is one random draw (`docs/GUMBEL_TEACHER.md`): choose the mean over draws or the most-likely-bit teacher | Decision |

## 4. Running on another machine

Follow `docs/EXPERIMENT_PROTOCOL.md`: clean checkout of the commit the user gives, the
teacher bundle from the main machine, only the datasets assigned to that machine, every
stage of a dataset on the same machine, results sent back with `pack`. One machine owns a
dataset: the same (experiment, dataset) is never run on two machines.
