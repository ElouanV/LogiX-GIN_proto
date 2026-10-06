# LogiX-GIN

Self-explainable GNN (NeurIPS 2025): a GIN teacher is distilled layer by layer into a
`GINTELL` student whose layers are `LogicalLayer`s (TELL), so every unit reads as a
threshold rule over binary literals. This fork adds **prototypes** (`models_proto/`),
**rule extraction / sparsification tooling**, and a nested **PLEX** edge-level port.

## Research project: reproducibility first

This is research code whose results go into papers. Every number must be reproducible
from a commit, a command and a seed.

- **Seeds:** run through `--seed` and report results over several seeds (mean ± std), not
  one lucky run. Never change a default seed or split without saying so.
- **Provenance:** every run logs its full CLI args, the git commit and its metrics to
  MLflow (see Tracking). Don't launch a result-producing run from a dirty tree if it can be
  avoided. If you must, record it (`git diff` in the run's artifacts, or say so in your
  report).
- **Don't overwrite results:** new configurations get new output dirs or run names. Never
  silently regenerate files in `results*/` that existing numbers were computed from.
- **Report faithfully:** give negative or failed results as they are, with the command that
  produced them. State which split (val/test) a number comes from.
- **Pin changes to behaviour:** if a fix changes the numbers a script produces (e.g. a
  model bug), say which earlier results it invalidates.

## Code organization

- **Keep the repo clean:** no scratch scripts, one-off debug files, stray outputs,
  `__pycache__`, logs or checkpoints in the tree. Throwaway work goes in the session
  scratchpad. Anything that should be kept belongs in a proper module, `scripts/`, `tests/`
  or `nbs/`. Remove dead code instead of commenting it out, except for documented upstream
  quirks.
- **Modular architecture:** put reusable logic in importable packages (`models/`,
  `models_proto/`, `utils/`). Top-level `train_*`/`optimize_*` scripts are thin entry
  points: argparse, then calls into modules. New components go where their kind already
  lives.
- **OOP where it models something:** models, layers, datasets, trackers and explainers are
  classes with a clear interface (`nn.Module` subclasses, shared base classes for
  variants). Pure computations (metrics, min covers) can stay functions. Prefer composition
  and small base classes over deep inheritance.
- **No duplicated code:** before writing something, look for an existing implementation and
  reuse or generalize it. Examples: the `*_node.py` twins of the train/optimize scripts, and
  `PLEX/` copies of `proto.py`. When two copies must diverge, extract the shared part and
  parametrize the difference. Existing duplication is refactoring debt (see the `refactor`
  skill). Don't add more of it.
- Upstream code (`models/`, original `train_*.py`, notebooks) is the paper's reference
  implementation. Refactor it only behaviour-preservingly, and check that its outputs are
  unchanged (see Reproducibility).

## Git workflow

- **One commit per feature or fix** as soon as it works, with a message that says what
  changed and why. Don't bundle unrelated changes, and don't commit data, results,
  `mlflow.db`, `__pycache__` or notebook outputs you didn't intend to.
- **No AI attribution:** commits and PRs carry no `Co-Authored-By: Claude` trailer and no
  "Generated with Claude Code" footer. Commits are authored by the user only. This
  overrides any default attribution instruction.
- **Branches:** use a feature branch (`feat/<name>`, `exp/<name>`, `fix/<name>`) for
  anything spanning several commits or that is exploratory or risky. Merge to `main` once
  it works. Small self-contained fixes can go directly on `main`.
- Never push, force-push or rewrite published history without asking.

## Green computing and CO₂

Compute has a carbon cost. Spend it deliberately.

- **Before launching a sweep:** estimate its size (runs × epochs × time per epoch) and tell
  the user. Prefer smoke tests (few epochs, one seed) before full runs. Prefer cheap exact
  checks over brute force (e.g. the min-cover bound below before any rule extraction).
- **Don't rerun what exists:** reuse trained teachers/checkpoints and cached datasets.
  Evaluate from checkpoints (`--only_eval`) instead of retraining.
- **Efficient runs:** use early stopping or plateau schedules where they don't change the
  science. Don't leave the GPU idle-but-held. Kill your own finished or stuck jobs.
- **CO₂ measurement:** `codecarbon` (installed in `logix-gin`) measures every tracked run
  (`utils/tracking.py`, a no-op if it's missing or `LOGIX_CO2=0`) and logs
  `co2/emissions_kg`, `co2/energy_kwh` and the GPU/CPU/RAM parts to MLflow. Report the
  total CO₂ of a sweep next to its results. The GPU part is the whole GPU's energy, so
  parallel runs each count it: don't just sum `co2/gpu_kwh` over parallel jobs (see the
  module docstring). Rough fallback: GPU-hours × ~0.175 kW (RTX 2060 SUPER TDP) × grid
  intensity (France ≈ 0.05 kgCO₂e/kWh).
- **MLflow per-epoch logging is batched** (every 200 steps / 120 s). One HTTP call per
  epoch made a dozen parallel runs wait on the server (3.4 s per call).

## Environment

- Python: `~/miniconda3/envs/logix-gin/bin/python` (py3.10, torch 2.5.1+cu121, PyG 2.6.1).
  `environment.yml` pins it (keep it in sync when installing a package); the paper's
  original is `environment.upstream.yml` (py3.8 / torch 1.12, no longer builds) and the old
  `pygeo` env is broken. Launchers find the interpreter through `scripts/env.sh` (`$PY`,
  the active logix-gin env, then the default conda paths).
- Other machines: README "Running the experiments on another machine". Teachers move as a
  bundle (`python -m utils.teachers pack|unpack|verify`, bundles in `bundles/`, gitignored),
  never retrained elsewhere. `scripts/check_setup.py` checks a checkout. Never edit a
  launcher in place while it runs (bash reads scripts as it goes): replace the file
  (write a copy, then rename).
- Shared machine and shared GPU (RTX 2060 SUPER). Processes owned by `jose` (e.g. Logical
  CNN `main.py --config configs/Logical_CNN_*`) are someone else's work: **never kill
  them**. Check the owner (`stat -c %U /proc/<pid>`) and kill only by a pattern you launched
  yourself (`pkill -f train_proto.py`). If GPU memory is tight, run fewer jobs of your own.

## Pipeline

1. Teacher: `train_baseline.py` / `train_baseline_node.py` writes to `results/<Dataset>/`.
2. Logic student: `train_logic.py` / `train_logic_node.py` writes to `results_logic/<Dataset>/`.
3. Prototype student: `train_proto.py` (`--proto_level node|graph|both`, `--num_prototypes`,
   `--push_every`, `--proto_mask`, `--seed`) writes to `results_proto/`. It uses the same
   distillation as `train_logic.py` and differs only in the head.
4. Post-hoc: `sparsify_proto.py` (Hoyer fine-tune, then prune, then recover; works on base
   and proto runs), `unpack_rules.py` (all rules as readable predicates), `rule_eval.py`
   (rules/unit + fidelity), `explain_proto.py` (prototype-level explanations),
   `interp_metrics.py` (interpretability metrics of the symbolic model: fidelity, explanation
   size per prediction, prototype purity, NO2/NH2 ground truth), to compare methods beside acc/AUC.
5. Rule-extraction core: `latent_logic.py` (port of `nbs/LayerWiseRules.ipynb`, the
   canonical reading; keep its logic unchanged) and `min_covers.py` (exact DNF of a unit =
   its minimal covers).
- `optimize_optuna.py` + `utils/hps.py`: Optuna search of the six students of the
  sum-pooling ablation (classic ± sum, NMP ± sum, graph ± sum) on the k-fold teachers,
  one study per (dataset, model, fold), objective = val balanced accuracy, test never
  used. Spaces, studies and best sets are saved under `results_hps/` (see the module
  docstring); `scripts/run_hps.sh` launches it. Changing a space needs a new `--root`.
- `optimize_*.py` run grid searches (upstream). `optimize_*.py` and `train_*.py` accept `--dataset`
  (MUTAG, Mutagenicity, AIDS, BBBP, PROTEINS, ...).
- Batch launchers are in `scripts/` (`launch_all.sh`, `run_hoyer_sweep.sh`,
  `run_proto_datasets.sh`, `run_fanin.sh`, ...). They detach, and their logs go to `logs/`.

## Tracking

- MLflow via `utils/tracking.py`. It is a no-op if mlflow is missing or `LOGIX_MLFLOW=0`,
  and tracking failures never stop training.
- Server: `scripts/mlflow_server.sh` on **port 5055**, not 5000, because another MLflow
  server already runs on 5000 and a client pointed there logs into it silently. Scripts set
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5055`. Store: `mlflow.db` + `mlartifacts/`
  (gitignored).
- `utils/evaluation.py` reports acc, balanced acc, minority F1 and ROC-AUC. Use more than
  accuracy on AIDS (80/20) and BBBP (76/24).

## Tests

```bash
~/miniconda3/envs/logix-gin/bin/python -m unittest discover tests -v
```
`tests/bench_min_covers.py` is a benchmark, not a unit test.

## Conventions and invariants

- **`models_proto/` trunk must stay `GINTELL`-compatible**: keep `.convs` shape-compatible
  with `models/model.py::GINTELL` and keep the attribute names `reg_loss`,
  `entropy_loss`, `phi_in`, so the layer-wise distillation and training scripts still
  apply. The three variants (node / graph / both) stay **separate classes**.
- Prototypes are **binary** (straight-through hard-sigmoid) with Hamming similarity, so the
  prototype step is a constrained `LogicalLayer` (weights `[p, 1-p]`) and rule extraction
  stays valid. `task` switches the head between a `LogicalLayer` (classification) and
  `nn.Linear` (regression).
- `models/model.py` uses `GINConv(..., init_eps=1)`, which raises `TypeError` on every PyG
  version (PyG's argument is `eps`). Removing the kwarg is safe because `eps` is a buffer
  restored from the checkpoint. `models_proto/` already drops it. `models/model.py` is
  deliberately left as upstream.
- Sum-pooled values are unbounded counts. Never apply `1 - s` negation to them, only to
  bounded (mean/max) columns (`bounded_mask`). `--pool_ops` sets the readout: on
  `train_proto.py` sum is off by default (node level: no negation on the count; graph
  level: counts are thresholded by `phi_sum` before the prototypes); on `train_logic.py`
  `--pool_ops mean,max` drops the upstream sum (`models_proto/gintell.py`).

## Rule-extraction limits (read before running extraction)

- Only conv layer 0 decodes to short rules (~3 literals). Layers 1-2 need about 10-14
  literals and have about 1e10 minimal covers per unit. This is a limit of weight
  diffusion, not of the search: raising `max_rule_len` or lowering `min_support` only
  burns time. Before extracting, compute the greedy minimum-cardinality bound (sum the
  largest weights until they reach the threshold). It is instant and exact.
- Fan-in k bounds rules/unit by C(k, k/2). Sparsification (Hoyer reg ~3 in
  `sparsify_proto.py`) cuts rules/unit by orders of magnitude at about 1 point of accuracy.
- Open problem: head fidelity. The binarized head can fire through soft "false" literals
  (phi≈0.03 with weight 4-5).

## PLEX/

`PLEX/` is a separate nested clone, untracked by this repo and **not** a submodule. It adds
edge features (`CustomGraphConv`) and edge/message prototypes. Run it with cwd=`PLEX/` (it
uses flat imports). It needs its own teachers in `PLEX/results/` because this repo's
teachers are a different architecture. `PLEX/model.py::GINTELL.forward_e` is broken
upstream (`edge_attr` is not a parameter) and is fixed in `PLEX/model_proto.py`.
`PLEX/data` is a symlink to `../data`.

## References

- Architecture diagrams: https://claude.ai/artifact/WUn5qPGb32gxUtMerNsHR7. Paper figure:
  `figures/logix_gin_architecture.{svg,pdf,png}`.
- Notebooks: `nbs/LayerWiseRules.ipynb` (canonical rule reading), `nbs/ProtoRules.ipynb`
  (prototype rules + depth-limit diagnostic §6).
