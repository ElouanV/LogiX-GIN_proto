# Experimental protocol (every machine)

This document is for whoever runs the paper's experiments on a machine other than the
main one, most likely an AI coding agent. Follow it exactly: results from different
machines go into the same tables, so every machine must run the same code, data, splits,
teachers, search and evaluation. If something here cannot be followed, stop and ask the
user. Do not work around it.

`CLAUDE.md` (repository conventions) also applies. Where the two differ, this document
wins for running experiments.

## 1. Rules

**Must:**
1. Run only the datasets the user assigned to this machine, and run every stage of each
   one on this machine: the Optuna studies, the 10 final folds and the summary.
2. Run from a clean checkout of the commit the user gave you. The pipeline refuses to
   start if tracked files are modified.
3. Use the main machine's teachers (the teacher bundle, §3.4).
4. Use the committed fold indices in `splits/<dataset>_kfold.json`. They are read
   automatically.
5. Keep every output. Send the results back with `pack` (§6).
6. Report negative and failed results as they are.

**Must not:**
1. Edit code, configs (`configs/**/*.yaml`), search spaces, seeds, splits or `n_trials`,
   even to "fix" a poor result or a crash. A crash is reported, not patched (§5).
2. Retrain teachers, unless the user explicitly asks for it for a given dataset. A
   retrained teacher differs (GPU nondeterminism), so students of one dataset must never
   mix teachers from two machines.
3. Regenerate `splits/*.json`, or run `python -m utils.splits export`.
4. Delete or overwrite anything in `results*/`, `bundles/` or `logs/`.
5. Kill processes you did not start (the GPU may be shared), or push to git.
6. Re-run a finished unit "to check". Units resume from disk, and a finished one is
   final.

The one parameter you may choose is `--jobs`, the number of units run in parallel (§4).
It changes speed and memory use, not results: each study runs its trials one after the
other with a seeded sampler, and pruning depends on epochs, not time.

## 2. What to get from the user before starting

| Item | Example | Why |
|---|---|---|
| Repository URL, branch and **commit** | `exp/paper-experiments` @ `<sha>` | every machine runs the same code |
| Datasets assigned to this machine | `AIDS BBBP` | one machine owns a dataset |
| Teacher bundle file | `teachers_kfold.tar.gz` (~15 MB) | identical teachers everywhere |
| Notion token (optional) | `NOTION_TOKEN=secret_...` | shared progress table |
| MLflow choice | SSH tunnel to the main machine, or local, or off | run tracking |

## 3. Setup

Run all commands from the repository root.

### 3.1 Code

```bash
git clone <repository URL> LogiX-GIN && cd LogiX-GIN
git checkout <commit>
git status --short --untracked-files=no            # must print nothing
```

### 3.2 Environment

The environment is pinned in `environment.yml`: Python 3.10, torch 2.5.1+cu121,
PyG 2.6.1, optuna 5.0.0, codecarbon, mlflow.

```bash
conda env create -f environment.yml
conda activate logix-gin
```

Do not upgrade or add packages. If the environment cannot be built on this machine (for
example an incompatible CUDA driver), stop and tell the user.

### 3.3 Datasets and checks

```bash
python scripts/check_setup.py --download --datasets <your datasets>
```

Every line must be `ok`, except the teacher lines (fixed in §3.4) and the MLflow line,
which is optional. A package version that differs from `environment.yml` is a `FAIL`:
fix the environment, never the pin.

Then check the splits:

```bash
python -m utils.splits check <your datasets>
```

It must end with `splits ok`. A mismatch means the dataset files differ from the main
machine's (other download, other preprocessing): stop and report it.

### 3.4 Teachers

```bash
python -m utils.teachers unpack <path>/teachers_kfold.tar.gz
```

This extracts the 10 teachers of each dataset, rebuilds their `data.pkl` from the
committed splits, and evaluates every teacher. It must end with `teachers verified`.

A `MISMATCH` line means a teacher does not reproduce its recorded val/test accuracy on
this machine. Stop and report the lines. Do not retrain.

Run `python scripts/check_setup.py --datasets <your datasets>` again: the teacher lines of your datasets must now
read `10/10 folds`.

### 3.5 Tracking (optional, never blocks a run)

- **MLflow**, one of:
  - a tunnel to the main machine (`ssh -N -L 5055:127.0.0.1:5055 <main host>`, then
    `export MLFLOW_TRACKING_URI=http://127.0.0.1:5055`);
  - or a local server (`scripts/mlflow_server.sh`).

  Training doesn't need MLflow, but the per-run CO₂ figures are stored only there, so
  don't turn it off (`LOGIX_MLFLOW=0`) unless the user agrees.
- **Notion**:
  ```bash
  export NOTION_TOKEN=<secret>
  export NOTION_PROGRESS_DB=e0c0a6960cab48ac9de50515b6875c5f
  ```
  Every unit's status then appears in the "Run progress" database of the page
  "LogiX-GIN prototypes: paper experiments". Without these variables, progress is only
  written to `logs/progress/<host>.jsonl`.
- **CO₂** is measured automatically by codecarbon. Leave it on (`LOGIX_CO2=0` would turn
  it off). The carbon intensity is France's by default: if the machine is in another
  country, `export LOGIX_CO2_COUNTRY=<ISO 3166 alpha-3 code>` (for example `ITA`) before
  running, and report it.

## 4. Running

```bash
python run_experiment.py configs/experiments/sum_ablation.yaml run --datasets <D1> <D2> --dry
```

`--dry` lists the units left to run and prints any preflight problem. With the teachers
unpacked, no `teacher/...` unit should be listed: if some are, the bundle is incomplete,
so stop and report it.

Then run, detached so it survives the session:

```bash
setsid nohup python run_experiment.py configs/experiments/sum_ablation.yaml run \
    --datasets <D1> <D2> --jobs <J> > logs/sum_ablation_<host>.log 2>&1 < /dev/null &
```

- **Choosing `--jobs`.** One unit uses about 0.3–1 GB of GPU memory. 6 jobs fit on an
  8 GB GPU. Start with `floor(free GPU GB / 1.2)`, at most 8. If you see CUDA out of
  memory errors, stop the run (kill the PIDs you started), lower `--jobs`, and start it
  again: finished units are skipped and an interrupted study resumes.
- **Monitoring:**
  - `python run_experiment.py configs/experiments/sum_ablation.yaml status --datasets <D1> <D2>`
  - the run log;
  - per-unit logs in `logs/sum_ablation/<D>_<stage>_<model>_fold<k>.log`;
  - the Notion table.
- **What the pipeline does for each dataset** (all fixed by the YAML, listed so you can
  check it):
  1. Teachers: already present from the bundle.
  2. One Optuna study per model on fold 0:
     - 25 trials, TPE sampler with seed 0, median pruner;
     - objective: validation balanced accuracy;
     - the test set is never used to choose anything;
     - output in `results_hps/<D>/<model>/fold0/`.
  3. Each of the 10 folds retrained with the best setting of its model's study, in
     `results_final/sum_ablation/<D>/<model>/fold<k>/final.json`. Each final run checks
     that it used the committed indices of its fold.
  4. `summary.csv`: mean ± std over the folds.
- **Expected duration** on one RTX 2060-class GPU, 6 jobs:

  | Datasets | Duration |
  |---|---|
  | MUTAG | ~5 h |
  | PROTEINS, BaMultiShapes, BA2Motifs | ~1 day each |
  | AIDS, BBBP | ~2 days each |
  | NCI1, Mutagenicity | ~4 days each |

- **Disk:** up to ~5 GB per dataset (each trial stores a copy of its data split).

## 5. When something goes wrong

| Situation | What to do |
|---|---|
| A unit is `FAILED` in the run log | Read its unit log. Report the unit key and the end of the traceback to the user. Do not change code or config. Units that depend on it are `skipped`; the others go on. |
| CUDA out of memory | Lower `--jobs` and start again (§4). |
| Machine rebooted or run killed | Run the same command again; it resumes. Studies mark the interrupted trial as failed and continue. |
| A model scores at chance (val balanced acc 0.5, one class predicted) | This is a result, not a failure. Let the study continue and report it. |
| Preflight refuses to start | Fix the cause: a modified tracked file (`git status`), or a missing or mismatching split. Never use `--allow_dirty` for real runs. |
| You think the protocol is wrong | Ask the user. Do not deviate. |

## 6. Sending the results back

When `status` shows every unit of your datasets done:

```bash
python run_experiment.py configs/experiments/sum_ablation.yaml summary --datasets <D1> <D2>
python run_experiment.py configs/experiments/sum_ablation.yaml pack --datasets <D1> <D2>
```

`pack` writes `bundles/sum_ablation_<datasets>_<host>.tar.gz`. It contains the studies
(journal, trials, best set), the final runs (checkpoint, arguments, metrics), the
teacher summaries, the unit logs, this host's progress log and the config snapshots. It
leaves out `data.pkl` files and trial checkpoints, which can be rebuilt or aren't
needed. Give the file to the user; on the main machine they run `run_experiment.py ...
unpack <file>`.

Then report to the user:
- the machine (hostname and GPU) and the commit you ran;
- the datasets you ran, the `summary` tables, and the total CO₂ (sum of the MLflow metric
  `co2/emissions_kg` over this machine's runs) with the `LOGIX_CO2_COUNTRY` used;
- every failed or skipped unit, with its error;
- anything that differed from this document.

## 7. Current status of the datasets

- **BA2Motifs:** on hold. 4 of its 10 teachers did not train with the shared teacher
  settings. Do not run it until the user says its teachers are fixed and a new teacher
  bundle is provided.
- **The other 7 datasets:** ready, with teachers in `teachers_kfold.tar.gz`.
