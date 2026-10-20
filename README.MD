# On Logic-based Self-Explainable Graph Neural Networks - LogiX-GIN
Official repository of [LogiX-GIN](https://neurips.cc/virtual/2025/poster/118261), accepted at NeurIPS 2025.

To cite our work, please use:
```
@inproceedings{ragno2025,
  title={On Logic-based Self-Explainable Graph Neural Networks},
  author={Ragno, Alessio and Plantevit, Marc and Robardet, Celine},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

The environment details are available in the `environment.yml` file (the original
environment of the paper is `environment.upstream.yml`).

## Running the experiments (any machine)
The full protocol that every machine follows is `docs/EXPERIMENT_PROTOCOL.md`.
Experiments are YAML configs (`configs/experiments/*.yaml`; models in `configs/models.yaml`,
search spaces in `configs/search_spaces.yaml`). One machine owns a dataset: it trains the
teachers, runs the hyper-parameter studies and the final 10-fold runs, so nothing is copied.
```
git clone <this repository> && cd LogiX-GIN && git checkout <branch>
conda env create -f environment.yml && conda activate logix-gin
python scripts/check_setup.py --download
python run_experiment.py configs/experiments/sum_ablation.yaml run --datasets AIDS   # this machine's dataset(s)
python run_experiment.py configs/experiments/sum_ablation.yaml status
```
New dataset: add an entry to `configs/datasets.yaml` (how to load it, the names of its
node features and classes, its ground-truth region if it has one), commit its folds with
`python -m utils.splits export <dataset>`, then add it to an experiment YAML.

Splits: the indices of every k-fold split are committed in `splits/<dataset>_kfold.json`
and are the reference for every model, external baselines included
(`utils.splits.load_split(dataset, fold)`; `python -m utils.splits check` compares them
with a fresh draw and with the teachers).

Progress goes to `logs/progress/<host>.jsonl` and, with `NOTION_TOKEN` (an internal
integration's secret, the "Run progress" database shared with it) and
`NOTION_PROGRESS_DB=e0c0a6960cab48ac9de50515b6875c5f` set, to the shared Notion database
(`utils/progress.py`; `python -m utils.progress push` backfills it). MLflow:
tunnel to the main server (`ssh -N -L 5055:127.0.0.1:5055 <main host>`) so every run lands
in one store, or start a local `scripts/mlflow_server.sh`, or `LOGIX_MLFLOW=0`. Teachers
trained elsewhere can be moved with `python -m utils.teachers pack|unpack` (a retrained
teacher differs, so never mix teachers of two machines within one dataset).

## Training Models:
It is possible to train the models using the following commands
- Black-box models
`python train_baseline.py` for a single run or `python optimize_baseline.py` to do a grid search
- Logic models
`python train_logic.py` for a single run or `python optimize_logic.py` to do a grid search

## Analyzing results:
- `nbs/Results.ipynb` allows visualizing classification results
- `nbs/InstanceLevelExpl.ipynb` allows obtaining node level explanation and calculating fidelity
- `nbs/PatternAnalysis.ipynb` allows studying the effects of pruning
- `nbs/LayerWiseRules.ipynb` allows obtaining layerwise rules of the models

## Explorer: navigate a trained model
`explorer/` is an interactive web view of a trained LogiX-GIN (prototype or classic). It shows
the *symbolic* model exactly as `interp_metrics.py` scores it: every literal's step function and
neighbourhood count, every unit's threshold and its shortest sufficient reason at each node,
the prototypes' cared bits and matches, and the head's decision, with an animated forward pass.
It only reads checkpoints; nothing in a run directory is written.

```bash
python -m explorer serve --root results_final            # browse every run, http://127.0.0.1:8765/
python -m explorer serve --run "<seed dir>[/qat/<cfg>]"   # one run
python -m explorer export "<seed dir>" --split test --graphs 40   # self-contained HTML in explorer_exports/
```
The first open of a run measures literal ranges, firing rates, rule counts and prototype matches
on the training split (~10 s on Mutagenicity, CPU). `--ref_max N` caps that pass.

## Example:
The repository contains a simple example of a model trained on MUTAG with 2 layers using seed 0. This example was obtained by running the following commands: 
- `python train_baseline.py --dataset MUTAG --seed 0 --epochs 3000 --batch_size 128 --hidden_dim 32 --num_layers 2 --dropout 0.5 --l2 0.0001 --nogumbel --lr 0.001`
- `python train_baseline.py --dataset MUTAG --seed 0 --epochs 3000 --batch_size 32 --hidden_dim 32 --num_layers 2 --l2 0.0 --lr 0.01`
- `python train_logic.py --dataset MUTAG --seed 0 --epochs 5000 --batch_size 32 --l2 0.0 --lr 0.01 --fc_reg 0.01 --conv_reg 0.001`



