"""Grid search over the teacher's hyperparameters (upstream), graph or node task.

    python optimize_baseline.py --dataset PROTEINS             # train_baseline.py
    python optimize_baseline.py --task node --dataset Cora     # train_baseline_node.py
"""
import argparse
import functools
import importlib

from utils.grid_search import CustomGridSearch

GRAPH_PARAMS = {
    'nogumbel': [False, True],
    'epochs': [3000],
    'lr': [0.001, 0.01],
    'l2': [1e-4],
    'dropout': [0, 0.5],
    'num_layers': [5],
    'hidden_dim': [16, 32],
    'batch_size': [32, 128],
}
PARAMS = {'graph': GRAPH_PARAMS, 'node': {**GRAPH_PARAMS, 'num_layers': [3]}}
TRAIN_MODULE = {'graph': 'train_baseline', 'node': 'train_baseline_node'}
DEFAULT_DATASET = {'graph': 'PROTEINS', 'node': 'Cora'}


def scoring_function(task, dataset_name, **params):
    train = importlib.import_module(TRAIN_MODULE[task]).train
    return train(dataset_name, params)['val_acc_mean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_baseline.py')
    parser.add_argument('--task', default='graph', choices=['graph', 'node'], help='graph or node classification')
    parser.add_argument('--dataset', default=None, type=str, help='default: PROTEINS (graph), Cora (node)')
    parser.add_argument('--n_jobs', default=10, type=int, help='Number of jobs')
    args = parser.parse_args()

    grid_search = CustomGridSearch(PARAMS[args.task], functools.partial(scoring_function, args.task),
                                   n_jobs=args.n_jobs)
    grid_search.fit(args.dataset or DEFAULT_DATASET[args.task])
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
