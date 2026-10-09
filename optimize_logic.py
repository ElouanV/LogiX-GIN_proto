"""Grid search over the LogiX-GIN student's hyperparameters (upstream), graph or node task.

    python optimize_logic.py --dataset PROTEINS             # train_logic.py
    python optimize_logic.py --task node --dataset Cora     # train_logic_node.py
"""
import argparse
import functools
import importlib

from utils.grid_search import CustomGridSearch

GRAPH_PARAMS = {
    'warmup_epochs': [3000],
    'epochs': [5000],
    'lr': [0.0001, 0.001, 0.01],
    'l2': [0, 1e-4],
    'batch_size': [16, 32, 64, 128],
    'conv_reg': [0.1, 0.01, 0.001],
    'fc_reg': [0.1, 0.01, 0.001],
    'dropout': [0, 0.1, 0.3, 0.5],
}
NODE_PARAMS = {k: v for k, v in GRAPH_PARAMS.items() if k != 'dropout'}
NODE_PARAMS['batch_size'] = [32, 128]
PARAMS = {'graph': GRAPH_PARAMS, 'node': NODE_PARAMS}
TRAIN_MODULE = {'graph': 'train_logic', 'node': 'train_logic_node'}
DEFAULT_JOBS = {'graph': 10, 'node': 4}


def scoring_function(task, dataset_name, **params):
    module = importlib.import_module(TRAIN_MODULE[task])
    baseline_path = module.get_best_baseline_path(dataset_name)
    return module.train_eval(dataset_name, baseline_path, params)['val_acc_mean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_logic.py')
    parser.add_argument('--task', default='graph', choices=['graph', 'node'], help='graph or node classification')
    parser.add_argument('--dataset', default='PROTEINS', type=str, help='Dataset to use')
    parser.add_argument('--n_jobs', default=None, type=int, help='Number of jobs (default: 10 graph, 4 node)')
    args = parser.parse_args()

    grid_search = CustomGridSearch(PARAMS[args.task], functools.partial(scoring_function, args.task),
                                   n_jobs=args.n_jobs or DEFAULT_JOBS[args.task])
    grid_search.fit(args.dataset)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
