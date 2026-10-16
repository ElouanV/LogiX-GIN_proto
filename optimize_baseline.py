"""Grid search over the teacher's hyperparameters (upstream).

    python optimize_baseline.py --dataset PROTEINS             # train_baseline.py
"""
import argparse

from train_baseline import train
from utils.grid_search import CustomGridSearch

PARAMS = {
    'nogumbel': [False, True],
    'epochs': [3000],
    'lr': [0.001, 0.01],
    'l2': [1e-4],
    'dropout': [0, 0.5],
    'num_layers': [5],
    'hidden_dim': [16, 32],
    'batch_size': [32, 128],
}


def scoring_function(dataset_name, **params):
    return train(dataset_name, params)['val_acc_mean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_baseline.py')
    parser.add_argument('--dataset', default='PROTEINS', type=str, help='Dataset to use')
    parser.add_argument('--n_jobs', default=10, type=int, help='Number of jobs')
    args = parser.parse_args()

    grid_search = CustomGridSearch(PARAMS, scoring_function, n_jobs=args.n_jobs)
    grid_search.fit(args.dataset)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
