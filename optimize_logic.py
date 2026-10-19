"""Grid search over the LogiX-GIN student's hyperparameters (upstream).

    python optimize_logic.py --dataset PROTEINS             # train_logic.py
"""
import argparse

from train_logic import train_eval
from utils.runs import best_teacher_dir
from utils.grid_search import CustomGridSearch

PARAMS = {
    'warmup_epochs': [3000],
    'epochs': [5000],
    'lr': [0.0001, 0.001, 0.01],
    'l2': [0, 1e-4],
    'batch_size': [16, 32, 64, 128],
    'conv_reg': [0.1, 0.01, 0.001],
    'fc_reg': [0.1, 0.01, 0.001],
    'dropout': [0, 0.1, 0.3, 0.5],
}


def scoring_function(dataset_name, **params):
    return train_eval(dataset_name, best_teacher_dir(dataset_name), params)['val_acc_mean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_logic.py')
    parser.add_argument('--dataset', default='PROTEINS', type=str, help='Dataset to use')
    parser.add_argument('--n_jobs', default=10, type=int, help='Number of jobs')
    args = parser.parse_args()

    grid_search = CustomGridSearch(PARAMS, scoring_function, n_jobs=args.n_jobs)
    grid_search.fit(args.dataset)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
