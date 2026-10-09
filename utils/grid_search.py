"""Exhaustive grid search in random order, parallel with joblib (upstream optimize_*.py)."""
import itertools
import random

from joblib import Parallel, delayed


class CustomGridSearch:
    def __init__(self, param_grid, scoring_function, n_jobs=1):
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(self, dataset_name):
        # Generate all possible combinations of parameters
        param_combinations = list(itertools.product(*(self.param_grid[param] for param in self.param_grid)))
        random.shuffle(param_combinations)
        print('Total combinations:', len(param_combinations))
        
        # Wrapper function to evaluate a single combination of parameters
        def evaluate_params(params):
            param_dict = {param: params[i] for i, param in enumerate(self.param_grid)}
            print(param_dict)
            score = self.scoring_function(dataset_name, **param_dict)
            return param_dict, score

        # Parallel processing of parameter combinations
        results = Parallel(n_jobs=self.n_jobs)(delayed(evaluate_params)(params) for params in param_combinations)
        
        # Process results to find the best parameters and score
        for param_dict, score in results:
            self.results_.append({'params': param_dict, 'score': score})
            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = param_dict
        
        return self

    def get_results(self):
        return self.results_

