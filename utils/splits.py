"""Train/val/test splits of a graph dataset, indexed by seed.

``random`` is the upstream protocol: a stratified 80/20 split with ``random_state=seed``,
whose 20% is halved (unstratified) into val and test. ``kfold`` is stratified 10-fold
cross-validation: the seed is the fold index, fold ``seed`` is the test set, and a
stratified 1/9 of the nine other folds is the val set (so 80/10/10 as well). The folds
themselves are drawn once with ``random_state=0``, so seeds 0-9 cover every graph once.

Students (train_logic.py, train_proto.py) reuse the teacher's ``data.pkl``, so only the
teacher scripts call this.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

SPLITS = ('random', 'kfold')
N_FOLDS = 10


def split_indices(y, seed, split='random'):
    """(train, val, test) index lists of a dataset with labels ``y`` for one seed."""
    y = np.asarray(y).reshape(-1)
    indices = list(range(len(y)))
    if split == 'random':
        train, val_test = train_test_split(indices, test_size=0.2, shuffle=True, stratify=y, random_state=seed)
        return train, val_test[:len(val_test) // 2], val_test[len(val_test) // 2:]
    if split == 'kfold':
        if not 0 <= seed < N_FOLDS:
            raise ValueError(f'kfold split: seed is the fold index, 0..{N_FOLDS - 1}, got {seed}')
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        rest, test = list(folds.split(indices, y))[seed]
        train, val = train_test_split(rest.tolist(), test_size=1 / (N_FOLDS - 1), shuffle=True,
                                      stratify=y[rest], random_state=seed)
        return train, val, test.tolist()
    raise ValueError(f'unknown split {split!r}, expected one of {SPLITS}')
