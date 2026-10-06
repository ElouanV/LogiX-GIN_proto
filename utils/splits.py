"""Train/val/test splits of a graph dataset, indexed by seed.

``random`` is the upstream protocol: a stratified 80/20 split with ``random_state=seed``,
whose 20% is halved (unstratified) into val and test. ``kfold`` is stratified 10-fold
cross-validation: the seed is the fold index, fold ``seed`` is the test set, and a
stratified 1/9 of the nine other folds is the val set (so 80/10/10 as well). The folds
themselves are drawn once with ``random_state=0``, so seeds 0-9 cover every graph once.

Students (train_logic.py, train_proto.py) reuse the teacher's ``data.pkl``, so only the
teacher scripts call this.

The k-fold indices are also saved, one file per dataset, in ``splits/<dataset>_kfold.json``
(committed): every model of the paper, external baselines included, must be evaluated on
exactly these folds. When the file exists it is the reference: ``split_indices(...,
dataset=...)`` reads it (after checking the labels it was made for) instead of drawing the
folds again, so a different sklearn or numpy cannot change them. When it is missing, the
first k-fold teacher writes it.

    python -m utils.splits export MUTAG PROTEINS     # write the files (from the labels)
    python -m utils.splits check                     # files vs a fresh draw and the teachers' data.pkl

A baseline loads a fold with ``load_split('MUTAG', 3)`` -> {'train': [...], 'val': [...],
'test': [...]}: indices into ``utils.utils.get_dataset('MUTAG')``.
"""
import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

SPLITS = ('random', 'kfold')
N_FOLDS = 10
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO, 'splits')
KFOLD_PROTOCOL = (f'StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state=0), test = fold k; '
                  'val = train_test_split(rest, test_size=1/9, stratify, random_state=k)')


def split_file(dataset, split='kfold'):
    return os.path.join(SPLIT_DIR, f'{dataset}_{split}.json')


def labels_digest(y):
    return hashlib.sha256(np.asarray(y).reshape(-1).astype(np.int64).tobytes()).hexdigest()


def load_split(dataset, fold, split='kfold', y=None):
    """Saved indices of one fold; with ``y``, first check they were made for these labels."""
    rec = json.load(open(split_file(dataset, split)))
    if y is not None:
        y = np.asarray(y).reshape(-1)
        if rec['n_graphs'] != len(y) or rec['labels_sha256'] != labels_digest(y):
            raise RuntimeError(f'{split_file(dataset, split)} was made for other labels than the loaded '
                               f'{dataset} ({rec["n_graphs"]} graphs, {len(y)} loaded)')
    return rec['folds'][fold]


def same_split(dataset, fold, data):
    """sha256 of the split file if ``data`` (a run's data.pkl dict) holds fold ``fold``'s
    indices, else RuntimeError."""
    saved = load_split(dataset, fold)
    for s in ('train', 'val', 'test'):
        if [int(i) for i in data[f'{s}_indices']] != saved[s]:
            raise RuntimeError(f'{dataset} fold {fold}: {s} indices of the run differ from {split_file(dataset)}')
    return hashlib.sha256(open(split_file(dataset), 'rb').read()).hexdigest()


def save_splits(dataset, y, split='kfold'):
    """Write every fold of ``dataset`` (atomically: parallel teachers may race)."""
    import sklearn
    y = np.asarray(y).reshape(-1)
    folds = []
    for k in range(N_FOLDS):
        tr, va, te = draw_split(y, k, split)
        folds.append({'fold': k, 'train': [int(i) for i in tr], 'val': [int(i) for i in va],
                      'test': [int(i) for i in te]})
    head = {'dataset': dataset, 'split': split, 'n_graphs': len(y), 'labels_sha256': labels_digest(y),
            'label_counts': {str(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))},
            'protocol': KFOLD_PROTOCOL, 'sklearn': sklearn.__version__}
    # one fold per line: readable header, diffable folds
    body = ',\n'.join('  ' + json.dumps(f, separators=(',', ':')) for f in folds)
    text = json.dumps(head, indent=1)[:-2] + ',\n "folds": [\n' + body + '\n ]\n}\n'
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = split_file(dataset, split)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def split_indices(y, seed, split='random', dataset=None):
    """(train, val, test) index lists of a dataset with labels ``y`` for one seed. A k-fold
    split of a named ``dataset`` comes from its saved file (written first if missing)."""
    if split == 'kfold' and dataset is not None:
        if not os.path.exists(split_file(dataset)):
            save_splits(dataset, y)
        if not 0 <= seed < N_FOLDS:
            raise ValueError(f'kfold split: seed is the fold index, 0..{N_FOLDS - 1}, got {seed}')
        f = load_split(dataset, seed, y=y)
        return f['train'], f['val'], f['test']
    return draw_split(y, seed, split)


def draw_split(y, seed, split='random'):
    """The split drawn from the labels (no saved file)."""
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


def check(datasets=None):
    """Each saved file against a fresh draw and against every k-fold teacher's data.pkl."""
    from utils.utils import get_dataset
    bad = 0
    files = sorted(glob.glob(os.path.join(SPLIT_DIR, '*_kfold.json')))
    for path in files:
        ds = json.load(open(path))['dataset']
        if datasets and ds not in datasets:
            continue
        y = get_dataset(ds).data.y
        for k in range(N_FOLDS):
            saved = load_split(ds, k, y=y)
            drawn = dict(zip(('train', 'val', 'test'), draw_split(y, k, 'kfold')))
            if any(list(saved[s]) != list(drawn[s]) for s in drawn):
                bad += 1
                print(f'{ds} fold {k}: saved indices differ from a fresh draw (other sklearn/numpy?)')
        for pkl in glob.glob(os.path.join(REPO, 'results', ds, '*split=kfold', '[0-9]*', 'data.pkl')):
            k = int(os.path.basename(os.path.dirname(pkl)))
            data, saved = pickle.load(open(pkl, 'rb')), load_split(ds, k)
            if any(list(data[f'{s}_indices']) != saved[s] for s in ('train', 'val', 'test')):
                bad += 1
                print(f'{pkl}: teacher trained on other indices than {path}')
        print(f'{ds}: checked')
    print('splits ok' if not bad else f'{bad} mismatch(es)')
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['export', 'check'])
    ap.add_argument('datasets', nargs='*')
    a = ap.parse_args()
    if a.action == 'export':
        from utils.utils import get_dataset
        for ds in a.datasets:
            print(save_splits(ds, get_dataset(ds).data.y))
    else:
        raise SystemExit(0 if check(a.datasets) else 1)


if __name__ == '__main__':
    main()
