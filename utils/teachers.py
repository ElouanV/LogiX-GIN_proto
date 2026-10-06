"""Move trained teachers between machines.

Students are distilled from a fixed teacher per fold, so every machine running part of
a study must use the same teachers: retraining them elsewhere gives different
checkpoints (GPU nondeterminism) and incomparable results. A bundle holds, per teacher
configuration, ``args.json``, ``results.json`` / ``total_results.csv`` and each seed's
``args.json`` + ``best.pt`` (~250 KB). ``data.pkl`` (the bulk: pickled datasets) is not
shipped: it is rebuilt on unpack from the split (utils/splits.py), and the rebuilt
indices are checked against the originals at pack time. After unpacking, ``verify``
evaluates every teacher and compares val / test accuracy with total_results.csv.

    python -m utils.teachers pack teachers_kfold.tar.gz            # every split=kfold teacher
    python -m utils.teachers unpack teachers_kfold.tar.gz          # on the other machine
    python -m utils.teachers verify                                # (unpack already verifies)
"""
import argparse
import glob
import io
import json
import os
import pickle
import tarfile

import pandas as pd

PATTERN = 'results/*/*split=kfold'
SEED_FILES = ('args.json', 'best.pt')
CONFIG_FILES = ('results.json', 'total_results.csv')


def teacher_configs(pattern=PATTERN):
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def _split_of(config_dir):
    cfg = os.path.basename(config_dir)
    return 'kfold' if 'split=kfold' in cfg else 'random'


def rebuild_data(config_dir, seed):
    """data.pkl of one teacher seed, as train_baseline.py writes it."""
    from utils.splits import split_indices
    from utils.utils import get_dataset
    dataset = get_dataset(config_dir.split(os.sep)[-2])
    tr, va, te = split_indices(dataset.data.y, seed, _split_of(config_dir), config_dir.split(os.sep)[-2])
    return {'train_indices': tr, 'val_indices': va, 'test_indices': te,
            'train_dataset': dataset[tr], 'val_dataset': dataset[va], 'test_dataset': dataset[te]}


def seeds_of(config_dir):
    return sorted(int(os.path.basename(s)) for s in glob.glob(os.path.join(config_dir, '[0-9]*'))
                  if os.path.exists(os.path.join(s, 'best.pt')))


def pack(out, pattern=PATTERN):
    configs = teacher_configs(pattern)
    with tarfile.open(out, 'w:gz') as tar:
        for c in configs:
            for f in CONFIG_FILES:
                if os.path.exists(os.path.join(c, f)):
                    tar.add(os.path.join(c, f))
            for seed in seeds_of(c):
                d = os.path.join(c, str(seed))
                old = pickle.load(open(os.path.join(d, 'data.pkl'), 'rb'))
                new = rebuild_data(c, seed)
                for k in ('train_indices', 'val_indices', 'test_indices'):
                    if list(old[k]) != list(new[k]):
                        raise RuntimeError(f'{d}: {k} cannot be rebuilt from the split, not packing')
                for f in SEED_FILES:
                    tar.add(os.path.join(d, f))
        manifest = json.dumps({'configs': configs, 'seeds': {c: seeds_of(c) for c in configs}}, indent=1).encode()
        info = tarfile.TarInfo('teachers_manifest.json')
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    print(f'{out}: {len(configs)} teacher configurations')


def unpack(bundle, overwrite=False):
    with tarfile.open(bundle, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('teachers_manifest.json'))
        for m in tar.getmembers():
            if m.name == 'teachers_manifest.json':
                continue
            if not m.name.startswith('results/') or '..' in m.name:
                raise RuntimeError(f'unexpected path in bundle: {m.name}')
            if os.path.exists(m.name) and not overwrite:
                raise RuntimeError(f'{m.name} exists; pass --overwrite to replace it')
        tar.extractall(filter='data')
    for c, seeds in manifest['seeds'].items():
        for seed in seeds:
            with open(os.path.join(c, str(seed), 'data.pkl'), 'wb') as f:
                pickle.dump(rebuild_data(c, seed), f)
    return verify(manifest['configs'])


def verify(configs=None, tol=1e-9):
    """Evaluate every teacher seed and compare with its total_results.csv."""
    import torch
    from torch_geometric.loader import DataLoader
    from models.model import GIN
    from train_baseline import test_epoch
    from utils.utils import set_seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bad = 0
    for c in configs or teacher_configs():
        rec = pd.read_csv(os.path.join(c, 'total_results.csv')).set_index('seed')
        for seed in seeds_of(c):
            d = os.path.join(c, str(seed))
            a = json.load(open(os.path.join(d, 'args.json')))
            data = pickle.load(open(os.path.join(d, 'data.pkl'), 'rb'))
            nf = data['train_dataset'].num_features or 10
            # the teacher samples Gumbel noise even at test time: replay eval_seed's RNG sequence
            set_seed(seed)
            model = GIN(num_features=nf, num_classes=data['train_dataset'].num_classes, hidden_dim=a['hidden_dim'],
                        num_layers=a['num_layers'], nogumbel=a['nogumbel']).to(device)
            model.load_state_dict(torch.load(os.path.join(d, 'best.pt'), map_location=device))
            got = {s: test_epoch(model, DataLoader(data[f'{s}_dataset'], batch_size=64), device) for s in ('val', 'test')}
            diff = max(abs(got[s] - rec.loc[seed, f'{s}_acc']) for s in got)
            if diff > tol:
                bad += 1
                print(f'MISMATCH {d}: {got} vs recorded val {rec.loc[seed, "val_acc"]}, test {rec.loc[seed, "test_acc"]}')
    print('teachers verified' if not bad else f'{bad} teacher(s) do not reproduce their recorded accuracy')
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['pack', 'unpack', 'verify'])
    ap.add_argument('bundle', nargs='?')
    ap.add_argument('--pattern', default=PATTERN, help='teacher configuration dirs to pack')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if a.action == 'pack':
        pack(a.bundle, a.pattern)
    elif a.action == 'unpack':
        raise SystemExit(0 if unpack(a.bundle, a.overwrite) else 1)
    else:
        raise SystemExit(0 if verify() else 1)


if __name__ == '__main__':
    main()
