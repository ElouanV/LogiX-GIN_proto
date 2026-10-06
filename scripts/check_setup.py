"""Check that this checkout can run the experiments (a new machine, after git clone).

    python scripts/check_setup.py              # environment, GPU, datasets, teachers
    python scripts/check_setup.py --download   # also download / process every dataset

Checks: package versions against environment.yml, CUDA, that each dataset of the
k-fold study loads with [0,1] node features, that the k-fold teachers are present
(python -m utils.teachers unpack <bundle> brings them), and whether an MLflow server
answers at $MLFLOW_TRACKING_URI (tracking is optional: runs continue without it).
"""
import argparse
import importlib
import importlib.metadata
import os
import re
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

DATASETS = ['MUTAG', 'PROTEINS', 'BaMultiShapes', 'BA2Motifs', 'AIDS', 'BBBP', 'NCI1', 'Mutagenicity']
PKG = {'torch': 'torch', 'torch-geometric': 'torch_geometric', 'numpy': 'numpy', 'scikit-learn': 'sklearn',
       'pandas': 'pandas', 'mlflow': 'mlflow', 'optuna': 'optuna', 'codecarbon': 'codecarbon', 'rdkit': 'rdkit'}

ok = True


def report(good, msg):
    global ok
    ok &= bool(good)
    print(('  ok   ' if good else '  FAIL ') + msg)


def pinned():
    pins = {}
    for line in open(os.path.join(REPO, 'environment.yml')):
        m = re.match(r'\s*-\s*([A-Za-z0-9_.-]+)==(\S+)', line)
        if m:
            pins[m[1].lower()] = m[2]
    return pins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--download', action='store_true', help='load every dataset (downloads on first use)')
    a = ap.parse_args()

    print('environment')
    report(sys.version_info[:2] == (3, 10), f'python {sys.version.split()[0]} (environment.yml: 3.10)')
    pins = pinned()
    for dist, mod in PKG.items():
        try:
            importlib.import_module(mod)
            v = importlib.metadata.version(dist)
            want = pins.get(dist)
            report(want is None or v == want, f'{dist} {v}' + ('' if v == want else f' (environment.yml: {want})'))
        except Exception as e:
            report(False, f'{dist}: {e!r}')

    print('gpu')
    import torch
    report(torch.cuda.is_available(), f'CUDA available: {torch.cuda.is_available()}'
           + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ' - training would run on CPU'))

    print('datasets')
    from utils.utils import get_dataset
    for name in DATASETS:
        present = os.path.isdir(os.path.join('data', name))
        if not (present or a.download):
            print(f'  --   {name}: not downloaded yet (first use downloads it; --download to do it now)')
            continue
        try:
            d = get_dataset(name)
            x = d.data.x
            report(bool(((x >= 0) & (x <= 1)).all()), f'{name}: {len(d)} graphs, {d.num_features} features in [0,1]')
        except Exception as e:
            report(False, f'{name}: {e!r}')

    print('teachers (k-fold)')
    from utils.teachers import seeds_of, teacher_configs
    configs = {c.split(os.sep)[-2]: c for c in teacher_configs()}
    for name in DATASETS:
        n = len(seeds_of(configs[name])) if name in configs else 0
        report(n == 10, f'{name}: {n}/10 folds' + ('' if n == 10 else ' - python -m utils.teachers unpack <bundle>'))

    print('tracking')
    uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5055')
    try:
        urllib.request.urlopen(uri.rstrip('/') + '/health', timeout=3)
        print(f'  ok   MLflow server at {uri}')
    except Exception:
        print(f'  --   no MLflow server at {uri}: start scripts/mlflow_server.sh, tunnel to the main '
              f'machine (ssh -L 5055:127.0.0.1:5055 <host>), or set LOGIX_MLFLOW=0')
    print('ready' if ok else 'NOT ready: fix the FAIL lines above')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
