"""Run directories: where a trained model lives, how its directory is named and how to load
it back. Every script that reads or writes a run goes through here.

Layout (``<cfg>`` = ``k=v|k=v|...`` of the run's arguments, sorted, see ``config_name``):

    results/<ds>/<teacher cfg>/<seed>                                 teacher (train_baseline.py)
    <root>/<ds>/<cfg>/<teacher cfg>/<seed>                            student (train_logic.py,
                                                                      train_proto.py); root =
                                                                      results_logic, results_proto,
                                                                      an hps study or a final run
    <run>/sparse/<cfg>, <run>/qat/<cfg>, <run>/harden/<cfg>           fine-tunes, nestable

Each holds ``args.json``, ``data.pkl`` (split indices and the split datasets; a link to the
source run's for a fine-tune) and ``best.pt`` (a full pickled model for students, a state dict
for teachers).
"""
import hashlib
import json
import os
import pickle
from dataclasses import dataclass

import torch
from torch_geometric.loader import DataLoader

TEACHER_ROOT = 'results'
LOGIC_ROOT = 'results_logic'
PROTO_ROOT = 'results_proto'
FINETUNES = ('sparse', 'qat', 'harden')
EVAL_BATCH_SIZE = 64          # val / test / unshuffled loaders
FEATURELESS_DIM = 10          # input width of datasets without node features (x = ones)
SPLITS = ('train', 'val', 'test')


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def config_name(args):
    return '|'.join(f'{k}={args[k]}' for k in sorted(args))


def teacher_dir(dataset, args, seed=None, root=TEACHER_ROOT):
    """Teacher configuration directory (with ``seed``: that seed's run), created."""
    path = f'{root}/{dataset}/{config_name(args)}'
    if seed is not None:
        path = f'{path}/{seed}'
    os.makedirs(path, exist_ok=True)
    return path


def student_dir(dataset, args, teacher_args, seed=None, root=LOGIC_ROOT, hash_long=False):
    """Student run directory ``<root>/<ds>/<cfg>/<teacher cfg>[/<seed>]``, created.

    ``hash_long``: a ``<cfg>`` longer than one path component allows (255 bytes) becomes
    ``<proto_level>-<sha1[:12]>``, its full name written to ``<root>/<ds>/<name>/config.txt``
    (prototype runs)."""
    cfg = full = config_name(args)
    if hash_long and len(cfg.encode()) > 255:
        cfg = f"{args['proto_level']}-{hashlib.sha1(full.encode()).hexdigest()[:12]}"
    path = f'{root}/{dataset}/{cfg}/{config_name(teacher_args)}'
    if seed is not None:
        path = f'{path}/{seed}'
    os.makedirs(path, exist_ok=True)
    if cfg != full:
        with open(f'{root}/{dataset}/{cfg}/config.txt', 'w') as f:
            f.write(full + '\n')
    return path


def num_features(dataset):
    return dataset.num_features or FEATURELESS_DIM


def split_datasets(dataset, indices):
    """{'train': dataset[train_indices], 'val': ..., 'test': ...}."""
    return {s: dataset[indices[f'{s}_indices']] for s in SPLITS}


def make_loaders(datasets, batch_size, eval_batch_size=EVAL_BATCH_SIZE):
    """Shuffled train loader at ``batch_size``, unshuffled val / test loaders."""
    return {s: DataLoader(d, batch_size=batch_size if s == 'train' else eval_batch_size, shuffle=s == 'train')
            for s, d in datasets.items()}


def save_split(path, indices, datasets):
    """data.pkl of a run: the split indices and the split datasets."""
    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({**{f'{s}_indices': indices[f'{s}_indices'] for s in SPLITS},
                     **{f'{s}_dataset': datasets[s] for s in SPLITS}}, f)


def write_args(path, args):
    """args.json of a run (numpy scalars as Python numbers); returns the written dict."""
    args = {k: (v.item() if hasattr(v, 'item') else v) for k, v in args.items()}
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(args, f)
    return args


def load_teacher(path, n_features, num_classes, device):
    """The frozen GIN teacher saved at ``path`` (a teacher seed directory), in eval mode."""
    from models.model import GIN
    targs = json.load(open(os.path.join(path, 'args.json')))
    teacher = GIN(num_features=n_features, num_classes=num_classes, hidden_dim=targs['hidden_dim'],
                  num_layers=targs['num_layers'], nogumbel=targs['nogumbel']).to(device)
    teacher.load_state_dict(torch.load(os.path.join(path, 'best.pt'), map_location='cpu'))
    return teacher.eval()


class RunDir:
    """A trained run directory (see the module docstring), read lazily.

    ``stage`` is 'teacher', 'logic' or 'proto' (from args.json); ``finetunes`` the
    [(kind, cfg)] chain below the trained run ``source``; ``dataset``, ``config``,
    ``teacher_config`` and ``seed`` come from the path, ``dataset`` can be overridden."""

    def __init__(self, path, dataset=None):
        self.path = path
        parts = os.path.normpath(path).split(os.sep)
        self.finetunes = []
        while len(parts) > 2 and parts[-2] in FINETUNES:
            self.finetunes.insert(0, (parts[-2], parts[-1]))
            parts = parts[:-2]
        self.source = os.sep.join(parts)
        if TEACHER_ROOT in parts and len(parts) - parts.index(TEACHER_ROOT) == 4:
            ds, self.teacher_config, self.seed = parts[-3:]
            self.config = None
        elif len(parts) >= 4:
            ds, self.config, self.teacher_config, self.seed = parts[-4:]
        else:
            raise ValueError(f'{path} is not a run directory (<ds>/<cfg>/<teacher cfg>/<seed>)')
        self.dataset = dataset or ds
        self._args = self._data = None

    def file(self, name):
        return os.path.join(self.path, name)

    @property
    def args(self):
        if self._args is None:
            if not os.path.exists(self.file('args.json')):
                raise ValueError(f'{self.path} is not a run directory (no args.json)')
            self._args = json.load(open(self.file('args.json')))
        return self._args

    @property
    def stage(self):
        if 'proto_level' in self.args:
            return 'proto'
        return 'logic' if 'conv_reg' in self.args else 'teacher'

    @property
    def data(self):
        """data.pkl: {split}_indices and the pickled {split}_dataset."""
        if self._data is None:
            self._data = pickle.load(open(self.file('data.pkl'), 'rb'))
        return self._data

    def pickled(self, split):
        """The split's dataset as pickled in data.pkl."""
        return self.data[f'{split}_dataset']

    def subsets(self, splits=SPLITS):
        """The split's graphs re-read from the dataset (utils.utils.get_dataset) by index."""
        from utils.datasets import get_dataset
        ds = get_dataset(self.dataset)
        return {s: ds[self.data[f'{s}_indices']] for s in splits}

    def model(self, device='cpu', ckpt='best.pt'):
        """The pickled student model of a logic / prototype run."""
        return torch.load(self.file(ckpt), map_location=device, weights_only=False)

    @property
    def teacher_path(self):
        return os.path.join(TEACHER_ROOT, self.dataset, self.teacher_config, self.seed)

    def teacher(self, n_features, num_classes, device):
        return load_teacher(self.teacher_path, n_features, num_classes, device)

    def write_finetune(self, out_dir, key, settings, record, record_name=None):
        """A fine-tune's run directory: args.json (this run's args + ``{key: settings}``), a
        link to this run's data.pkl and ``<record_name or key>.json`` (record); returns the
        record's path. best.pt is written by the caller."""
        with open(os.path.join(out_dir, 'args.json'), 'w') as f:
            json.dump({**self.args, key: settings}, f)
        link = os.path.join(out_dir, 'data.pkl')
        if not os.path.exists(link):
            os.symlink(os.path.abspath(self.file('data.pkl')), link)
        path = os.path.join(out_dir, f'{record_name or key}.json')
        with open(path, 'w') as f:
            json.dump(record, f, indent=1)
        return path


@dataclass
class LoadedRun:
    """A trained run opened for fine-tuning or evaluation (``load_run``)."""
    run: RunDir
    model: torch.nn.Module
    loaders: dict             # train (shuffled), val, test
    device: torch.device
    num_classes: int
    teacher: torch.nn.Module = None


def load_run(path, dataset=None, batch_size=None, teacher=False):
    """The run at ``path``: its checkpoint, its split loaders (train at ``batch_size``, default
    the run's) and, with ``teacher``, its frozen teacher."""
    from utils.datasets import get_dataset
    run = RunDir(path, dataset)
    dev = get_device()
    ds = get_dataset(run.dataset)
    loaders = make_loaders(split_datasets(ds, run.data), batch_size or run.args['batch_size'])
    model = run.model(dev)
    return LoadedRun(run, model, loaders, dev, ds.num_classes,
                     run.teacher(num_features(ds), ds.num_classes, dev) if teacher else None)
