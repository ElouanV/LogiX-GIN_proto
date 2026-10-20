"""One run opened for the explorer, and the runs a root directory holds."""
import datetime
import json
import os
import re

from explorer.reader import SymbolicReader
from utils.runs import SPLITS, RunDir
from utils.tracking import _git


class RunSession:
    """A run directory read by SymbolicReader, with its card and split indexes cached.

    ``ref_split`` (default train) is the split the card's observed ranges, firing rates and
    prototype matches are measured on; ``ref_max`` caps its graphs (None: all)."""

    def __init__(self, path, dataset=None, ckpt='best.pt', device='cpu', ref_split='train', ref_max=None):
        self.run = RunDir(path, dataset)
        if self.run.stage == 'teacher':
            raise ValueError(f'{path} is a teacher run: the explorer reads LogiX-GIN students')
        self.ckpt, self.ref_split, self.ref_max = ckpt, ref_split, ref_max
        self.reader = SymbolicReader(self.run.model(device, ckpt), self.run.dataset, device)
        self._card, self._index = None, {}

    def meta(self):
        r = self.run
        return {'path': r.path, 'dataset': r.dataset, 'stage': r.stage, 'seed': r.seed, 'ckpt': self.ckpt,
                'finetunes': [k for k, _ in r.finetunes], 'args': r.args, 'stored_metrics': self.stored_metrics(),
                'read_at': datetime.datetime.now().isoformat(timespec='seconds'), **_git()}

    def stored_metrics(self):
        """interp_metrics.py results saved next to the checkpoint, per split (if any)."""
        out = {}
        for s in SPLITS:
            f = self.run.file(f'interp_{s}.json')
            if os.path.exists(f):
                with open(f) as fh:
                    out[s] = json.load(fh)
        return out

    def card(self):
        if self._card is None:
            self._card = self.reader.card(self.run.pickled(self.ref_split), self.ref_split, self.ref_max)
        return self._card

    def index(self, split):
        if split not in self._index:
            self._index[split] = self.reader.index(self.run.pickled(split))
        return self._index[split]

    def trace(self, split, i):
        ds = self.run.pickled(split)
        if not 0 <= i < len(ds):
            raise IndexError(f'{split} has {len(ds)} graphs, no #{i}')
        return self.reader.trace(ds[i], split, i)


HASHED = re.compile(r'^(node|graph|both)-[0-9a-f]{12}$')


def run_label(path, root):
    """A short name for a run: its path under ``root`` without the argument strings."""
    parts = os.path.relpath(path, root).split(os.sep)
    keep = [p for p in parts if '=' not in p and not HASHED.match(p) and p != 'run']
    return ' / '.join(p for i, p in enumerate(keep) if p not in keep[:i])


def discover(root):
    """Student run directories under ``root`` (best.pt + args.json + data.pkl), sorted."""
    found = []
    for d, _, files in os.walk(root):
        if not {'best.pt', 'args.json', 'data.pkl'} <= set(files):
            continue
        try:
            run = RunDir(d)
            if run.stage == 'teacher':
                continue
            level = run.args.get('proto_level', 'classic')
        except (ValueError, OSError):
            continue
        found.append({'path': d, 'label': run_label(d, root) or d, 'dataset': run.dataset, 'level': level,
                      'finetunes': [k for k, _ in run.finetunes]})
    return sorted(found, key=lambda r: r['label'])
