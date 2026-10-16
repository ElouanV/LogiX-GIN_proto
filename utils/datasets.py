"""Dataset registry: configs/datasets.yaml says how to load each dataset and how to name its
node features, classes and ground-truth region; ``spec(name)`` returns it as a DatasetSpec
and ``get_dataset(name)`` loads it. A new dataset is one YAML entry (plus a loader or a
pre-transform here if its format is new).
"""
import functools
import os
from dataclasses import dataclass, field

import networkx as nx
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, 'configs', 'datasets.yaml')


# ----------------------------------------------------------------------------- transforms

class MolOneHot:
    """MoleculeNet atoms -> binary literals the logic layers can read.

    MoleculeNet's 9 atom columns are integer codes (atomic number, degree, charge,
    ...), which break the [x, 1-x] negation of LogicalLayer. Kept: one-hot element
    (every element present in BBBP), is_aromatic, is_in_ring. The graph label becomes
    a long class index.
    """
    ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'Na', 'B', 'Ca']
    FEATURES = ELEMENTS + ['aromatic', 'in_ring']
    Z = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 5, 35: 6, 53: 7, 15: 8, 1: 9, 11: 10, 5: 11, 20: 12}

    def __call__(self, data):
        z = data.x[:, 0].long().tolist()
        oh = torch.zeros(len(z), len(self.ELEMENTS))
        for i, a in enumerate(z):
            oh[i, self.Z[a]] = 1.0
        data.x = torch.cat([oh, data.x[:, 7:9].float()], 1)
        data.y = data.y.view(-1).long()
        return data

    def __repr__(self):
        return 'MolOneHot()'


PRE_TRANSFORMS = {'mol_onehot': MolOneHot}


# ----------------------------------------------------------------------------- ground truth

def toxicophore_nodes(x, edge_index, features, hops=1):
    """Nodes within `hops` of an NO2 or NH2 group (N with >= 2 O, resp. >= 2 H, neighbours);
    ``features`` names the one-hot atom columns of ``x``."""
    atom = x.argmax(1)
    N, O, H = features.index('N'), features.index('O'), features.index('H')
    src, dst = edge_index
    region = torch.zeros(len(atom), dtype=torch.bool)
    for n in torch.nonzero(atom == N).flatten().tolist():
        nb = dst[src == n]
        for other in (O, H):
            grp = nb[atom[nb] == other]
            if len(grp) >= 2:
                region[n] = True
                region[grp] = True
    for _ in range(hops):
        grow = region.clone()
        grow[dst[region[src]]] = True
        region = grow
    return region


def motif_nodes(x, edge_index, features=None):
    """Nodes on a cycle (biconnected components of >= 3 nodes). The BA bases of BA2Motifs and
    BAMultiShapes are trees (m = 1), so these are exactly the planted motifs: nodes 20-24 of
    BA2Motifs, the house / 3x3 grid / wheel of BAMultiShapes."""
    g = nx.Graph()
    g.add_nodes_from(range(len(x)))
    g.add_edges_from(edge_index.t().tolist())
    region = torch.zeros(len(x), dtype=torch.bool)
    for comp in nx.biconnected_components(g):
        if len(comp) > 2:
            region[list(comp)] = True
    return region


REGIONS = {'toxicophore': toxicophore_nodes, 'motif': motif_nodes}


# ----------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class GroundTruth:
    classes: tuple            # labels whose explanations are scored
    region: str               # REGIONS key
    motif: str = 'its motif'
    shading: str = 'the ground-truth region'


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: str = 'tu'
    root: str = None
    loader_name: str = None
    use_node_attr: bool = True
    pre_transform: str = None
    features: tuple = ()
    classes: tuple = ()
    ground_truth: GroundTruth = None
    extra: dict = field(default_factory=dict)

    def feature_name(self, i, default=None):
        return self.features[i] if i < len(self.features) else (f'f{i}' if default is None else default)

    def feature_names(self, n):
        return [self.feature_name(i) for i in range(n)]

    def class_name(self, c):
        return self.classes[c] if c < len(self.classes) else f'class {c}'

    def region(self, x, edge_index):
        """The ground-truth node region of one graph [N] (bool), or None without ground truth."""
        if self.ground_truth is None:
            return None
        return REGIONS[self.ground_truth.region](x, edge_index, features=list(self.features))

    def load(self):
        root = self.root or f'data/{self.name}'
        name = self.loader_name or self.name
        pre = PRE_TRANSFORMS[self.pre_transform]() if self.pre_transform else None
        if self.loader == 'tu':
            from torch_geometric.datasets import TUDataset
            return TUDataset(root=root, name=name, use_node_attr=self.use_node_attr)
        if self.loader == 'moleculenet':
            from torch_geometric.datasets import MoleculeNet
            return MoleculeNet(name=name, root=root, **({'pre_transform': pre} if pre else {}))
        if self.loader == 'synthetic':
            from utils.syn_dataset import SynGraphDataset
            return SynGraphDataset(root=root, name=name)
        if self.loader == 'bamultishapes':
            from utils.ba_multi_shapes import BAMultiShapesDataset
            return BAMultiShapesDataset(root=root)
        if self.loader == 'spmotif':
            from utils.spmotif_dataset import SPMotif
            return SPMotif(root=root, mode=self.extra.get('mode', 'train'), transform=None)
        if self.loader == 'planetoid':
            from torch_geometric.datasets import Planetoid
            return Planetoid(root=root, name=name)
        if self.loader == 'ogb_mag':
            from torch_geometric.datasets import OGB_MAG
            return OGB_MAG(root=root)
        raise ValueError(f'{self.name}: unknown loader {self.loader!r}')


KEYS = {'loader', 'root', 'name', 'use_node_attr', 'pre_transform', 'features', 'classes', 'ground_truth'}


@functools.lru_cache(maxsize=None)
def registry(path=CONFIG):
    raw = yaml.safe_load(open(path)) or {}
    out = {}
    for name, e in raw.items():
        gt = e.get('ground_truth')
        out[name] = DatasetSpec(
            name=name, loader=e.get('loader', 'tu'), root=e.get('root'), loader_name=e.get('name'),
            use_node_attr=e.get('use_node_attr', True), pre_transform=e.get('pre_transform'),
            features=tuple(e.get('features', ())), classes=tuple(e.get('classes', ())),
            ground_truth=GroundTruth(**{**gt, 'classes': tuple(gt['classes'])}) if gt else None,
            extra={k: v for k, v in e.items() if k not in KEYS})
    return out


def spec(name):
    """The dataset's DatasetSpec; an unlisted name is a TUDataset with generic names."""
    return registry().get(name) or DatasetSpec(name=name)


def get_dataset(name):
    return spec(name).load()
