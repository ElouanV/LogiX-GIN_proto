import torch
import random
import numpy as np
import os
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid, OGB_MAG, MoleculeNet
from utils.ba_multi_shapes import BAMultiShapesDataset
from utils.syn_dataset import SynGraphDataset
from utils.spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import train_test_split
import shutil
import glob
import pandas as pd
import argparse
import pickle
import json

def create_folder(dataset_name, args, seed=None):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    path = f'results/{dataset_name}/{args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

def create_folder_logic(dataset_name, args, baseline_args, seed=None, root='results_logic'):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    baseline_args_s = '|'.join([f"{k}={baseline_args[k]}" for k in sorted(baseline_args.keys())])
    path = f'{root}/{dataset_name}/{args_s}/{baseline_args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

BBBP_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'Na', 'B', 'Ca']
BBBP_FEATURES = BBBP_ELEMENTS + ['aromatic', 'in_ring']
_BBBP_Z = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 5, 35: 6, 53: 7, 15: 8, 1: 9, 11: 10, 5: 11, 20: 12}


class MolOneHot:
    """MoleculeNet atoms -> binary literals the logic layers can read.

    MoleculeNet's 9 atom columns are integer codes (atomic number, degree, charge,
    ...), which break the [x, 1-x] negation of LogicalLayer. Kept: one-hot element
    (every element present in BBBP), is_aromatic, is_in_ring. The graph label becomes
    a long class index.
    """
    def __call__(self, data):
        z = data.x[:, 0].long().tolist()
        oh = torch.zeros(len(z), len(BBBP_ELEMENTS))
        for i, a in enumerate(z):
            oh[i, _BBBP_Z[a]] = 1.0
        data.x = torch.cat([oh, data.x[:, 7:9].float()], 1)
        data.y = data.y.view(-1).long()
        return data

    def __repr__(self):
        return 'MolOneHot()'


def get_dataset(dataset_name):
    if dataset_name == 'Ba2Motifs':
        return  SynGraphDataset(root='data/ba_2motifs', name='ba_2motifs')
    elif dataset_name == 'Ba2MotifsNoisy':
        return  SynGraphDataset(root='data/ba_2motifs', name='ba_2motifsnoisy')
    elif dataset_name == 'TreeGrid':
        return  SynGraphDataset(root='data/tree_grid', name='tree_grid')
    elif dataset_name == 'BaShapes':
        return  SynGraphDataset(root='data/ba_shapes', name='ba_shapes')
    elif dataset_name == 'BaCommunity':
        return  SynGraphDataset(root='data/ba_community', name='ba_community')

    elif dataset_name == 'SPMotif':
        return SPMotif(root='data/SPMotif-0.333', mode='train', transform=None)

    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        return Planetoid(root=f'data/{dataset_name}', name=dataset_name)
    elif dataset_name == 'OGB_MAG':
        return OGB_MAG(root=f'data/{dataset_name}')
    elif dataset_name == 'BBBP':
        # binary atom literals (see MolOneHot); cached separately from the raw version
        return MoleculeNet(name='BBBP', root='data/BBBP_onehot', pre_transform=MolOneHot())
    elif dataset_name == 'BBBP_raw':
        return MoleculeNet(name='BBBP', root='data/BBBP')
    elif dataset_name in ('AIDS', 'PROTEINS'):
        # use_node_attr=True would prepend 4 continuous columns (chem, charge, x, y
        # coordinates) to the 38 one-hot atom labels, and for PROTEINS one continuous
        # column (values in [-538, 798]) to the 3 one-hot SSE labels; the logic layers
        # need [0,1] literals
        return TUDataset(root=f'data/{dataset_name}', name=dataset_name, use_node_attr=False)
    elif dataset_name == 'BaMultiShapes':
        return BAMultiShapesDataset(root=f'data/{dataset_name}')
    return TUDataset(root=f'data/{dataset_name}', name=dataset_name, use_node_attr=True)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)

def zero_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad[param.grad != param.grad] = 0  # Set NaN gradients to 0