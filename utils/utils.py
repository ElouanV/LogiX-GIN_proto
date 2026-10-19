"""Seeding and gradient helpers.

The upstream notebooks (nbs/*.ipynb) do ``from utils.utils import *`` and use what it
brought in: ``get_dataset`` and the torch / PyG / sklearn / pandas names below are
re-exported for them. New code imports from the defining modules (utils.datasets,
utils.runs, ...).
"""
import argparse  # noqa: F401
import glob  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import pickle  # noqa: F401
import random
import shutil  # noqa: F401

import numpy as np
import pandas as pd  # noqa: F401
import torch
import torch.nn.functional as F  # noqa: F401
import torch_geometric.transforms as T  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401
from torch import nn  # noqa: F401
from torch_geometric.datasets import MoleculeNet, OGB_MAG, Planetoid, TUDataset  # noqa: F401
from torch_geometric.loader import DataLoader  # noqa: F401
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool  # noqa: F401

from utils.ba_multi_shapes import BAMultiShapesDataset  # noqa: F401
from utils.datasets import get_dataset  # noqa: F401
from utils.spmotif_dataset import *  # noqa: F401,F403
from utils.syn_dataset import SynGraphDataset  # noqa: F401


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)


def zero_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad[param.grad != param.grad] = 0  # Set NaN gradients to 0
