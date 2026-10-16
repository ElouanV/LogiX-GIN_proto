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

# get_dataset lives in utils/datasets.py; re-exported here because the upstream notebooks
# do `from utils.utils import *`
from utils.datasets import get_dataset  # noqa: F401

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

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)

def zero_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad[param.grad != param.grad] = 0  # Set NaN gradients to 0