from typing import *
import os
import torch
import dgl
import random
import numpy as np
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, CitationFull, Reddit, Reddit2, PPI, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_torch_coo_tensor, add_self_loops, add_remaining_self_loops, sort_edge_index, to_undirected
import pickle
from torch_sparse import coalesce

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = args.keys()
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    t.set_cols_align(["l", "r"])
    # t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]] + [[k, args[k]] for k in keys])
    print(t.draw())

def dataset_loader(path = './datasets', name='photo'):
    path = path + '/' + name.lower()
    print("dataset is downloaded in the directory：'"+path+"'.")
    if name.lower() == 'photo':
        dataset = Amazon(path, name='Photo', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'computers':
        dataset = Amazon(path, name = 'Computers', transform =T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'cs':
        dataset = Coauthor(path, name='CS', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'physics':
        dataset = Coauthor(path, name='Physics', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'cora':
        dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'citeseer':
        dataset = Planetoid(path, name='Citeseer', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'pubmed':
        dataset = Planetoid(path, name='Pubmed', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'dblp':
        dataset = CitationFull(path, name='DBLP', transform=T.NormalizeFeatures())
        return dataset
    elif name.lower() == 'wikics':
        dataset = WikiCS(path+'/WikiCS', is_undirected=True)
        return dataset
    elif name.lower() == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path+'/arxiv')
        return dataset
    elif name.lower() == 'reddit':
        dataset = Reddit2(root=path+'/reddit2')
        return dataset
    elif name.lower() == 'ppi':
        train_dataset = PPI(root=path+'/ppi', split='train')
        val_dataset = PPI(root=path+'/ppi', split='val')
        test_dataset = PPI(root=path+'/ppi', split='test')
        return train_dataset, val_dataset, test_dataset
    elif name.lower() == 'flickr':
        dataset = Flickr(root=path+'/flickr')
        return dataset
    else:
        raise ValueError("No such dataset!")

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    # print(new_edge_index)
    sym_edge_index = torch.flip(new_edge_index, (0,))
    edge_index = torch.cat([edge_index, new_edge_index, sym_edge_index], dim=1)

    edge_index = sort_edge_index(edge_index)

    return coalesce_edge_index(edge_index)[0]

def coalesce_edge_index(edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> (
torch.Tensor, torch.FloatTensor):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = edge_weights if edge_weights is not None else torch.ones((num_edges,), dtype=torch.float32,
                                                                            device=edge_index.device)
    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes)

def edgeindex2adj(edge_index, num_nodes):
    adj_shape = (num_nodes, num_nodes)  # n*n
    edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
    adj = to_torch_coo_tensor(edge_index, size = adj_shape)
    return adj

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1

    train_size = int(num_samples * train_ratio)
    valid_size = int(num_samples * (1 - train_ratio - test_ratio))
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'val': indices[train_size: train_size + valid_size],
        'test': indices[train_size + valid_size:]
    }

def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

