import argparse
from parsers import parameter_parser
from ruamel.yaml import YAML
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj, dropout_edge, add_remaining_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GAE
from model import GCN_Dropout, MLP_Dropout,  Model
from eval import node_classification, node_classification_wiki, LREvaluator
from utils import tab_printer, dataset_loader, drop_feature, get_split, save_variable, edgeindex2adj, add_edge
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
import copy
import pickle
from torch.utils.tensorboard import SummaryWriter

def get_args(yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")

    parser = argparse.ArgumentParser(description='Parser for NETON')
    #basic
    parser.add_argument("--path", type=str, default='./datasets', help="the directory for loading datasets.")
    parser.add_argument("--dataset", type=str, default="ppi", help="the name of dataset for use")
    parser.add_argument("--neighbor_masking_prob", type=float, default=0.4, help="the probability for masking a neighbor.")
    parser.add_argument("--embed_dropout_prob", type=float, default=0.4, help="the probability for dropout an embedding dimension.")
    parser.add_argument("--dim_h", type=int, default=1024, help="the dimension of encoder layers.")
    parser.add_argument("--dim_p", type=int, default=256, help="the dimension of projector layers.")
    parser.add_argument("--base_layer", type=str, default='Linear', help="the basic encoder layer.")
    parser.add_argument("--activation", type=str, default='prelu', help="the activation function of encoder layer.")
    parser.add_argument("--num_layer", type=int, default=2, help="the number of encoder layers.")
    parser.add_argument("--use-ln", type=bool, default=True, help="whether compute metrics.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="the learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0, help="the weight decay rate for optimizer.")
    parser.add_argument("--train_epoch", type=int, default=1000, help="the number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=0, help="the batch size for batch semi-loss.")

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = args.dataset
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    # Update params from .yamls
    args = parser.parse_args()
    return args


def train(model: Model, train_loader, optimizer, args):
    model.train()
    total_loss = total_num = 0
    for sub_data in train_loader:
        optimizer.zero_grad()
        sub_data.to(model.device)
        x = sub_data.x
        edge_index = sub_data.edge_index
        z = model(x, edge_index)
        edge_ind, edge_mask = dropout_edge(edge_index, p=args.neighbor_masking_prob)  # , force_undirected=True)
        loss = model.loss(z, edge_ind, args.batch_size, epoch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * x.shape[0]
        total_num += x.shape[0]
    return total_loss / total_num


def test(model: Model, train_loader, val_loader, test_loader, args):
    model.eval()
    zs = []
    ys = []
    node_counter = []
    for loader in [train_loader, val_loader, test_loader]:
        counter = 0
        for sub_data in loader:
            sub_data.to(model.device)
            x = sub_data.x
            edge_index = sub_data.edge_index
            z = model(x, edge_index)
            counter += sub_data.num_nodes
            zs.append(z)
            ys.append(sub_data.y)
        node_counter.append(counter)
    z, y = torch.cat(zs, dim=0), torch.cat(ys, dim=0)
    split_idx = [sum(node_counter[:i+1]) for i in range(len(node_counter))]
    LREvaluator()(z, y, multi_class=True, split_idx= split_idx)



if __name__ == '__main__':
    args = get_args()
    tab_printer(args)
    plt.style.use(['science', 'ieee', 'std-colors', 'no-latex','grid'])
    plt.rc('font', family='Times New Roman')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    assert args.dataset.lower() == 'ppi'
    train_set, val_set, test_set = dataset_loader(path = args.path, name = args.dataset)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    if args.base_layer == 'GConv':
        encoder = GCN_Dropout(train_set.num_features, args).to(device)
    else:
        encoder = MLP_Dropout(train_set.num_features, args).to(device)

    model = Model(encoder, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    with tqdm(total=args.train_epoch, desc='(T)') as pbar:
        for epoch in range(1, args.train_epoch+1):
            loss = train(model, train_loader, optimizer, args)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test(model, train_loader, val_loader, test_loader, args)
