import argparse
# from parsers import parameter_parser
from ruamel.yaml import YAML
import os.path as osp
import random
import time as t
import yaml
from yaml import SafeLoader
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as mtick
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj, dropout_edge, add_remaining_self_loops,\
    segregate_self_loops, index_to_mask, to_undirected

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GAE
from model import GCN_Dropout, MLP_Dropout, Model
from eval import node_classification, node_classification_citation, node_classification_wiki, LREvaluator
from utils import tab_printer, dataset_loader, drop_feature, get_split, save_variable, edgeindex2adj, add_edge
import pickle
from torch.utils.tensorboard import SummaryWriter

def get_args(yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")

    parser = argparse.ArgumentParser(description='Parser for NETON')
    #basic
    parser.add_argument("--path", type=str, default='./datasets', help="the directory for loading datasets.")
    parser.add_argument("--dataset", type=str, default="photo", help="the name of dataset for use")
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
    # parser.add_argument("--tau", type=float, default=1.0, help="the temperature for contrast.")

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

def train(model: Model, x, edge_index, optimizer, epoch, args):
    model.train()
    optimizer.zero_grad()

    z = model(x, edge_index)
    edge_ind, edge_mask = dropout_edge(edge_index, p=args.neighbor_masking_prob)

    loss = model.loss(z, edge_ind, args.batch_size, epoch)

    loss.backward()
    optimizer.step()
    return loss.item()


def test_cv(model: Model, data, args):
    model.eval()
    z = model(data.x, data.edge_index)
    if args.dataset.lower() == 'wikics':
        node_classification_wiki(z, data)
    elif args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        node_classification_citation(z, data.y, train_mask=data.train_mask, val_mask = data.val_mask, test_mask = data.test_mask)
    else:
        node_classification(z, data.y)

if __name__ == '__main__':
    args = get_args()
    tab_printer(args)
    plt.style.use(['science', 'ieee', 'std-colors', 'no-latex','grid'])
    plt.rc('font', family='Times New Roman')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    dataset = dataset_loader(path = args.path, name = args.dataset)
    data = dataset[0].to(device)

    if args.base_layer == 'GConv':
        encoder = GCN_Dropout(dataset.num_features, args).to(device)
    else:
        encoder = MLP_Dropout(dataset.num_features, args).to(device)
    model = Model(encoder, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses = []
    with tqdm(total=args.train_epoch, desc='(T)') as pbar:
        for epoch in range(1, args.train_epoch+1):
            loss = train(model, data.x, data.edge_index, optimizer, epoch, args)
            losses.append(loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_cv(model, data, args)
