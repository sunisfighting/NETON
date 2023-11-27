import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  SAGEConv, GATConv, ClusterGCNConv, GCNConv, DeepGCNLayer, GENConv, LayerNorm
from torch_geometric.utils import add_remaining_self_loops, get_laplacian
from utils import edgeindex2adj
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import silhouette_score as sh_score
from tqdm import tqdm



class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self): # Override function that finds format to use.
        self.format = "%1.1f" # Give format here


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

class GCN_Dropout(torch.nn.Module):
    def __init__(self, in_channels: int, args):
        super(GCN_Dropout, self).__init__()

        base_layer = GCNConv
        out_channels = args.dim_h
        p_drop = args.embed_dropout_prob
        self.num_layer = args.num_layer
        self.args = args
        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu':F.rrelu, 'elu':F.elu})[args.activation]
        # assert num_layer >= 2
        self.factor = 2
        self.convs = torch.nn.ModuleList()
        self.drops = nn.ModuleList()
        if args.use_ln:
            self.lns = nn.ModuleList()

        if self.num_layer >=2:
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(base_layer(in_channels, self.factor*out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(self.factor * out_channels))
            for _ in range(1, self.num_layer-1):
                self.drops.append(nn.Dropout(p=p_drop))
                self.convs.append(base_layer(self.factor*out_channels, self.factor*out_channels))
                if args.use_ln:
                    self.lns.append(nn.LayerNorm(self.factor * out_channels))
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(base_layer(self.factor*out_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))
        else:
            self.drops.append(nn.Dropout(p=p_drop))
            self.convs.append(base_layer(in_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.num_layer):
            x= self.drops[i](x)
            x = self.activation(self.convs[i](x, edge_index))
            if self.args.use_ln:
                x = self.lns[i](x)
        return x

    @torch.no_grad()  # for large graph
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Inferring')
        device = x_all.device
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = self.activation(conv(self.drops[i](x), batch.edge_index.to(device)))
                if self.args.use_ln:
                    x = self.lns[i](x)
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all.to(device)


class MLP_Dropout(torch.nn.Module):
    def __init__(self, in_channels: int, args):
        super(MLP_Dropout, self).__init__()
        out_channels = args.dim_h
        p_drop = args.embed_dropout_prob
        self.num_layer = args.num_layer
        self.args = args
        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu':F.rrelu, 'elu':F.elu})[args.activation]
        self.factor = 2
        self.lins = nn.ModuleList()
        self.drops = nn.ModuleList()
        if args.use_ln:
            self.lns = nn.ModuleList()
        if self.num_layer >= 2:
            self.lins.append(nn.Linear(in_channels, self.factor*out_channels))
            self.drops.append(nn.Dropout(p=p_drop))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(self.factor * out_channels))
            for _ in range(1, self.num_layer-1):
                self.drops.append(nn.Dropout(p=p_drop))
                self.lins.append(nn.Linear(self.factor*out_channels, self.factor*out_channels))
                if args.use_ln:
                    self.lns.append(nn.LayerNorm(self.factor * out_channels))
            self.drops.append(nn.Dropout(p=p_drop))
            self.lins.append(nn.Linear(self.factor*out_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))
        else:
            self.drops.append(nn.Dropout(p=p_drop))
            self.lins.append(nn.Linear(in_channels, out_channels))
            if args.use_ln:
                self.lns.append(nn.LayerNorm(out_channels))


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.num_layer):
            x = self.drops[i](x)
            x = self.lins[i](x)
            x = self.activation(x)
            if self.args.use_ln:
                x = self.lns[i](x)
        return x

    @torch.no_grad()  # for large graph
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset))
        pbar.set_description('Inferring')
        device = x_all.device
        xs = []
        for batch in subgraph_loader:
            x = x_all[batch.n_id.to(x_all.device)][:batch.batch_size].to(device)
            for i in range(self.num_layer):
                x = self.activation(self.lins[i](self.drops[i](x)))
                if self.args.use_ln:
                    x = self.lns[i](x)
            xs.append(x.cpu())
            pbar.update(batch.batch_size)
        x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class Model(torch.nn.Module):
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.encoder = encoder

        self.fc1 = torch.nn.Linear(args.dim_h, args.dim_p)
        self.activation = F.elu
        self.fc2 = torch.nn.Linear(args.dim_p, args.dim_h)
        self.EPS = torch.Tensor([1e-6]).to(self.device)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(z)))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.mm(z1, z2.t())

    def loss(self, z: torch.Tensor, edge_index: torch.Tensor, batch_size: int = 0, epoch: int =0):
        h =F.normalize(self.projection(z))

        if batch_size == 0:
            adj = edgeindex2adj(edge_index, h.shape[0]).to_dense()
            rec = (torch.mm(h, h.t()) + torch.Tensor([1.0]).to(self.device)) / 2


            loss = -((adj * torch.log(rec + self.EPS)).sum(1) / adj.sum(1) + (
                        (1 - adj) * torch.log(1 - rec + self.EPS)).sum(1) / (1 - adj).sum(1))
        else:
            loss = self.batch_loss(h, edge_index, batch_size)
        ret = loss.mean()
        return ret

    def batch_loss(self, h: torch.Tensor, edge_index: torch.Tensor, batch_size: int = 0):
        device = h.device
        num_nodes = h.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        adj = edgeindex2adj(edge_index, h.shape[0])  # torch.sparse.tensor
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]  # index for node
            adj_b = adj.index_select(dim=0, index=mask).to_dense()  # row_sampling
            rec_b = (torch.mm(h[mask], h.t()) + torch.Tensor([1.0]).to(self.device)) / 2

            losses.append(-((adj_b * torch.log(rec_b + self.EPS)).sum(1) / adj_b.sum(1) + (
                    (1 - adj_b) * torch.log(1 - rec_b + self.EPS)).sum(1) / (1 - adj_b).sum(1)))
        return torch.cat(losses)


