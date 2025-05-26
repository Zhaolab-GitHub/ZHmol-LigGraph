import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_add_pool
import numpy as np

class Net_screen(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(Net_screen, self).__init__()
        print('num_features: ', num_features)
        print('num_classes: ', num_classes)
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = 3)
        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = 3) for _ in range(args.n_graph_layer)])

        self.lins = torch.nn.ModuleList([torch.nn.Linear(args.d_graph_layer, args.d_FC_layer) if i == 0 else
                                         torch.nn.Linear(args.d_FC_layer, args.d_FC_layer) for i in range(args.n_FC_layer)])
        self.lin3 = torch.nn.Linear(args.d_FC_layer, num_classes)
        self.relu = torch.nn.ReLU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.last = args.last
        self.flexible = args.flexible

    def forward(self, x, edge_index, edge_attr, flexible_idx, batchs, energy=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.relu(x)
            x = self.Dropout(x)

        if self.flexible:
            x = global_add_pool(x[flexible_idx], batchs[flexible_idx])
        else:
            x = global_mean_pool(x, batchs)

        x = self.lins[0](x)
        x = self.relu(x)
        x = self.Dropout(x)
        for i in range(1, len(self.lins)):
            x = self.lins[i](x)
            x = self.relu(x)
            x = self.Dropout(x)
        x = self.lin3(x)
        if not hasattr(self, 'last') or self.last == 'log':
            return F.log_softmax(x, dim=1)
        elif not hasattr(self, 'last') or self.last == 'sigmoid':
            return torch.sigmoid(x)
        elif not hasattr(self, 'last') or self.last == 'logsigmoid':
            return F.logsigmoid(x)
        else:
            return F.softmax(x, dim=1)

class Net_coor(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Net_coor, self).__init__()
        print("get the model of Net_coor")
        self.conv1 = TransformerConv(num_features, args.d_graph_layer, edge_dim = args.edge_dim)

        self.convs = torch.nn.ModuleList([TransformerConv(args.d_graph_layer, args.d_graph_layer, edge_dim = args.edge_dim) for _ in range(args.n_graph_layer)])

        self.convl = TransformerConv(args.d_graph_layer, 3, edge_dim = args.edge_dim)

        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        if args.residue:
            self.residue = True
        else:
            self.residue = False

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gelu(x)
        x = self.Dropout(x)

        for i in range(len(self.convs)):
            if self.residue:
                identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.gelu(x)
            if self.residue:
                x = x + identity
            x = self.Dropout(x)

        x = self.convl(x, edge_index, edge_attr)
        return x
