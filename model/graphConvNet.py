import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.graph_conv import GraphConv

from torch_geometric.utils import scatter_



# torch.set_default_tensor_type(torch.DoubleTensor)

class GraphConv_GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_class=2, drop_prob=0.5, max_k=3, device=None):
        super(GraphConv_GNN, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_class = n_class
        self.conv1 = GraphConv(in_channels, out_channels)
        self.conv2 = GraphConv(out_channels, out_channels)

        self.max_k = max_k

        self.layer_norm = torch.nn.LayerNorm(self.out_channels)
        self.dropout = torch.nn.Dropout(p=drop_prob)


        self.bn_out =torch.nn.BatchNorm1d(self.out_channels * 3 * max_k)

        
        self.out_fun = torch.nn.LogSoftmax(dim=1)
        

        self.lin1 = torch.nn.Linear(self.out_channels * 3 * max_k, self.out_channels * 2)

        
        self.lin2 = torch.nn.Linear(self.out_channels * 2,self.out_channels)
        self.lin3 = torch.nn.Linear(self.out_channels, self.n_class)

        self.pre_train = False

        self.reset_parameters()

    def reset_parameters(self):

        print("reset parameters")
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.layer_norm.reset_parameters()

        self.bn_out.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data, hidden_layer_aggregator=None):
        l = data.x

        edge_index = data.edge_index
        h_i = self.conv1(l, edge_index).to(self.device)
        h_i = self.layer_norm(h_i)

        k = self.max_k

        size = data.batch.max().item() + 1

        h_graph_mean = scatter_('mean', h_i, data.batch, dim_size=size)
        h_graph_max = scatter_('max', h_i, data.batch, dim_size=size)
        h_graph_sum = scatter_('add', h_i, data.batch, dim_size=size)
        H = [torch.cat([h_graph_mean, h_graph_max, h_graph_sum], 1)]

        for i in range(k - 1):
            h_i = self.conv2(h_i, edge_index)
            h_i = self.layer_norm(h_i)

            '''
                       concat the mean, the sum and the max of the hidden state for each graph.
            '''
            h_graph_mean = scatter_('mean', h_i, data.batch, dim_size=size)
            h_graph_max = scatter_('max', h_i, data.batch, dim_size=size)
            h_graph_sum = scatter_('add', h_i, data.batch, dim_size=size)
            h_graph = torch.cat((h_graph_mean, h_graph_max, h_graph_sum), 1)

            H.append(h_graph)



        h_k = torch.cat(H, dim=1)
        
        # ------------#
        x = self.bn_out(h_k)

        x = (F.relu(self.lin1(x)))

        x = self.dropout(x)
        x = (F.relu(self.lin2(x)))
        x = self.dropout(x)
        x = self.out_fun(self.lin3(x))


        return x
