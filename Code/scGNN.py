import pandas as pd
import torch.optim as optm
from torch.nn import CosineSimilarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        # H' = σ(A·H·W)
        support = self.linear(x)
        output = torch.spmm(adj, support) 
        return F.leaky_relu(output, negative_slope=0.2)


def normalize_adj(adj, norm_type="sym"):
    if isinstance(adj, torch.sparse.FloatTensor):
        adj = adj.to_dense()  

    if norm_type == "sym":
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  
        d_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = torch.matmul(torch.matmul(d_inv_sqrt, adj), d_inv_sqrt)

    return adj



class HybridLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super().__init__()
        self.gcn = GCNLayer(in_dim, out_dim)
        self.gat = AttentionLayer(in_dim, out_dim, alpha)
        self.eta = Parameter(torch.Tensor([0.5]))  

    def forward(self, x, adj):
        gcn_out = self.gcn(x, adj)
        gat_out = self.gat(x, adj)
        return self.eta * gcn_out + (1 - self.eta) * gat_out

class ResGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.res_linear = nn.Linear(in_features, out_features) if in_features != out_features else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if self.res_linear:
            nn.init.xavier_uniform_(self.res_linear.weight)
            nn.init.zeros_(self.res_linear.bias)

    def forward(self, x, adj):
        res = x
        x = self.linear(x)
        x = torch.spmm(adj, x)
        if self.res_linear:
            res = self.res_linear(res)
        return F.leaky_relu(x + res, negative_slope=0.2)


class GENELink(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim,
                 num_head1, num_head2, alpha, device, type='MLP', reduction='mean'):
        super(GENELink, self).__init__()
        self.hybrid = HybridLayer(input_dim, hidden1_dim, alpha)
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction

        if reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif reduction == 'concate':
            self.hidden1_dim = num_head1 * hidden1_dim
            self.hidden2_dim = num_head2 * hidden2_dim

        self.ConvLayer1 = [ResAttentionLayer(hidden1_dim, hidden1_dim, alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module(f'ResConvLayer1_Head{i}', attention)

        self.ConvLayer2 = [ResAttentionLayer(self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module(f'ResConvLayer2_Head{i}', attention)

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        if self.type == 'MLP':
            self.linear = nn.Linear(2 * output_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.hybrid.gcn.reset_parameters()
        self.hybrid.gat.reset_parameters()

        for attention in self.ConvLayer1:
            attention.reset_parameters()
        for attention in self.ConvLayer2:
            attention.reset_parameters()
        nn.init.xavier_uniform_(self.tf_linear1.weight)
        nn.init.xavier_uniform_(self.target_linear1.weight)
        nn.init.xavier_uniform_(self.tf_linear2.weight)
        nn.init.xavier_uniform_(self.target_linear2.weight)

    def encode(self, x, adj):
        gcn_adj = normalize_adj(adj, "sym")  
        gat_adj = normalize_adj(adj, "rw") + torch.eye(adj.size(0), device=self.device)  

        x = self.hybrid(x, gcn_adj)  
        x = x * 0.7 + x.detach() * 0.3  
        x = F.selu(x)
        if self.reduction == 'concate':
            x = torch.cat([att(x, gat_adj) for att in self.ConvLayer1], dim=1)  
            x = F.selu(x)
        elif self.reduction == 'mean':
            x = torch.mean(torch.stack([att(x, gat_adj) for att in self.ConvLayer1]), dim=0)
            x = F.selu(x)

        if self.training:
            mask = torch.bernoulli(torch.full(x.shape, 0.8, device=self.device))
            x = x * mask / 0.8

        out = torch.mean(torch.stack([att(x, gat_adj) for att in self.ConvLayer2]), dim=0)
        return out

    def decode(self, tf_embed, target_embed):
        if self.type == 'dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob, dim=1).view(-1, 1)
            return prob
        elif self.type == 'cosine':
            prob = torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1, 1)
            return prob
        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(f'{self.type} is not available')

    def forward(self, x, adj, train_sample):
        embed = self.encode(x, adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)

        self.tf_output = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)
        return pred

    def get_embedding(self):
        return self.tf_output, self.target_output

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):
        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T, negative_slope=self.alpha)
        return e

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)
        output_data = F.leaky_relu(h_pass, negative_slope=self.alpha)
        output_data = F.normalize(output_data, p=2, dim=1)
        if self.bias is not None:
            output_data = output_data + self.bias
        return output_data

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.spmm(adj, support) 
        return F.leaky_relu(output, negative_slope=0.2)

class ResAttentionLayer(AttentionLayer):
    def __init__(self, input_dim, output_dim, alpha=0.2):
        super().__init__(input_dim, output_dim, alpha)
        self.res_linear = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, adj):
        res = x
        out = super().forward(x, adj)
        if self.res_linear:
            res = self.res_linear(res)
        return out + res  


