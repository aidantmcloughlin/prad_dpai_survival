
from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import (
    GCNConv, GraphConv, GatedGraphConv, 
    GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
    )
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import (
    global_mean_pool as gavgp, 
    global_max_pool as gmp, 
    global_add_pool as gap)
from torch_geometric.transforms.normalize_features import NormalizeFeatures


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


# class PreTrainedClinComponent(nn.Module):
#     def __init__(self, params):
#         super(PretrainedComponent, self).__init__()
        
#         concat_pt_layers = [
#             nn.Linear(
#                 self.params['inputs_shapes']['clin'], 
#                 self.params['concat_hidden_sizes'][0]), 
#             nn.ReLU(),
#             nn.Dropout(0.25)]
#         for i in range(1, len(self.params['concat_hidden_sizes'])):
#             concat_pt_layers.append(nn.Linear(
#                 self.params['concat_hidden_sizes'][i-1], 
#                 self.params['concat_hidden_sizes'][i]))
#             concat_pt_layers.append(nn.ReLU())
#             concat_pt_layers.append(nn.Dropout(0.25))
#         concat_pt_layers.append(nn.Linear(
#             self.params['concat_hidden_sizes'][-1], self.params['n_classes']))

#         self.pretrain_model = nn.Sequential(*concat_pt_layers)   

#     def forward(self, **kwargs):
#         return self.pretrain_model(kwargs['clin_data'])

#     def pretrain(self, pretrain_dataloader, epochs=50, lr=1e-3):
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         for epoch in range(epochs):
#             for batch in pretrain_dataloader:
#                 x, y = batch
#                 optimizer.zero_grad()
#                 output = self.forward(x)
#                 loss = F.mse_loss(output, y)
#                 loss.backward()
#                 optimizer.step()



class PatchGCN(torch.nn.Module):
    def __init__(self, params, input_dim=2227, num_layers=4, edge_agg='spatial', 
        multires=False, resample=0, fusion=None, num_features=1024, 
        hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, 
        ):
        super(PatchGCN, self).__init__()

        ### Main params:
        self.params = params
        self.n_classes = self.params['n_classes']

        ### Auxiliary PatchGCN Hyperparams:
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(
                *[nn.Dropout(self.resample), nn.Linear(1024, 256), 
                nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(
                *[nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            
            if i == 1:
                norm = None
            else:
                norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm=norm, act=act, block='res', dropout=0.25, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(
            *[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)]
            )

        self.path_attention_head = Attn_Net_Gated(
            L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1
            )
        self.path_rho = nn.Sequential(
            *[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(dropout)]
            )


        ## Processed WSI dimension reduction module ===========================
        if self.params['wsi_hidden_sizes']:
            wsi_layers = [
                nn.Linear(
                    hidden_dim*4, 
                    self.params['wsi_hidden_sizes'][0]), 
                # nn.BatchNorm1d(self.params['wsi_hidden_sizes'][0]),
                nn.ReLU()]
            for i in range(1, len(self.params['wsi_hidden_sizes'])):
                wsi_layers.append(nn.Linear(
                    self.params['wsi_hidden_sizes'][i-1], self.params['wsi_hidden_sizes'][i]))
                # wsi_layers.append(nn.BatchNorm1d(self.params['wsi_hidden_sizes'][i]))
                wsi_layers.append(nn.ReLU())
            self.wsi_model = nn.Sequential(*wsi_layers)
        else: ## allowing for empty WSI model.
            self.wsi_model = nn.Sequential()

        ### Linear Classifier Module/s: =======================================
        if bool(self.params['wsi_hidden_sizes']):
            wsi_feat_final_size = self.params['wsi_hidden_sizes'][-1]
        else:
            wsi_feat_final_size = hidden_dim*4
        

        if self.params['pretrain_clin'] is True:
            pass
        
        else:
            concat_in_sh = (
                        self.params['inputs_shapes']['clin'] + 
                        wsi_feat_final_size
                )
            
            if bool(self.params['concat_hidden_sizes']):
                concatclin_layers = [
                    nn.Linear(
                        concat_in_sh, 
                        self.params['concat_hidden_sizes'][0]), 
                    nn.ReLU(),]
                for i in range(1, len(self.params['concat_hidden_sizes'])):
                    concatclin_layers.append(nn.Linear(
                        self.params['concat_hidden_sizes'][i-1], 
                        self.params['concat_hidden_sizes'][i]))
                    concatclin_layers.append(nn.ReLU())
                concatclin_layers.append(nn.Linear(
                    self.params['concat_hidden_sizes'][-1], self.params['n_classes']))

            else:
                concatclin_layers = [
                    nn.Linear(
                        concat_in_sh,
                        self.params['n_classes']
                    )
                ]
        
        self.output_model = nn.Sequential(*concatclin_layers)   

        # Initialize used_params as an empty set
        self.used_params = set()

        # Register hooks for all parameters
        for name, param in self.named_parameters():
            param.register_hook(lambda grad, name=name: self.used_params.add(name))

    def forward(self,  **kwargs):
        data = kwargs['wsi_data']
                
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        x = self.fc(data.x)
        x_ = x 
        
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path)

        wsi_out = self.wsi_model(h)

        ## Concatenate the clinical data.
        full_feat_vec = torch.concat((wsi_out, kwargs['clin_data_list'][0]), axis=1)

        logits = self.output_model(full_feat_vec)

        # logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {
            'hazards': hazards, 
            'S': S, 
            'Y_hat': Y_hat,
            'wsi_feature': wsi_out,
            'other': {},
            }
        return results_dict
        