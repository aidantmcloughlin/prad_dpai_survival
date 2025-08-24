# Adapted from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
# Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes"""

    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()

        att_a = [nn.Linear(L, D), nn.Tanh()]

        att_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            att_a.append(nn.Dropout(p_dropout_atn))
            att_b.append(nn.Dropout(p_dropout_atn))

        self.attention_a = nn.Sequential(*att_a)
        self.attention_b = nn.Sequential(*att_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM(nn.Module):
    def __init__(
        self,
        params,
        model_size="small",
        input_feature_size=1024,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    ):
        super(CLAM, self).__init__()
        
        self.params = params
        self.n_classes = self.params['n_classes']



        size_dict = {
            "micro": [input_feature_size, 384, 128],
            "tiny": [input_feature_size, 384, 256],
            "small": [input_feature_size, 512, 256],
            "big": [input_feature_size, 512, 384],
        }
        size = size_dict[model_size]

        # From experiments, adding a first FC layer to reduce dimension helps even when size[0]==size[1].
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]

        if dropout:
            fc.append(nn.Dropout(p_dropout_fc))

        fc.append(
            # The attention network has a head for each class.
            Attn_Net_Gated(
                L=size[1],
                D=size[2],
                dropout=dropout,
                p_dropout_atn=p_dropout_atn,
                n_classes=self.n_classes,
            )
        )
        self.attention_net = nn.Sequential(*fc)


        ## Processed WSI dimension reduction module ===========================
        if self.params['wsi_hidden_sizes']:
            wsi_layers = [
                nn.Linear(
                    size[1], 
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

        ## Concat Clin Component ==============================================
        if bool(self.params['wsi_hidden_sizes']):
            self.wsi_out_dim = self.params['wsi_hidden_sizes'][-1]
        else:
            self.wsi_out_dim = size[1]

        concat_in_sh = (
            self.params['inputs_shapes']['clin'] + 
            self.wsi_out_dim
            )
        concatclin_layers = [
            nn.Linear(
                concat_in_sh, 
                self.params['concat_hidden_sizes'][0]), 
            nn.ReLU()]
        for i in range(1, len(self.params['concat_hidden_sizes'])):
            concatclin_layers.append(nn.Linear(
                self.params['concat_hidden_sizes'][i-1], 
                self.params['concat_hidden_sizes'][i]))
            concatclin_layers.append(nn.ReLU())
        concatclin_layers.append(nn.Linear(
            self.params['concat_hidden_sizes'][-1], 1))
        

        # We use an independent linear layer/s to predict each logit.
        self.classifiers = nn.ModuleList(
            [nn.Sequential(*concatclin_layers) for i in range(self.n_classes)]
        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, **kwargs):

        if kwargs['wsi_data'].shape[0] != 1:
            raise ValueError('Model designed to compute 1 case per batch')
        
        A_, h = self.attention_net(kwargs['wsi_data'].squeeze())  # NxK
        A_raw = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_raw, dim=1)  # softmax over N
        M = torch.mm(A, h)  # recompute slide embeddings

        M = self.wsi_model(M)

        # We have one attention score per class [discrete survival time].
        logits = torch.empty(1, self.n_classes).float().to(h.device)
        
        for c in range(self.n_classes):
            ## concatenate with clinical information
            concat_clin = torch.concatenate(
                (M[c], kwargs['clin_data_list'][0].squeeze()), axis=0)
            logits[0, c] = self.classifiers[c](concat_clin.unsqueeze(0))

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        results_dict = {
            'hazards': hazards,
            'S': S,
            'Y_hat': Y_hat,
            'logits': logits,
            'wsi_feature': M,
            'other': {},
        }
        return results_dict
