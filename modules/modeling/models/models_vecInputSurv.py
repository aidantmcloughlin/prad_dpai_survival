import torch
from torch import nn


class vecInputSurv(nn.Module):
    def __init__(self, params):
        super(vecInputSurv, self).__init__()

        self.params = params
        
        ### Creating a sequential container
        ## WSI Component ======================================================
        if self.params['wsi_hidden_sizes']:
            wsi_layers = [
                nn.Linear(
                    self.params['inputs_shapes']['image'][0], 
                    self.params['wsi_hidden_sizes'][0]), 
                nn.BatchNorm1d(self.params['wsi_hidden_sizes'][0]),
                nn.ReLU(),
                nn.Dropout(0.1), ]
            for i in range(1, len(self.params['wsi_hidden_sizes'])):
                wsi_layers.append(nn.Linear(
                    self.params['wsi_hidden_sizes'][i-1], self.params['wsi_hidden_sizes'][i]))
                wsi_layers.append(nn.BatchNorm1d(self.params['wsi_hidden_sizes'][i]))
                wsi_layers.append(nn.ReLU())
                wsi_layers.append(nn.Dropout(0.1))

            self.wsi_out_dim = self.params['wsi_hidden_sizes'][-1]
        else:
            wsi_layers = []
            self.wsi_out_dim = self.params['inputs_shapes']['image'][0]
        self.wsi_model = nn.Sequential(*wsi_layers)

        ## Concat Clin Component ==============================================
        concat_in_sh = (
            self.params['inputs_shapes']['clin'] + 
            self.wsi_out_dim
            )
        concatclin_layers = [
            nn.Linear(
                concat_in_sh, 
                self.params['concat_hidden_sizes'][0]), 
            nn.BatchNorm1d(self.params['concat_hidden_sizes'][0]),
            nn.ReLU()]
        for i in range(1, len(self.params['concat_hidden_sizes'])):
            concatclin_layers.append(nn.Linear(
                self.params['concat_hidden_sizes'][i-1], 
                self.params['concat_hidden_sizes'][i]))
            concatclin_layers.append(nn.BatchNorm1d(
                self.params['concat_hidden_sizes'][i]))
            concatclin_layers.append(nn.ReLU())
        concatclin_layers.append(nn.Linear(
            self.params['concat_hidden_sizes'][-1], self.params['n_classes']))
        
        self.output_model = nn.Sequential(*concatclin_layers)
        
        

    def forward(self, **kwargs):
        wsi_out = self.wsi_model(kwargs['wsi_data'])
        concat_clin = torch.concatenate((wsi_out, kwargs['clin_data_list'][0]), axis=1)
        logits = self.output_model(concat_clin)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        results_dict = {
            'hazards': hazards,
            'S': S,
            'Y_hat': Y_hat,
            'logits': logits,
            'wsi_feature': wsi_out,
            'other': {},
        }

        return results_dict
