import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from repos.CMTA.models.cmta.util import initialize_weights
from repos.CMTA.models.cmta.util import NystromAttention
from repos.CMTA.models.cmta.util import BilinearFusion
from repos.CMTA.models.cmta.util import SNN_Block
from repos.CMTA.models.cmta.util import MultiheadAttention

def _series_intersection(s1, s2):
    return pd.Series(list(set(s1) & set(s2)))

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=2, #8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(features.device)
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).to(features.device)
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class CMTA(nn.Module):
    def __init__(self, params):
        super(CMTA, self).__init__()

        self.params = params

        ## Params, pathways setup =============================================
        
        self.n_classes = self.params['n_classes']
        self.fusion = self.params['fusion']
        ### Size maps of embeddings:
        self.size_dict = {
            "pathomics": {
                "small": [1024, 256, 256], "large": [1024, 512, 256],
                "very_small": [1024, 128],},
            "genomics": {
                "small": [256, 256], "large": [1024, 1024, 1024, 256],
                "very_small": [256, 128],}
        }
        self.model_size = 'small'

        ## Process Pathways Signatures File:
        self.signatures = pd.read_csv(self.params['pways_file'])
        self.omic_names = []
        self.omic_preint_sizes = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            self.omic_preint_sizes.append(len(omic))
            omic = sorted(_series_intersection(omic, self.params['omics_cols']))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

        ## omics preprocessing for captum
        all_gene_names = []
        for group in self.omic_names:
            all_gene_names.extend(group)
        all_gene_names = np.asarray(all_gene_names)

        all_gene_names = np.unique(all_gene_names)
        all_gene_names = list(all_gene_names)
        self.all_gene_names = all_gene_names

        ### Defining Network Parameters =======================================

        ### Pathomics Embedding Network 
        hidden = self.size_dict["pathomics"][self.model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.1))
        self.pathomics_fc = nn.Sequential(*fc)
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][self.model_size]
        sig_networks = []
        for input_dim in self.omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.1))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        # Decoder
        self.pathomics_decoder = Transformer_P(feature_dim=hidden[-1])

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=hidden[-1], num_heads=1)
        # G->P Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=hidden[-1], num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        # Decoder
        self.genomics_decoder = Transformer_G(feature_dim=hidden[-1])

        ### Classification Layer ==============================================
        concat_in_sh = (
            self.params['inputs_shapes']['clin'] + 
            hidden[-1]
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
            self.params['concat_hidden_sizes'][-1], self.n_classes))

        ### Fusion Component:
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
            # self.classifier = nn.Linear(
            #     hidden[-1] + self.params['inputs_shapes']['clin'], self.n_classes)
            self.classifier = nn.Sequential(*concatclin_layers)
        # elif self.fusion == "bilinear":
        #     self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        #     self.classifier = nn.Linear(
        #         hidden[-1] + self.params['inputs_shapes']['clin'], self.n_classes)
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        
        

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["wsi_data"]
        x_omic = kwargs['omics_pways_list']
        x_clin = kwargs['clin_data_list'][0]

        # Enbedding
        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat.squeeze()) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 1024]
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path.squeeze()).unsqueeze(0)

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        # cross-omics attention
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        ## decoder
        ## pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        ## genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        ### fusion + prediction
        if self.fusion == "concat":
            fused_rep = torch.concat((
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,), dim=1)
            # logits = self.mm(
            #     torch.concat((fused_rep, kwargs['clin_data'].squeeze()), dim=1)
            fused_rep = self.mm(fused_rep)
            all_features = torch.concat((fused_rep, x_clin), dim=1)
            logits = self.classifier(all_features)
            
        # elif self.fusion == "bilinear":
        #     fused_rep = self.mm(
        #         (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
        #         (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
        #     )  # take cls token to make prediction
        #     all_features = torch.concat((fused_rep, kwargs['clin_data']), dim=1)
        #     logits = self.classifier(all_features)  # [1, n_classes]
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        ## post-processing of logits
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)

        
        other_outs_dict = {
            'P': cls_token_pathomics_encoder,
            'P_hat': cls_token_pathomics_decoder,
            'G': cls_token_genomics_encoder,
            'G_hat': cls_token_genomics_decoder,
        }
        
        results_dict = {
            'hazards': hazards,
            'S': S,
            'Y_hat': Y_hat,
            'logits': logits,
            'wsi_feature': fused_rep,
            'other': other_outs_dict,
            # 'attn_pathways': attn_pathways,
            # 'cross_attn_pathways': cross_attn_pathways,
            # 'cross_attn_histology': cross_attn_histology,
        }

        return results_dict

    def pre_captum_omics_embed(self, x_omic_list):
        # genomics embedding
        genomics_features = [
            self.genomics_fc[idx].forward(sig_feat.squeeze()) for 
            idx, sig_feat in enumerate(x_omic_list)]
        
        return genomics_features

    def captum(self, x_path, clin_data_list, *inputs):
        # meta genomics and pathomics features
        
        clin_data = clin_data_list[0]
        genomics_features = inputs

        ## expand other inputs to match number of examples from genomics_features:
        x_path = x_path.repeat(genomics_features[0].shape[0], 1, 1)
        clin_data = clin_data.repeat(genomics_features[0].shape[0], 1,)

        ### Enbeddings
    
        ## genomics:
        genomics_features = torch.stack(genomics_features, dim=1)#.unsqueeze(0)  # [1, 6, 1024]
        ## pathomics: 
        pathomics_features = self.pathomics_fc(x_path)#.unsqueeze(0)

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        # cross-omics attention
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        ## decoder
        ## pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        ## genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        ### fusion + prediction
        if self.fusion == "concat":
            fused_rep = torch.concat((
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,), dim=1)
            logits = self.mm(
                torch.concat((fused_rep, clin_data), dim=1)
            )  # take cls token to make prediction
        # elif self.fusion == "bilinear":
        #     fused_rep = self.mm(
        #         (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
        #         (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
        #     )  # take cls token to make prediction
        #     all_features = torch.concat((fused_rep, clin_data), dim=1)
        #     logits = self.classifier(all_features)  # [1, n_classes]
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        ## post-processing of logits
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        

        return risk


    def captum_gene(self, x_path, clin_data_list, *inputs):
        # meta genomics and pathomics features
        
        clin_data = clin_data_list[0]
        
        x_omic_list = inputs

        ## expand other inputs to match number of examples from genomics_features:
        x_path = x_path.repeat(x_omic_list[0].shape[0], 1, 1)
        clin_data = clin_data.repeat(x_omic_list[0].shape[0], 1,)

        ### Embeddings
    
        ## genomics:
        genomics_features = [
            self.genomics_fc[idx].forward(sig_feat.squeeze()) for 
            idx, sig_feat in enumerate(x_omic_list)]

        genomics_features = torch.stack(genomics_features, dim=1)#.unsqueeze(0)  # [1, 6, 1024]
        ## pathomics: 
        pathomics_features = self.pathomics_fc(x_path)#.unsqueeze(0)

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        # cross-omics attention
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        ## decoder
        ## pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        ## genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        ### fusion + prediction
        if self.fusion == "concat":
            fused_rep = torch.concat((
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,), dim=1)
            logits = self.mm(
                torch.concat((fused_rep, clin_data), dim=1)
            )  # take cls token to make prediction
        # elif self.fusion == "bilinear":
        #     fused_rep = self.mm(
        #         (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
        #         (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
        #     )  # take cls token to make prediction
        #     all_features = torch.concat((fused_rep, clin_data), dim=1)
        #     logits = self.classifier(all_features)  # [1, n_classes]
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        ## post-processing of logits
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        

        return risk
