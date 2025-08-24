import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import Mlp
from einops import rearrange, reduce
from torch import nn, einsum
from timm.models.layers import trunc_normal_
import math
from collections import defaultdict 

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.1))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


# Thanks to open-source implementation of iRPE: https://github.com/microsoft/Cream/blob/76d03f6f7438388855df1cad62741721f990f9ac/iRPE/DETR-with-iRPE/models/rpe_attention/irpe.py#L19
@torch.no_grad()
def piecewise_index(relative_position, alpha=1.9, beta=1.9*4, gamma=1.9*6, shift=7, dtype=torch.int32):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha*2
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - 2*alpha)).round().clip(max=shift)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    idx[mask] = torch.sign(idx[mask])*1
    return idx

#---->WindowAttention
# Thanks to open-source implementation of Swin-Transformer: https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/models/swin_transformer.py#L77 
class WindowAttention(nn.Module):
    
    def __init__(self, dim=512, window_size=49, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = int(np.sqrt(window_size))
        self.num_heads = 8
        head_dim = dim // 8
        self.scale = head_dim ** -0.5

        #---->RelativePosition
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*self.shift+1, self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        #---->Attention
        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # save attention weights
        self.attention_weights = None
        self.coords_attention = None

    def forward(self, x, coords):
        B, N, C = x.shape #[b, n, c]

        #---->partition windows
        x = rearrange(x, 'b (w ws) c -> b w ws c', ws=self.window_size)
        x = rearrange(x, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]

        coords = rearrange(coords, 'b (w ws) c -> b w ws c', ws=self.window_size)
        coords = rearrange(coords, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]
        

        #---->attention
        B_, N, C = x.shape
        #[b*num_window, window, 3*C]->[b*num_window, window, 3, num_head, C//num_head]->[3, b*num_window, num_head, window, C//num_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2] #[b*num_window, num_head, window, C//num_head]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[num_window, num_head, window, window]

        max_L = coords.shape[1] #[num_window, window_size, 2]
        relative_coords = coords.view((-1, max_L, 1, 2))-coords.view((-1, 1, max_L, 2))
        relative_coords = relative_coords.int()
        relative_coords[:, :, :, 0] = piecewise_index(relative_coords[:, :, :, 0], shift=self.shift)
        relative_coords[:, :, :, 1] = piecewise_index(relative_coords[:, :, :, 1], shift=self.shift)
        relative_coords = relative_coords.abs()

        relative_position_index = relative_coords.sum(-1)  # num_window, Wh*Ww, Wh*Ww
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(-1, self.window_size, self.window_size, self.num_heads)
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()

        attn = attn + relative_position_bias

        self.attention_weights = attn.clone().detach().cpu()
        self.coords_attention = coords.clone().detach().cpu()

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v)
        
        x = out.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        #---->window_reverse
        x = rearrange(x, '(b w) ws c -> b (w ws) c', b=B)
        return x

#---->WindowAttention
class ShuffleWindowAttention(nn.Module):
    
    def __init__(self, dim=512, window_size=49, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = 8
        head_dim = dim // 8
        self.scale = head_dim ** -0.5

        #---->Attention
        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # save attention weights
        self.attention_weights_ShuffleWindowAttention = None
        self.coords_attention = None


    def forward(self, x, coords):
        B, N, C = x.shape #[b, n, c]

        #---->partition windows
        # Thanks to open-source implementation of Shuffle-Transformer: https://github.com/mulinmeng/Shuffle-Transformer/blob/8ba81eacf01314d4d26ff514f61badc7eebd33de/models/shuffle_transformer.py#L69
        x = rearrange(x, 'b (ws w) c -> b w ws c', ws=self.window_size)
        x = rearrange(x, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]

        coords = rearrange(coords, 'b (ws w) c -> b w ws c', ws=self.window_size)
        coords = rearrange(coords, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]
        self.coords_attention = coords.clone().detach().cpu()

        #---->attention
        B_, N, C = x.shape

        #[b*num_window, window, 3*C]->[b*num_window, window, 3, num_head, C//num_head]->[3, b*num_window, num_head, window, C//num_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #[b*num_window, num_head, window, C//num_head]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[num_window, num_head, window, window]
        
        self.attention_weights_ShuffleWindowAttention = attn.clone().detach().cpu()

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, '(b w) ws c -> b (ws w) c', b=B)
        return x


class LocalLayer(nn.Module):
    
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, window_size=49):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.wattn = WindowAttention(dim=dim, window_size=window_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

        # save attention weights
        self.attention_weights_LocalLayer = None
        self.coords_attention_LocalLayer = None

    def forward(self, x, coords):
        #---->pad
        h_ = x.shape[1]
        add_length = (h_//self.window_size)*self.window_size - h_ 
        if add_length != 0:
            x = rearrange(x, 'b n c -> b c n')
            coords = rearrange(coords, 'b n c -> b c n')
            #---->feature
            x = F.pad(input=x, pad=(add_length//2, add_length-add_length//2), mode='reflect') 
            x = rearrange(x, 'b c n -> b n c')
            #---->coords
            coords = F.pad(input=coords, pad=(add_length//2, add_length-add_length//2), mode='reflect') 
            coords = rearrange(coords, 'b c n -> b n c')
        #---->windowd
        x = x + self.wattn(self.norm1(x), coords)
        self.attention_weights_LocalLayer = self.wattn.attention_weights
        self.coords_attention_LocalLayer = self.wattn.coords_attention
        x = self.act(x)
        x = self.drop(x)
        return x

class HVTSurv(nn.Module):
    def __init__(self, params):
        super(HVTSurv, self).__init__()
        self.params = params
        self.n_classes = self.params['n_classes']
        self.dim_fc1 = 512

        ## initial dimension reduction step.
        self._fc1 = nn.Sequential(
            nn.Linear(
                self.params['inputs_shapes']['image'][1]-2, 
                self.dim_fc1), 
            nn.ReLU())
        
        self.layer1 = LocalLayer(
            dim=self.dim_fc1, window_size=self.params['window_size']
            )
        
        ## shuffle
        self.shiftwattn = ShuffleWindowAttention(
            dim = self.dim_fc1, window_size=self.params['window_size'])
        self.mlp1 = Mlp(
            in_features=self.dim_fc1, hidden_features=self.dim_fc1, 
            act_layer=nn.GELU, drop=0.1)
        self.norm1 = nn.LayerNorm(self.dim_fc1)
        self.norm2 = nn.LayerNorm(self.dim_fc1)
        self.norm3 = nn.LayerNorm(self.dim_fc1)
        self.attnpool = Attn_Net(
            L=self.dim_fc1, D=256, dropout = self.params['dropout'], 
            n_classes=1)

        ## Processed WSI dimension reduction module ===========================
        if self.params['wsi_hidden_sizes']:
            wsi_layers = [
                nn.Linear(
                    self.dim_fc1, 
                    self.params['wsi_hidden_sizes'][0]), 
                # nn.BatchNorm1d(self.params['wsi_hidden_sizes'][0]),
                nn.ReLU()]
            for i in range(1, len(self.params['wsi_hidden_sizes'])):
                wsi_layers.append(nn.Linear(
                    self.params['wsi_hidden_sizes'][i-1], self.params['wsi_hidden_sizes'][i]))
                # wsi_layers.append(nn.BatchNorm1d(self.params['wsi_hidden_sizes'][i]))
                wsi_layers.append(nn.ReLU())
            self.wsi_model = nn.Sequential(*wsi_layers)
            self.params['wsi_final_size'] = self.params['wsi_hidden_sizes'][-1]
        else: ## allowing for empty WSI model.
            self.wsi_model = nn.Sequential()
            self.params['wsi_final_size'] = self.dim_fc1

        ## Clinical concatenated to outputs module ============================
        concat_in_sh = (
                self.params['inputs_shapes']['clin'] + 
                self.params['wsi_final_size']
                )
        if bool(self.params['concat_hidden_sizes']):
            concatclin_layers = [
                nn.Linear(
                    concat_in_sh, 
                    self.params['concat_hidden_sizes'][0]), 
                # nn.BatchNorm1d(self.params['concat_hidden_sizes'][0]),
                nn.ReLU()]
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

    def forward(self, **kwargs):
    
        h_all = kwargs['wsi_data'] #list [[B,n,1026],...,[],[]]
        slide_ids = kwargs['slide_ids']

        feature_patient = []
        dic_attention_weights_LocalLayer = defaultdict()
        dic_coords_LocalLayer = defaultdict()
        dic_attention_weights_ShuffleWindow = defaultdict()
        dic_coords_ShuffleWindow = defaultdict()
        dic_attention_weights_PatientLayer = defaultdict()
        dic_coords_PatientLayer = defaultdict()

        for h, slide_id in zip(h_all, slide_ids): #All WSIs corresponding to a patient

            #---->Separate feature, coords information
            coords = h[:, :, :2].clone() #[n, 2]
            h = h[:, :, 2:] #[n, 1024]

            #---->Dimensionality reduction
            h = self._fc1(h) #[B, n, self.dim_fc1]

            #---->LocalLayer
            feature = self.layer1(h, coords)
            
            # get attention weights and coords for LocalLayer
            #attention
            slide_attention_weights_LocalLayer = self.layer1.attention_weights_LocalLayer
            slide_attention_weights_LocalLayer = rearrange(slide_attention_weights_LocalLayer.mean(dim=[1, 3]), 'num_w size_w -> (num_w size_w)').numpy()
            dic_attention_weights_LocalLayer[slide_id] = slide_attention_weights_LocalLayer
            # coords
            coords_attention_LocalLayer = self.layer1.coords_attention_LocalLayer
            coords_attention_LocalLayer = rearrange(coords_attention_LocalLayer, 'num_w size_w two_coords -> (num_w size_w) two_coords').numpy()
            dic_coords_LocalLayer[slide_id] = coords_attention_LocalLayer
  
            #---->shufflewindow
            feature = feature + self.shiftwattn(self.norm1(feature), torch.tensor(coords_attention_LocalLayer.reshape((1, *coords_attention_LocalLayer.shape))))

            # get attention weights and coords for ShuffleWindow
            # attention
            slide_attention_weights_ShuffleWindow = self.shiftwattn.attention_weights_ShuffleWindowAttention
            slide_attention_weights_ShuffleWindow = rearrange(slide_attention_weights_ShuffleWindow.mean(dim=[1, 3]), 'num_w size_w -> (num_w size_w)').numpy()
            dic_attention_weights_ShuffleWindow[slide_id] = slide_attention_weights_ShuffleWindow
            # coords
            coords_attention_ShuffleWindow = self.shiftwattn.coords_attention
            coords_attention_ShuffleWindow = rearrange(coords_attention_ShuffleWindow, 'num_w size_w two_coords -> (num_w size_w) two_coords').numpy()
            dic_coords_ShuffleWindow[slide_id] = coords_attention_ShuffleWindow
            
            feature = feature + self.mlp1(self.norm2(feature))

            feature_patient.append(feature)

        #---->concat sub-WSIs
        feature = torch.cat(feature_patient, dim=1)
        ## dim: (batch, n_patch, self.dim_fc1)

        
        #---->patient-level attention
        feature = self.norm3(feature) # B, N, C
        A, feature = self.attnpool(feature.squeeze(0))  # B C 1
        A = torch.transpose(A, 1, 0)
        
        # get attention weights and coords for patient-level attention
        attention_weights_PatientLayer = A.clone().detach().cpu().squeeze().numpy()
        start = 0
        for slide_id in slide_ids:
            # coords
            coords_attention_PatientLayer = dic_coords_LocalLayer[slide_id]
            dic_coords_PatientLayer[slide_id] = coords_attention_PatientLayer
            # attention
            end = start + coords_attention_PatientLayer.shape[0]
            dic_attention_weights_PatientLayer[slide_id] = attention_weights_PatientLayer[start:end]
        
        A = F.softmax(A, dim=1)
        feature = torch.mm(A, feature)
        ## dim: (batch, self.dim_fc1)

        if 'only_wsi_embedding' in kwargs.keys():
            return feature

        ## Further WSI feature dim reduct, then clinical concatenated model.
        wsi_out = self.wsi_model(feature)


        clin_all = kwargs['clin_data_list'][0]
        concat_clin = torch.concatenate((wsi_out, clin_all), axis=1)

        #---->predict output
        logits = self.output_model(concat_clin)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat,
                        'attention_weights': {'LocalLayer': dic_attention_weights_LocalLayer,
                                              'coords_LocalLayer': dic_coords_LocalLayer,
                                              'ShuffleWindow': dic_attention_weights_ShuffleWindow,
                                              'coords_ShuffleWindow': dic_coords_ShuffleWindow,
                                              'PatientLayer': dic_attention_weights_PatientLayer,
                                              'coords_PatientLayer': dic_coords_PatientLayer},
                        'wsi_out': wsi_out,
                        'wsi_feature': feature,
                        'other': {},
                        }

        return results_dict
        
