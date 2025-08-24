### Load modules ==============================================================
from pathlib import Path

import yaml
from addict import Dict
import os, shutil
import re, copy
from sklearn.model_selection import GridSearchCV
import glob
import torch
import torch.nn as nn
import torch_geometric
from torch.utils.data import Subset, DataLoader
from captum.attr import IntegratedGradients
import pandas as pd
import numpy as np
from itertools import product
from lifelines.statistics import logrank_test
import openslide

from pytorch_lightning import loggers as pl_loggers

from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer)
from sklearn.pipeline import Pipeline

def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)



### YAML Config Updater =======================================================
def update_yaml(yaml_file_path, **kwargs):

    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_config_file = yaml.safe_load(file)

    # Change the 'alpha_surv' value under 'Loss'
    if 'num_workers' in kwargs.keys():
        yaml_config_file['Data']['train_dataloader']['num_workers'] = kwargs['num_workers']
        yaml_config_file['Data']['test_dataloader']['num_workers'] = kwargs['num_workers']
    if 'epochs' in kwargs.keys():
        yaml_config_file['General']['epochs'] = kwargs['epochs']
    if 'patience' in kwargs.keys():
        yaml_config_file['General']['patience'] = kwargs['patience']
    if 'lr' in kwargs.keys():
        yaml_config_file['Optimizer']['lr'] = kwargs['lr']
    if 'n_bins' in kwargs.keys():
        yaml_config_file['ModelInfo']['n_classes'] = kwargs['n_bins']
    ## paths:
    if 'log_path' in kwargs.keys():
        yaml_config_file['Data']['log_path'] = kwargs['log_path']
    if 'label_dir' in kwargs.keys():
        yaml_config_file['Data']['label_dir'] = kwargs['label_dir']
    
    if 'wsifeat_dir' in kwargs.keys():
        yaml_config_file['Data']['wsifeat_dir'] = kwargs['wsifeat_dir']

    
    if 'pways_file' in kwargs.keys():
        yaml_config_file['Data']['pways_file'] = kwargs['pways_file']

    if 'omics_file' in kwargs.keys():
        yaml_config_file['Data']['omics_file'] = kwargs['omics_file']
    
    ## Loss Params:
    if 'alpha_surv' in kwargs.keys():
        yaml_config_file['Loss']['alpha_surv'] = kwargs['alpha_surv'] 
    if 'l1sim_alpha' in kwargs.keys():
        yaml_config_file['Loss']['l1sim_alpha'] = kwargs['l1sim_alpha'] 

    ## clinical var set:
    if 'clin_vars' in kwargs.keys():
        yaml_config_file['ModelInfo']['params']['clin_vars'] = kwargs['clin_vars'] 

    ## Model Params:
    if 'agg_type' in kwargs.keys():
        yaml_config_file['ModelInfo']['params']['agg_type'] = kwargs['agg_type']
    if 'fusion' in kwargs.keys():
        yaml_config_file['ModelInfo']['params']['fusion'] = kwargs['fusion']

     
    # Write the modified config back to the YAML file
    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(yaml_config_file, file, default_flow_style=False, sort_keys=False)

### checkers for magnification, paths =========================================

## Getting Locations of features for the various model types:
def check_magnification(
    slides_dir,
    patch_size,
    magnif
    ):

    ex_svs = os.path.join(
        slides_dir,
        [os.path.basename(f) for f in glob.glob(f"{slides_dir}/*")][0]
        # os.listdir(slides_dir)[0]
        )

    ## read file, get magnification.
    slide = openslide.OpenSlide(ex_svs)
    slide_mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    
    if int(slide_mag) != int(magnif):
        new_magnif = slide_mag
        new_patch_size = int(patch_size * (int(slide_mag) / int(magnif)))

        print("Slide magnitude not matching magnitude input. Adjusting patch size.")
        print("NEW MAG: " + new_magnif + "....NEW PATCH SIZE: " + str(new_patch_size))

        ## set to new params
        patch_size = new_patch_size
        magnif = new_magnif

    return patch_size, magnif


def get_wsifeature_and_labels_paths(
    feat_root_dir, 
    label_root_dir,
    encoder, model, patch_size, 
    data_seed,
    batch_name = 'batchstrat', 
    magnif = 40):


    ## these paths are RELATIVE to the data parent directory for the project.
    patch_mag_dir = os.path.join(feat_root_dir, str(patch_size) + '_' + str(magnif))
    encoder_dir = os.path.join(patch_mag_dir, encoder)

    if model == 'hvtsurv':
        wsi_feature_dir = os.path.join(encoder_dir, '_4_mask_window', 'seed'+str(data_seed), '_3_knn_2_0.5_random')
        label_dir = os.path.join(encoder_dir, '_4_mask_window', 'seed'+str(data_seed), batch_name + '_2')
    elif (model == 'CLAM' or 'CMTA' in model):
        wsi_feature_dir = os.path.join(encoder_dir, '_2_features', 'pt_files')
        label_dir = os.path.join(label_root_dir, 'seed'+str(data_seed), batch_name)
    elif model == 'clinonly':
        wsi_feature_dir = ''
        label_dir = os.path.join(label_root_dir, 'seed'+str(data_seed), batch_name)
    elif model == 'globavg':
        wsi_feature_dir = os.path.join(encoder_dir, 'globavgs')
        label_dir = os.path.join(label_root_dir, 'seed'+str(data_seed), batch_name)
    elif model == 'patchGCN':
        wsi_feature_dir = os.path.join(encoder_dir, 'pt_graphs')
        label_dir = os.path.join(label_root_dir, 'seed'+str(data_seed), batch_name)
    else:
        raise ValueError("haven't added pathing for this model yet.")

    return wsi_feature_dir, label_dir


### data preprocessing ========================================================
def preprocess_clin_data(
    clin_data, cols_keep, idx_col='case_id',
    train_preprocessor = None):
    
    if idx_col in list(clin_data.columns):
        ## confirm it is uq per row
        if len(np.unique(clin_data[idx_col].values)) != clin_data.shape[0]:
            raise ValueError("Idx column is not unique by row.")
        ## set it to row index.
        clin_data.set_index('case_id', inplace=True)
    else:
        raise ValueError("Idx column is not present in the Clin DF.")

    ## select it down to chosen clin columns
    clin_data = clin_data.loc[:, cols_keep]

    if clin_data.shape[1] > 0:
        cat_cols = list(clin_data.columns[[clin_data[col].dtype in ('object', 'string') for col in (
            list(clin_data.columns))
            ]])
        binary_cols = list(clin_data.columns[
            clin_data.apply(lambda col: col.dropna().isin([0, 1]).all())
            ])
        float_cols = [
            item for item in cols_keep if 
            item not in cat_cols+binary_cols]


        ## construct the scikit transformer, if needed
        if train_preprocessor is None:
            train_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), float_cols),
                    ('cat', OneHotEncoder(drop='first'), cat_cols),
                    ('binary', FunctionTransformer(validate=False), binary_cols)
                ]
            )
            ## fit transformer to the data
            train_preprocessor.fit(clin_data)

        ## return modeling data col names:
        if len(cat_cols) > 0:
            onehot_columns = train_preprocessor.named_transformers_['cat'].get_feature_names_out(
                input_features=cat_cols)

            # Combine all feature names
            model_data_colnames = float_cols + list(onehot_columns) + list(binary_cols)
        else:
            model_data_colnames = float_cols + list(binary_cols)
        ## Create transformed DF
        clin_model_data = train_preprocessor.transform(clin_data)
        clin_model_data = pd.DataFrame(
            clin_model_data, 
            index=clin_data.index,
            columns=model_data_colnames)

    else:
        clin_model_data = clin_data
        train_preprocessor = None

    return clin_model_data, train_preprocessor



def compare_dicts(dict1, dict2, path=""):
    differences = []

    # Check keys in dict1
    for key in dict1:
        if key not in dict2:
            differences.append(f"{path}.{key} only in dict1")
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(compare_dicts(dict1[key], dict2[key], path + "." + str(key)))
            elif dict1[key] != dict2[key]:
                differences.append(f"Difference at {path}.{key}: {dict1[key]} != {dict2[key]}")

    # Check keys in dict2
    for key in dict2:
        if key not in dict1:
            differences.append(f"{path}.{key} only in dict2")

    return differences

## multiple arrays intersection:
def intersect(arr1, arr2):
    return np.intersect1d(arr1, arr2)

###############################################################################
### Functions to run post-Hoc analysis on trained DPAI features
###############################################################################

def collect_all_data(data_module, stage='train'):
    time_labels = []
    censorships = []
    clin_feats_all = []
    all_case_ids = []
    actual_times = []
    # voxel_feats_all = []
    
    # Iterate over the DataLoader (assuming you want to collect from train, val, and test sets)
    if stage =='train':
        dls = [data_module.train_dataloader(shuffle=False), ]
    elif stage =='valid':
        dls = [data_module.val_dataloader(shuffle=False), ]
    elif stage == 'test':
        dls = [data_module.test_dataloader(shuffle=False),]

    for dataloader in dls:
        for batch in dataloader:
            (case_ids, slide_ids, 
            data_WSI, patch_mask,
            data_clin_list, omics_pways_list,
            label, event_time, censorship) = batch
            time_labels.append(label)
            censorships.append(censorship)
            clin_feats_all.append(data_clin_list[0])
            actual_times.append(event_time)
            all_case_ids.extend(case_ids)
    
    # Concatenate all the collected tensors
    all_event_times = torch.cat(time_labels, dim=0).numpy()
    all_censorships = torch.cat(censorships, dim=0).numpy()
    all_clin_feats = torch.cat(clin_feats_all, dim=0).numpy()
    all_actual_times = torch.cat(actual_times, dim=0).numpy()

    
    return all_event_times, all_censorships, all_clin_feats, all_case_ids, all_actual_times

def make_vox_feats_df(case_ids, vox_feats):
    vox_feats_df = pd.concat((
                pd.DataFrame(case_ids, columns=['case_id']),
                pd.DataFrame(
                    vox_feats, columns=[f"dpaifeat_{i}" for i in range(vox_feats.shape[1])]
                )
                ), axis=1)
    return vox_feats_df

def embed_all_data(model, data_module, stage='train'):
    if stage =='train':
        dls = [data_module.train_dataloader(shuffle=False), ]
    elif stage =='valid':
        dls = [data_module.val_dataloader(shuffle=False), ]
    elif stage == 'test':
        dls = [data_module.test_dataloader(shuffle=False),]

    all_vox_feats=[]
    all_risk_scores=[]
    all_case_ids=[]
    vox_feats_flat =[]
    model.eval()  # Set the model to evaluation mode

    for dataloader in dls:
        for batch in dataloader:
            (case_ids, slide_ids, 
            data_WSI, patch_mask,
            data_clin_list, omics_pways_list,
            label, event_time, censorship) = batch
            
            if 'torch_geometric' in type(data_WSI).__module__:
                data_WSI = data_WSI.to(model.device)
            elif model.model.__class__.__name__ == 'vecInputSurv':
                data_WSI = data_WSI.to(model.device)
                pass
            else:
                data_WSI = [t.to(model.device) for t in data_WSI]
            data_clin_list[0] = data_clin_list[0].to(model.device)
            if len(data_clin_list) > 1:
                data_clin_list[1] = data_clin_list[1].to(model.device)

            with torch.no_grad():
                if 'torch_geometric' in type(data_WSI).__module__:
                    results_dict = model.model(
                        wsi_data=data_WSI, 
                        clin_data_list=data_clin_list,
                        slide_ids = slide_ids
                        )
                elif model.model.__class__.__name__ == 'vecInputSurv':
                    results_dict = model.model(
                        wsi_data=data_WSI, 
                        clin_data_list=data_clin_list,
                        slide_ids = slide_ids
                        )
                elif model.model.__class__.__name__ == 'HVTSurv':
                    results_dict = model.model(
                        wsi_data=data_WSI[0].unsqueeze(0), 
                        clin_data_list=data_clin_list,
                        slide_ids = [slide_ids[0][0]]
                        )
                elif bool(re.search('CMTA', model.model.__class__.__name__)):
                    omics_pways_list = [t.to(model.device) for t in omics_pways_list]
                    results_dict = model.model(
                        wsi_data=data_WSI[0].unsqueeze(0), 
                        omics_pways_list=omics_pways_list,
                        clin_data_list=data_clin_list,
                        slide_ids = [slide_ids[0][0]],
                        return_attn=False,
                        )
                else:
                    results_dict = model.model(
                        wsi_data=data_WSI[0].unsqueeze(0), 
                        clin_data_list=data_clin_list,
                        slide_ids = slide_ids
                        )
            
            
            ## clear cache:
            del (data_WSI, data_clin_list)
            torch.cuda.empty_cache()
    
            ## compute risks
            risks = -torch.sum(results_dict['S'], dim=1).detach().cpu().numpy()
            ## concatenate
            if results_dict['wsi_feature'] is not None:
                if model.model.__class__.__name__ != 'vecInputSurv':
                    wsi_vector = results_dict['wsi_feature'].reshape(-1).detach().cpu().numpy()
                else:
                    wsi_vector = results_dict['wsi_feature'].detach().cpu().numpy()

                all_vox_feats.append(wsi_vector)
                all_case_ids.extend(case_ids)
                all_risk_scores.extend(risks.tolist())

            else:
                all_vox_feats = None
    
    if model.model.__class__.__name__ not in ['vecInputSurv', 'clinOnlySurv']:
        all_vox_feats = np.concatenate([a.reshape((1, -1)) for a in all_vox_feats], axis=0)
    else:
        all_vox_feats = None

    return all_vox_feats, all_risk_scores, all_case_ids


def load_loggers(cfg):

    log_path = cfg.Data.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    
    cfg.log_path = str(Path(log_path) / f'fold{cfg.Data.fold}')

    csv_log_path = os.path.join(log_path, 'csv_res', f'fold{cfg.Data.fold}')
    tb_log_path = os.path.join(log_path, 'tb_res', f'fold{cfg.Data.fold}')
    
    ## Check if the last model run completed.
    test_res_exists = os.path.isfile(
        os.path.join(cfg.log_path, 'result.csv')
        )

    do_warm_start = not test_res_exists
    completed_run = test_res_exists
    
    print("ASSESSING WHETHER RUN IS A WARM START:  " + str(do_warm_start))
    ## Clear log paths, if training (and there is already a test set results file.):
    if cfg.General.server == "train" and not do_warm_start and not completed_run:
        print("CLEARING ANY OLD LOG PATHS.")
        if os.path.isdir(cfg.log_path):
            shutil.rmtree(cfg.log_path)
        if os.path.isdir(csv_log_path):
            shutil.rmtree(csv_log_path)
        if os.path.isdir(tb_log_path):
            shutil.rmtree(tb_log_path)
    print(f'---->Log dir: {cfg.log_path}')

    ## Convert log_path to PosixPath:
    cfg.log_path = Path(cfg.log_path)
    
    #---->TensorBoard (only if training)
    if log_path.endswith('/'):
        tb_name = os.path.basename(log_path[:-1])
    else:
        tb_name = os.path.basename(log_path)
    tb_logger = pl_loggers.TensorBoardLogger(
        log_path,
        name = "tb_res",
        version = f'fold{cfg.Data.fold}',
        log_graph = False, default_hp_metric = False)
    
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(
        log_path,
        name = "csv_res", 
        version = f'fold{cfg.Data.fold}',)
    
    return ([tb_logger, csv_logger,], completed_run)


#---->Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=False,
        mode='min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = False))
        
    return Mycallbacks


#---->

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


#---->loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):

    def __init__(self, alpha_surv=0.15):
        self.alpha = alpha_surv

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = -1 if T_cont \in (-inf, 0), Y = 0 if T_cont \in [0, a_1),  Y = 1 if T_cont in [a_1, a_2), ..., Y = k-1 if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = -1,0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
# h(-1) = 0 ---> do not need to model
# S(-1) = P(Y > -1 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7, take_mean=True):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).half() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1) 
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + 
        torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
    censored_loss = - c * torch.log(
        torch.gather(S_padded, 1, Y+1).clamp(min=eps)
        )
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    
    if take_mean:
        loss = loss.mean()
    
    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha_surv=0.15):
        self.alpha = alpha_surv

    def __call__(self, res_dict, alpha=None, take_mean=True, ):
        if alpha is None:
            surv_loss = nll_loss(
                res_dict['hazards'], 
                res_dict['S'], 
                res_dict['Y'], 
                res_dict['c'], 
                alpha=self.alpha, take_mean=take_mean)
        else:
            surv_loss = nll_loss(
                res_dict['hazards'], 
                res_dict['S'], 
                res_dict['Y'], 
                res_dict['c'], 
                alpha=alpha, take_mean=take_mean)

        return surv_loss, {'surv_loss': surv_loss, 'ae_loss': torch.Tensor(0)}


class CMTALoss(object):
    def __init__(self, alpha_surv=0.15, l1sim_alpha=1):
        self.alpha = alpha_surv
        self.l1sim_alpha = l1sim_alpha
        self.sim_loss_func = nn.L1Loss(reduction='mean')

    def __call__(self, res_dict, alpha=None, take_mean=True, ):
        if alpha is None:
            surv_loss = nll_loss(
                res_dict['hazards'], 
                res_dict['S'], 
                res_dict['Y'], 
                res_dict['c'], 
                alpha=self.alpha, take_mean=take_mean)
        else:
            surv_loss = nll_loss(
                res_dict['hazards'], 
                res_dict['S'], 
                res_dict['Y'], 
                res_dict['c'], 
                alpha=alpha, take_mean=take_mean)

        ## similarity loss
        sim_loss_P = self.sim_loss_func(
            res_dict['other']['P'].detach(), res_dict['other']['P_hat']
            )
        sim_loss_G = self.sim_loss_func(
            res_dict['other']['G'].detach(), res_dict['other']['G_hat']
            )
        ## check for gleason-fused representation
        if 'gleason_token' in res_dict['other'].keys():
            sim_loss_P_gleason = self.sim_loss_func(
                res_dict['other']['gleason_token'].detach(), res_dict['other']['P_hat_gleason']
            )
            sim_loss_G_gleason = self.sim_loss_func(
                res_dict['other']['gleason_token'].detach(), res_dict['other']['G_hat_gleason']
            )
            sim_loss = self.l1sim_alpha * (
                sim_loss_P + sim_loss_G + sim_loss_P_gleason + sim_loss_G_gleason
                )
        elif 'P_hat_gleason' in res_dict['other'].keys():
            sim_loss_P_gleason = self.sim_loss_func(
                res_dict['other']['P'].detach(), res_dict['other']['P_hat_gleason']
            )
            sim_loss_G_gleason = self.sim_loss_func(
                res_dict['other']['G'].detach(), res_dict['other']['G_hat_gleason']
            )
            sim_loss = self.l1sim_alpha * (
                sim_loss_P + sim_loss_G + sim_loss_P_gleason + sim_loss_G_gleason
                )
        else:
            sim_loss = self.l1sim_alpha * (sim_loss_P + sim_loss_G)

        return surv_loss, {'surv_loss': surv_loss, 'ae_loss': torch.Tensor(0)}

class NLLSurvAELoss(object):
    def __init__(self, alpha_surv=0.15, kl_weight = 1):
        self.alpha = alpha_surv
        self.kl_weight = 1

    def __call__(self, res_dict, alpha=None, take_mean=True):
        if alpha is None:
            surv_loss = nll_loss(
                res_dict['hazards'], 
                res_dict['S'], 
                res_dict['Y'], 
                res_dict['c'], 
                alpha=self.alpha, take_mean=take_mean)
        else:
            surv_loss = nll_loss(
                res_dict['hazards'], 
                res_dict['S'], 
                res_dict['Y'], 
                res_dict['c'], 
                alpha=alpha, take_mean=take_mean)

        recon_loss = nn.MSELoss()(res_dict['wsi_recon'], res_dict['wsi_feature'])
        kl_loss = -0.5 * torch.sum(
            1 + res_dict['wsi_latent_logvar'] - 
            res_dict['wsi_latent_mu'].pow(2) - res_dict['wsi_latent_logvar'].exp())

        kl_loss = kl_loss / res_dict['wsi_latent_mu'].size(0)

        ae_loss = recon_loss + self.kl_weight * kl_loss

        return surv_loss + ae_loss, {'surv_loss': surv_loss, 'ae_loss': ae_loss}


class CoxSurvLoss(object):

    def __init__(self):
        pass

    def __call__(self, hazards, S, Y, c, alpha=None):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox


#---->metrics

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata < median] = 1
    survtime_all = survtime_all.reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)

#---->test

def _predictions_to_pycox(data, time_points=None):
    # predictions = {k: v['probabilities'] for k, v in data}
    predictions = {index: data[index][0] for index in range(len(data))}
    df = pd.DataFrame.from_dict(predictions)

    # Use predictions at same "time_points" for all models
    # Use MultiSurv's default output interval midpoints as default
    if time_points is None:
        time_points = np.arange(0, 4, 1)

    # Replace automatic index by time points
    df.insert(0, 'time', time_points)
    df = df.set_index('time')

    return df


### Integrated Gradients methods ==============================================
def getPathwayIGs(model, test_risk_scores_df):
    high_risk_idx = np.argmax(test_risk_scores_df['risk_score'])
    low_risk_idx = np.argmin(test_risk_scores_df['risk_score'])
    high_risk = test_risk_scores_df['case_id'][high_risk_idx]
    low_risk = test_risk_scores_df['case_id'][low_risk_idx]

    ## subset dataloader to only the cases we are examining:
    ds = model.surv_dm.test_dataloader().dataset
    subset = Subset(ds, [low_risk_idx, high_risk_idx])
    sub_loader = DataLoader(subset, batch_size=1)
        
    model.model.eval()
    attribution_dfs = []
    for batch in sub_loader:
        (case_ids, slide_ids, 
        data_WSI, patch_mask,
        data_clin_list, omics_pways_list,
        label, event_time, censorship) = batch

        genomics_features = model.model.pre_captum_omics_embed(omics_pways_list)

        wrapped_forward = lambda *inputs: model.model.captum(
            data_WSI, data_clin_list, *inputs)
        ig = IntegratedGradients(wrapped_forward)
        
        inputs = tuple(t.unsqueeze(0).detach().clone().requires_grad_() for t in genomics_features)
        baselines = tuple(torch.zeros_like(t) for t in inputs)
        attributions = ig.attribute(inputs, baselines=baselines, n_steps=15)

        pathway_attribution = [torch.sum(a).detach().item() for a in attributions[0:]]
        pathway_mean_attribution = [torch.mean(a).detach().item() for a in attributions[0:]]
        pathway_importance = [torch.sum(torch.abs(a)).detach().item() for a in attributions[0:]]
        pathway_mean_importance = [torch.mean(torch.abs(a)).detach().item() for a in attributions[0:]]
        attribution_dfs.append(pd.DataFrame({
            'case_id': case_ids[0],
            'is_high_risk': case_ids[0] == high_risk,
            'is_low_risk': case_ids[0] == low_risk,
            'pathway_name': model.model.signatures.columns,
            'pathway_attribution': pathway_attribution,
            'pathway_importance': pathway_importance,
            'pathway_mean_attribution': pathway_mean_attribution,
            'pathway_mean_importance': pathway_mean_importance,}))
            
    attribution_df = pd.concat(attribution_dfs, axis=0)
    attribution_df.to_csv(os.path.join(model.log_path, 'pathway_attributions.csv'), index=False)

    return attribution_df

def getGeneIGs(model, test_risk_scores_df):
    high_risk_idx = np.argmax(test_risk_scores_df['risk_score'])
    low_risk_idx = np.argmin(test_risk_scores_df['risk_score'])
    high_risk = test_risk_scores_df['case_id'][high_risk_idx]
    low_risk = test_risk_scores_df['case_id'][low_risk_idx]

    ## subset dataloader to only the cases we are examining:
    ds = model.surv_dm.test_dataloader().dataset
    subset = Subset(ds, [low_risk_idx, high_risk_idx])
    sub_loader = DataLoader(subset, batch_size=1)
        
    model.model.eval()
    attribution_dfs = []
    for batch in sub_loader:
        (case_ids, slide_ids, 
        data_WSI, patch_mask,
        data_clin_list, omics_pways_list,
        label, event_time, censorship) = batch

        wrapped_forward = lambda *inputs: model.model.captum_gene(
            data_WSI, data_clin_list, *inputs)
        ig = IntegratedGradients(wrapped_forward)
        
        inputs = tuple(t.unsqueeze(0).detach().clone().requires_grad_() for t in omics_pways_list)
        baselines = tuple(torch.zeros_like(t) for t in inputs)
        attributions = ig.attribute(inputs, baselines=baselines, n_steps=15)

        gene_attribution = [list(a[0].squeeze().detach().numpy()) for a in attributions[0:]]
        gene_names = [
            [f"{model.model.signatures.columns[i]}: {n}" for n in model.model.omic_names[i]] for i in 
            range(len(attributions))
            ]

        gene_attr_flat = [item for sublist in gene_attribution for item in sublist]
        gene_names_flat = [item for sublist in gene_names for item in sublist]

        attribution_dfs.append(pd.DataFrame({
            'case_id': case_ids[0],
            'is_high_risk': case_ids[0] == high_risk,
            'is_low_risk': case_ids[0] == low_risk,
            'gene_name': gene_names_flat,
            'gene_attribution': gene_attr_flat,}))
            
    attribution_df = pd.concat(attribution_dfs, axis=0)
    attribution_df.to_csv(os.path.join(model.log_path, 'gene_attributions.csv'), index=False)

    return attribution_df
