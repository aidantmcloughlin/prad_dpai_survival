import os, sys
import numpy as np
import re
import copy
from pathlib import Path
import inspect
import importlib
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict 
import pickle as pkl

#---->
from optims import create_optimizer
from utils_survival import (
    NLLSurvLoss, NLLSurvAELoss, CrossEntropySurvLoss, CMTALoss,
    cox_log_rank, _predictions_to_pycox, CoxSurvLoss,
    collect_all_data, embed_all_data,
    make_vox_feats_df,
    )
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_curve, auc
# from pycox.evaluation import EvalSurv
from sklearn.utils import resample


#---->
import torch
import torch.nn as nn
import torch.nn.functional as F

#---->
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, modelinfo, loss_config, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        
        self.loss = globals()[loss_config['loss_name']](
            **{k: v for k, v in loss_config.items() if k != 'loss_name'})
        self.optimizer = optimizer
        self.n_classes = modelinfo.params['n_classes']
        self.log_path = kargs['log']
        self.train_attention = defaultdict()
        self.valid_attention = defaultdict()
        self.test_attention = defaultdict()
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.train_losses = []
        
        self.model_class_name = self.model.__class__.__name__
        self.is_hvt_model = bool(re.search('HVT', self.model_class_name))
        self.is_ig_model = bool(re.search('CMTA', self.model_class_name))

        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #--->random
        self.shuffle = True
        self.count = 0

    #---->Remove v_num from the progress bar
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def compute_loss_and_risk(self, results_dict, label, c, take_mean = True):
        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']
        wsi_feature = results_dict['wsi_feature']
        if 'wsi_recon' in results_dict.keys():
            wsi_recon = results_dict['wsi_recon']
        else:
            wsi_recon = None
        if 'wsi_latent_logvar' in results_dict.keys():
            wsi_latent_mu = results_dict['wsi_out']
            wsi_latent_logvar = results_dict['wsi_latent_logvar']
        else:
            wsi_latent_mu = None
            wsi_latent_logvar = None

        
        #---->calculate loss
        loss, loss_dict = self.loss(
            {"hazards":hazards, "S":S, "Y":label.long(), "c":c,
            "wsi_recon":wsi_recon, "wsi_feature":wsi_feature, 
            "wsi_latent_mu":wsi_latent_mu, "wsi_latent_logvar":wsi_latent_logvar,
            "other": results_dict["other"]
            },
            take_mean=take_mean
            )


        if take_mean is False:
            loss = loss.flatten()
        risk = -torch.sum(S, dim=1)

        return loss, risk, loss_dict

    ## Train step:
    def training_step(self, batch, batch_idx):
        (case_ids, slide_ids, 
        data_WSI, patch_mask,
        data_clin_list, omics_pways_list,
        label, event_time, c) = batch
        
        
        if self.is_hvt_model:
            ## Batch size is 1:
            if len(case_ids) > 1:
                raise ValueError('Running HVT Models w batch size > 1 is not supported.')
            case_id = case_ids[0]
            slide_ids = [slide_id[0] for slide_id in slide_ids]
        

        results_dict = self.model(
            wsi_data=data_WSI, 
            clin_data_list=data_clin_list,
            omics_pways_list=omics_pways_list,
            patch_mask=patch_mask,
            slide_ids = slide_ids,
            return_attn=False,
            )

        results_dict_save = results_dict.copy()
        results_dict_save['hazards'] = results_dict_save['hazards'].clone().detach().cpu().numpy()
        results_dict_save['S'] = results_dict_save['S'].clone().detach().cpu().numpy()
        results_dict_save['Y_hat'] = results_dict_save['Y_hat'].clone().detach().cpu().numpy()

        if self.is_hvt_model:
            self.train_attention[case_id] = results_dict_save

        loss, risk, loss_dict = self.compute_loss_and_risk(results_dict, label, c)
        
        torch.cuda.empty_cache()

        train_step_loss = {'loss': loss} 
        self.train_losses.append({
            'loss': loss.detach().cpu().numpy().reshape(-1),
            'surv_loss': loss_dict['surv_loss'].detach().cpu().numpy().reshape(-1),
            'ae_loss': loss_dict['ae_loss'].detach().cpu().numpy().reshape(-1),
            })

        return train_step_loss
    

    def validation_step(self, batch, batch_idx):
        
        (case_ids, slide_ids, 
        data_WSI, patch_mask,
        data_clin_list, omics_pways_list,
        label, event_time, c) = batch
        if self.is_hvt_model:
            case_id = case_ids[0]
        slide_ids = [slide_id[0] for slide_id in slide_ids]

        results_dict = self.model(
            wsi_data=data_WSI, 
            clin_data_list=data_clin_list,
            omics_pways_list=omics_pways_list,
            patch_mask=patch_mask,
            slide_ids = slide_ids,
            return_attn=False,
            )

        results_dict_save = results_dict.copy()
        results_dict_save['hazards'] = results_dict_save['hazards'].clone().detach().cpu().numpy()
        results_dict_save['S'] = results_dict_save['S'].clone().detach().cpu().numpy()
        results_dict_save['Y_hat'] = results_dict_save['Y_hat'].clone().detach().cpu().numpy()

        if self.is_hvt_model:
            self.valid_attention[case_id] = results_dict_save

        loss, risk, loss_dict = self.compute_loss_and_risk(
            results_dict, label, c, take_mean = False)
        
        val_output = {
            'loss' : loss.detach().cpu().numpy().reshape(-1), 
            'surv_loss': loss_dict['surv_loss'].detach().cpu().numpy().reshape(-1),
            'ae_loss':  loss_dict['ae_loss'].detach().cpu().numpy().reshape(-1),
            'risk': risk.detach().cpu().numpy().reshape(-1), 
            'censorship' : c.cpu().numpy(),
            'event_time' : event_time.cpu().numpy()}
        self.val_step_outputs.append(val_output)

        return val_output


    def on_validation_epoch_end(self, ):
        ## Waiting for all processes to complete validation steps.
        self.trainer.strategy.barrier()
        
        all_val_outs = self.val_step_outputs
        if self.train_losses:
            all_train_loss = np.concatenate([x['loss'] for x in self.train_losses])
            all_train_surv_loss = np.concatenate([x['surv_loss'] for x in self.train_losses])
            all_train_ae_loss = np.concatenate([x['ae_loss'] for x in self.train_losses])
        else:
            all_train_loss, all_train_surv_loss, all_train_ae_loss = np.array([]), np.array([]), np.array([])
        all_val_loss = np.concatenate([x['loss'] for x in all_val_outs],)
        all_val_surv_loss = np.concatenate([x['surv_loss'] for x in all_val_outs],)
        all_val_ae_loss = np.concatenate([x['ae_loss'] for x in all_val_outs],)
        all_risk_scores = np.concatenate([x['risk'] for x in all_val_outs],)
        all_censorships = np.concatenate([x['censorship'] for x in all_val_outs],)
        all_event_times = np.concatenate([x['event_time'] for x in all_val_outs],)

        self.val_step_outputs.clear()
        self.train_losses.clear()
        
        c_index = np.nan
        pvalue_pred = np.nan
        try:
            c_index = concordance_index_censored(
                (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
            pvalue_pred = cox_log_rank(
                all_risk_scores, (1-all_censorships), all_event_times
            )
        except:
            pass

        
        self.log('train_loss', np.mean(all_train_loss), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_surv_loss', np.mean(all_train_surv_loss), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_ae_loss', np.mean(all_train_ae_loss), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss', np.mean(all_val_loss), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_surv_loss', np.mean(all_val_surv_loss), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_ae_loss', np.mean(all_val_ae_loss), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('c_index', c_index, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('p_value', pvalue_pred, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        
        ## Waiting for all processes to complete validation summary.
        self.trainer.strategy.barrier()
        torch.cuda.empty_cache()
    
    ### Finding out which parameters are unused in the model.
    def on_train_end(self):
        # Log unused parameters
        all_params = set([
            re.sub('^model.', '', item[0]) for item in sorted(self.named_parameters())
            ])


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)

        return [optimizer]


    def test_step(self, batch, batch_idx):

        (case_ids, slide_ids, 
        data_WSI, patch_mask,
        data_clin_list, omics_pways_list,
        label, event_time, c) = batch

        if self.is_hvt_model:
            case_id = case_ids[0]

        slide_ids = [slide_id[0] for slide_id in slide_ids]

        results_dict = self.model(
            wsi_data=data_WSI, 
            clin_data_list=data_clin_list,
            slide_ids = slide_ids,
            omics_pways_list=omics_pways_list,
            patch_mask=patch_mask,
            return_attn=True
            )

        results_dict_save = results_dict.copy()
        results_dict_save['hazards'] = results_dict_save['hazards'].clone().detach().cpu().numpy()
        results_dict_save['S'] = results_dict_save['S'].clone().detach().cpu().numpy()
        results_dict_save['Y_hat'] = results_dict_save['Y_hat'].clone().detach().cpu().numpy()
        
        if self.is_hvt_model:
            self.test_attention[case_id] = results_dict_save
        

        loss, risk, loss_dict = self.compute_loss_and_risk(results_dict, label, c)

        if sum(np.isnan(risk.detach().cpu().numpy())) > 0:
            print("check")
        
        test_output = {
            'case_id': np.array([case for case in case_ids]),
            'risk' : risk.detach().cpu().numpy(), 
            'censorship' : c.cpu().numpy(), 
            'event_time' : event_time.cpu().numpy(), 
            'label': label.cpu().numpy(),
            'S' : results_dict['S'].detach().cpu().numpy()
            }
        
        self.test_step_outputs.append(test_output)

        return test_output


    def on_test_epoch_end(self,):
        ## Waiting for all processes to complete test steps.
        self.trainer.strategy.barrier()
        all_test_outs = self.test_step_outputs
        
        all_risk_scores = np.concatenate([x['risk'] for x in all_test_outs],)
        all_censorships = np.concatenate([x['censorship'] for x in all_test_outs],)
        all_event_times = np.concatenate([x['event_time'] for x in all_test_outs],)
        all_labels = np.concatenate([x['label'] for x in all_test_outs],)
        all_survs = np.concatenate([x['S'] for x in all_test_outs], axis=0).squeeze()
        all_case_ids = np.concatenate([x['case_id'] for x in all_test_outs],)
       
        
        self.test_step_outputs.clear()

        if self.is_hvt_model:
            os.makedirs(os.path.join(self.log_path, 'attention'), exist_ok=True)
            attention_test_path = os.path.join(self.log_path, 'attention', f'test.npz')
            np.savez(attention_test_path, **{'main_dict':self.test_attention}, allow_pickle=True)
            self.test_attention = defaultdict()
            
        
        #---->Calculation of metrics
        c_index = concordance_index_censored(
            (1-all_censorships).astype(bool), all_event_times, 
            all_risk_scores, tied_tol=1e-08)[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1-all_censorships), all_event_times)
        print(f'c_index={c_index}, p_value={pvalue_pred}')
        
        ## Waiting for all processes to complete test summary.
        self.trainer.strategy.barrier()

        ##--->Get Patient-Level Test Set Embeddings

        all_test_event_times, all_test_censorships, all_test_clin_feats, all_test_cases, all_test_times = (
                collect_all_data(self.surv_dm, stage='test'))

        if self.model.__class__.__name__ not in ['clinOnlySurv', ]:
            ### Yield trained DPAI features and stash in log dir.            
            y_dtypes = [('event', bool), ('time', float)]
            y_test = np.array(list(zip(
                (1-all_test_censorships).astype(bool), all_test_event_times)), dtype=y_dtypes)

            all_train_vox_feats, all_train_risk_scores, all_train_case_ids = embed_all_data(
                self, self.surv_dm, stage='train')
            all_valid_vox_feats, all_valid_risk_scores, all_valid_case_ids = embed_all_data(
                self, self.surv_dm, stage='valid')
            all_test_vox_feats, all_test_risk_scores, all_test_case_ids = embed_all_data(
                self, self.surv_dm, stage='test')

            ### DPAI Features Convert DPAI features to DataFrames:
            if self.model.__class__.__name__ not in ['vecInputSurv', ]:
                train_vox_feats_df = make_vox_feats_df(all_train_case_ids, all_train_vox_feats)
                valid_vox_feats_df = make_vox_feats_df(all_valid_case_ids, all_valid_vox_feats)
                test_vox_feats_df = make_vox_feats_df(all_test_case_ids, all_test_vox_feats)

                ## Save various DataFrames:
                train_vox_feats_df.to_csv(os.path.join(self.log_path, 'train_dpai_features.csv'), index=False)
                valid_vox_feats_df.to_csv(os.path.join(self.log_path, 'valid_dpai_features.csv'), index=False)
                test_vox_feats_df.to_csv(os.path.join(self.log_path, 'test_dpai_features.csv'), index=False)

            ### Convert Risk Scores to DataFrame:
            train_risk_scores_df = pd.concat((
                pd.DataFrame(all_train_case_ids, columns=['case_id']),
                pd.DataFrame(all_train_risk_scores, columns=['risk_score']),
            ), axis=1)
            valid_risk_scores_df = pd.concat((
                pd.DataFrame(all_valid_case_ids, columns=['case_id']),
                pd.DataFrame(all_valid_risk_scores, columns=['risk_score']),
            ), axis=1)
            test_risk_scores_df = pd.concat((
                pd.DataFrame(all_case_ids, columns=['case_id']),
                pd.DataFrame(all_risk_scores, columns=['risk_score']),
            ), axis=1)

            ## Save various DataFrames to log dir
            train_risk_scores_df.to_csv(os.path.join(self.log_path, 'train_dpai_riskscores.csv'), index=False)
            valid_risk_scores_df.to_csv(os.path.join(self.log_path, 'valid_dpai_riskscores.csv'), index=False)
            test_risk_scores_df.to_csv(os.path.join(self.log_path, 'test_dpai_riskscores.csv'), index=False)
        

        ## save C-Index Risk Scores:
        
        print("Computing Bootstrap C-Indices")
        #---->bootstrap
        n = 1000
        skipped = 0
        boot_c_index = []
        boot_xgb_c_index = []
        boot_cox_c_index = []
        err = []
        for i in tqdm(range(n)):
            boot_ids = resample(np.arange(len(all_risk_scores)), replace=True)
            risk_scores = all_risk_scores[boot_ids]
            censorships = all_censorships[boot_ids]
            event_times = all_event_times[boot_ids]
            survs = all_survs[boot_ids]
            # When running samples with small number of patients (e.g. some
            # individual cancer types) sometimes there are no admissible pairs
            # to compute the C-index (or other metrics).
            # In those cases continue and print a warning at the end
            try:
                c_index_buff = concordance_index_censored(
                    (1-censorships).astype(bool), event_times, risk_scores, tied_tol=1e-08
                    )[0]
                #---->save
                boot_c_index.append(c_index_buff)

            except ValueError as error:                                  
                err.append(error)                                                     
                skipped += 1                                                    
                continue  
        if skipped > 0:
            print(f'Skipped {skipped} bootstraps ({err}).')

        #---->Calculate the gap between the bootstraps and the actual metric
        c_index_differences = sorted([x - c_index for x in boot_c_index])
        c_index_percent = np.nanpercentile(c_index_differences, [2.5, 97.5])
        c_index_low, c_index_high = tuple(round(c_index + x, 4)
                                for x in [c_index_percent[0], c_index_percent[1]])

        #---->Save all metrics as csv
        test_dict = {
            'c_index':c_index, 
            'c_index_high':c_index_high, 
            'c_index_low':c_index_low,
            'p_value':pvalue_pred, 
                }
        result = pd.DataFrame(list(test_dict.items()))
        result.to_csv(self.log_path / 'result.csv')

        # #---->Save the three indicators of all_risk_scores, all_censorships, and all_event_times, and ask for all folds
        np.savez(self.log_path / 'all_risk_scores.npz', all_risk_scores)
        np.savez(self.log_path / 'all_censorships.npz', all_censorships)
        np.savez(self.log_path / 'all_event_times.npz', all_event_times)
        np.savez(self.log_path / 'all_labels.npz', all_labels)
        np.savez(self.log_path / 'all_survs.npz', all_survs)


    def load_model(self):
        name = self.hparams.modelinfo.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
            Model = getattr(importlib.import_module(
                f'models_{name}'), camel_name)
        
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.modelinfo.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.modelinfo, arg)
        args1.update(other_args)
        return Model(**args1)
