import os
import shutil
import copy
import argparse
from pathlib import Path
import numpy as np
import glob
import re
###
from datasets_data_interface import DataInterface
from models_model_interface import ModelInterface
from utils_survival import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch_geometric


#---->main function
def deploy_from_config(cfg):

    input_cfg = copy.deepcopy(cfg)

    #---->Initialize the seed
    pl.seed_everything(cfg.General.seed)

    #----> Append parent dir:
    for d in list(np.intersect1d(
        [ 
            'wsifeat_dir', 
            'clin_file', 'label_dir', 'pways_file', 'omics_file','log_path'],
        list(cfg.Data.keys()))
        ):
        cfg.Data[d] = cfg.Data.parent_dir + cfg.Data[d]



    #---->Load loggers
    ## collect relevant hparams from the model interface
    cfg.load_loggers, completed_run = load_loggers(cfg)

    if cfg.General.server=='test' or (not completed_run):
        #---->Load callbacks
        cfg.callbacks = load_callbacks(cfg)

        ## Pass clin_vars list to Data Config as well
        cfg.Data['clin_vars'] = cfg.ModelInfo.params['clin_vars']

        #---->Define Data class
        DataInterface_dict = {
            'train_batch_size': cfg.Data.train_dataloader.batch_size,
            'train_num_workers': cfg.Data.train_dataloader.num_workers,
            'test_batch_size': cfg.Data.test_dataloader.batch_size,
            'test_num_workers': cfg.Data.test_dataloader.num_workers,
            'dataset_name': cfg.Data.dataset_name,
            'dataset_cfg': cfg.Data,}

        dm = DataInterface(**DataInterface_dict)

        ## collect input dimensionalities from the data interface
        inputs_shapes = {}
        
        ## Image features:
        if 'wsifeat_dir' in dm.kwargs['dataset_cfg'].keys():
            ## check if computing PCA:
            if bool(dm.kwargs['dataset_cfg']['pc_dim']):
                inputs_shapes['image'] = torch.Size(
                    [dm.kwargs['dataset_cfg']['pc_dim']])
            ## else: dimension of the patch/image feature vector.
            else:
                feat_dir = dm.kwargs['dataset_cfg']['wsifeat_dir']
                feat_ex_fname = os.path.join(
                    feat_dir,
                    [os.path.basename(f) for f in glob.glob(f"{feat_dir}/*")][0])
                if isinstance(torch.load(feat_ex_fname), torch_geometric.data.Data):
                    inputs_shapes['image'] = torch.load(feat_ex_fname).num_node_features
                else:
                    inputs_shapes['image'] = torch.load(feat_ex_fname).shape
        ## Clinical features:
        clin_df = pd.read_csv(dm.kwargs['dataset_cfg']['clin_file']).loc[
            :, cfg.ModelInfo.params['clin_vars']]
        dim_clin = 0
        for col in list(clin_df.columns):
            if clin_df[col].dtype in ('object', 'string'):
                dim_clin += len(np.unique(clin_df[col].values)) - 1
            else:
                dim_clin += 1
        inputs_shapes['clin'] = dim_clin
        ## Other Features: Pathways
        if 'pways_file' in dm.kwargs['dataset_cfg'].keys():
            pways_df = pd.read_csv(os.path.join(
                dm.kwargs['dataset_cfg']['pways_file']))
            inputs_shapes['num_pathways'] = pways_df.shape[1]
            cfg.ModelInfo['params']['pways_file'] = dm.kwargs['dataset_cfg']['pways_file']
        ## Other Features: Omics
        if 'omics_file' in dm.kwargs['dataset_cfg'].keys():
            ## extract genes as list:
            cfg.ModelInfo['params']['omics_cols'] = pd.read_csv(
                dm.kwargs['dataset_cfg']['omics_file'], index_col=0).index.to_numpy()
            


        cfg.ModelInfo['params']['inputs_shapes'] = inputs_shapes
        

        #---->Define Model class
        ModelInterface_dict = {'modelinfo': cfg.ModelInfo,
                                'loss_config': copy.deepcopy(cfg.Loss),
                                'optimizer': cfg.Optimizer,
                                'data': cfg.Data,
                                'log': cfg.log_path,
                                }
        model = ModelInterface(**ModelInterface_dict)

        model.surv_dm = dm

        all_clin_cols = cfg.ModelInfo.params['clin_vars']
        ## order as they are in model training:
        cat_cols = list(clin_df.columns[[clin_df[col].dtype in ('object', 'string') for col in (
            list(clin_df.columns))
            ]])
        binary_cols = list(clin_df.columns[
            clin_df.apply(lambda col: col.dropna().isin([0, 1]).all())
            ])
        float_cols = [
            item for item in clin_df.columns if 
            item not in cat_cols+binary_cols]
        clin_var_names_train = float_cols + cat_cols + binary_cols
        model.clin_var_names = clin_var_names_train

        
        #---->Instantiate Trainer
        if cfg.General.server=='train':
            devices="auto"
            strategy="ddp_find_unused_parameters_false" #"ddp"
        else:
            ## test:
            devices=1
            strategy="auto"

        if 'precision' not in cfg.General.keys():
            cfg.General.precision = None
        
        if 'train_steps_per_epoch' not in cfg.General.keys():
            cfg.General.train_steps_per_epoch = 10**10 ## larger value than ever needed.

        ## Load checkpoint file, if existing.
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if (
            'epoch' in str(model_path))
            ]

        model_paths = [s for s in model_paths if not re.search('v\d+\.ckpt$', s)]

        if len(model_paths) == 0:
            model_chkpt = None
        elif len(model_paths) == 1:
            model_chkpt = model_paths[0]
            print("Checkpoint path:" + str(model_chkpt))
        else:
            raise ValueError('multiple model checkpoints found.')


        trainer = Trainer(
            num_sanity_val_steps=0, # Train directly
            logger=cfg.load_loggers,
            log_every_n_steps=25,
            callbacks=cfg.callbacks,
            max_epochs = cfg.General.epochs,
            accelerator='cuda', 
            strategy=strategy, #'ddp'
            gradient_clip_val = 4,
            devices=devices,
            precision= cfg.General.precision,  # Half precision training
            accumulate_grad_batches=cfg.General.grad_acc,
            deterministic=False, #True
            check_val_every_n_epoch=1,
            reload_dataloaders_every_n_epochs=5,
        )

        

        #---->Train or test
        if cfg.General.server=='train':
            trainer.fit(model=model, datamodule=dm, ckpt_path=model_chkpt)
            
        else:
            new_model = ModelInterface.load_from_checkpoint(
                checkpoint_path=model_chkpt, 
                modelinfo=cfg.ModelInfo,
                cfg=cfg
                )

            ## append data module to the object.
            new_model.surv_dm = model.surv_dm
            new_model.clin_var_names = (
                float_cols + cat_cols + binary_cols
            )
            trainer.test(model=new_model, datamodule=new_model.surv_dm)

            ## Compute Integrated Gradients:
            if new_model.is_ig_model:
                test_risk_scores_df = pd.read_csv(
                    os.path.join(new_model.log_path, 'test_dpai_riskscores.csv'))
                attribution_df = getPathwayIGs(model, test_risk_scores_df)

                gene_attribution_df = getGeneIGs(model, test_risk_scores_df)



def setup_deploy_config(
    yaml_filepath,
    model_stage = "train", # "train", "test"
    fold = 0,
    ):
 
    #---->Read yaml configuration
    cfg = read_yaml(yaml_filepath)

    #---->Update: Save other params to config
    cfg.config_file = yaml_filepath
    cfg.General.server = model_stage
    cfg.Data.fold = fold

    #---->Main function
    deploy_from_config(copy.deepcopy(cfg))
 
def crossval_from_yaml_config(
    yaml_filepath, 
    fold_idx_start = 0, 
    fold_idx_end = None,
    test_only = False):
    ### Read yaml configuration
    cfg = read_yaml(yaml_filepath)

    ### Determine num folds:
    n_folds = len(os.listdir(os.path.join(cfg.Data.parent_dir, cfg.Data.label_dir)))

    if fold_idx_end is None:
        fold_idx_end = n_folds
    else:
        fold_idx_end = fold_idx_end + 1

    for f in range(fold_idx_start, fold_idx_end):
        if not test_only:
            print("TRAINING MODEL ON CV FOLD: " + str(f))
            setup_deploy_config(
                yaml_filepath, model_stage = "train", fold = f
                )
        
        setup_deploy_config(
            yaml_filepath, model_stage = "test", fold = f
            )



### Any 'Tuning Loop' deployment functions: ===================================

## Alpha, Clin vs No Clin Deployment Loop:
def looped_deployment(
    log_path_tag,
    yaml_path,
    clin_vars,
    lr,
    alpha_vec = [0, 0.25, 0.5, 0.75],
    test_no_clin = True,
    **kwargs):

    update_yaml(
        yaml_path,
        lr = lr,
        **kwargs
    )

    ## setup clin looping list
    clin_tag = kwargs.pop('clin_tag', 'clin')
    if test_no_clin:
        clin_sets = [[], clin_vars]
        clin_tags = ['no_clin', clin_tag]
    else:
        clin_sets = [clin_vars]
        clin_tags = [clin_tag]

    ## handle kwargs
    fold_idx_start = kwargs.pop('fold_idx_start', 0)
    fold_idx_end = kwargs.pop('fold_idx_end', None)
    test_only = kwargs.pop('test_only', False)

    for a in alpha_vec:
        for i in range(len(clin_sets)):
            c = clin_sets[i]
            c_tag = clin_tags[i]
            ## update config:
            update_yaml(
                yaml_path, 
                log_path = (
                    log_path_tag + 
                    "_" + c_tag + 
                    "_alpha" + str(a) + 
                    "_lr" + str(lr)
                    ),
                alpha_surv = a,
                clin_vars = c
                )

            ## CV deployer:
            crossval_from_yaml_config(
                yaml_path, 
                fold_idx_start=fold_idx_start,
                fold_idx_end=fold_idx_end,
                test_only=test_only,
                )


