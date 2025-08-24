import os, sys
import re
import glob
import shutil, subprocess
import argparse
from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd

## local module/s:
from utils_survival import (
    update_yaml, read_yaml, 
    check_magnification,
    get_wsifeature_and_labels_paths,
    intersect
    )
from deploy_from_config import (
    crossval_from_yaml_config,
    looped_deployment
    )

## Git TopLevel Path:
from git import Repo
current_file_path = os.path.abspath(__file__)
repo = Repo(current_file_path, search_parent_directories = True)
GIT_ROOT = repo.git.rev_parse("--show-toplevel")

modules_dir = os.path.join(GIT_ROOT, 'modules')

sys.path.append(modules_dir)
from proj_constants import (
    DATA_PAR_DIR,
    PROJ_REL_LOC,
    SLIDES_DIR,
    BATCH_SPLITTING_SEED,
    EMBED_ROOT_DIR, 
    LABEL_ROOT_DIR, 
    HVT_EPOCHS, HVT_PATIENCE, HVT_LR,
    CLAM_EPOCHS, CLAM_PATIENCE, CLAM_LR,
    GCN_EPOCHS, GCN_PATIENCE, GCN_LR,
    CMTA_EPOCHS, CMTA_PATIENCE, CMTA_LR,
    FCMODEL_EPOCHS, FCMODEL_PATIENCE, FCMODEL_LR,
    ALL_CLIN_VARS,
    RUN_PREPROC,
    TEST_ONLY,
    GENOMIC_SET_5,
    BEST_ALPHA,
    PATCH_SIZE,
    ENCODER,
    MAGNIF,
    VALID_PROP,
    N_BINS,
)


def update_yaml_preproc_deploy(
    data_parent_dir,
    epochs,
    patience,
    lr,
    clin_vars,
    encoder,
    patch_size,
    data_seed,
    clin_tag = 'clin',
    model = 'hvtsurv',
    batch_name = 'batchstrat',
    magnif = 40,
    alpha = BEST_ALPHA,
    n_bins = 4,
    val_tr_frac = 0.2,
    test_only = TEST_ONLY,
    **kwargs
    ):

    if len(clin_vars) == 0:
        clin_tag = 'noclin'

    if batch_name not in ['batchstrat', 'batchiso']:
        raise ValueError("Not Implemented")
    
    active_yaml_path = os.path.join(
        GIT_ROOT, PROJ_REL_LOC, 'configs', 'run_' + model + '.yml'
    )

    ## save available svs names to preproc from path:
    avail_svs_files = [os.path.basename(f) for f in glob.glob(
        f"{os.path.join(DATA_PAR_DIR, 'wsi_files')}/*"
        )]
    
    avail_wsi_file_names = [re.sub('.svs', '', s) for s in avail_svs_files]
    pd.DataFrame({'slide_id': avail_wsi_file_names}).to_csv(
        os.path.join(DATA_PAR_DIR, 'tabular_data/metadata/wsi_svs_fnames.csv'),
        index=False
    )
    

    ## check the magnification:
    patch_size, magnif = check_magnification(
        SLIDES_DIR,
        patch_size,
        magnif
    )

    wsi_feature_dir, label_dir = get_wsifeature_and_labels_paths(
        feat_root_dir = EMBED_ROOT_DIR, 
        label_root_dir = LABEL_ROOT_DIR,
        encoder = encoder, 
        model = model, 
        patch_size = patch_size, 
        data_seed = data_seed,
        batch_name = batch_name,
        magnif = magnif,
        )

    if wsi_feature_dir != '':
        update_yaml(
            active_yaml_path,
            epochs = epochs,
            patience = patience,
            lr = lr,
            n_bins = n_bins,
            data_parent_dir = data_parent_dir,
            wsifeat_dir = os.path.relpath(wsi_feature_dir, data_parent_dir),
            label_dir = os.path.relpath(label_dir, data_parent_dir),
            )
    else:
        update_yaml(
            active_yaml_path,
            epochs = epochs,
            patience = patience,
            lr = lr,
            n_bins = n_bins,
            data_parent_dir = data_parent_dir,
            label_dir = os.path.relpath(label_dir, data_parent_dir),
            )

    cfg_rm = read_yaml(active_yaml_path)
    
    ## update model tag to be saved as the logdir =============================

    if 'CMTA' in model:
        ## Include fusion type, l1 sim pen info:
        fusion = kwargs.pop('fusion', 'concat')
        update_yaml(active_yaml_path, fusion = fusion)
        model_tag = model + fusion + '_' + 'L1SimPen' + str(cfg_rm.Loss.l1sim_alpha)
        ## Include pathways, RNAseq info:
        pways_tag_dict={
        'hallmark_pathways_wprad.csv': '_PwayHmarkprad',}
        rnaseq_tag_dict={
            'rnaseq_in_pways_tpm_norm.csv': '_RnaTpmNorm',
            'rnaseq_in_pways_tpm.csv': '_RnaTpm'}
        pways_file = os.path.basename(cfg_rm.Data.pways_file)
        rnaseq_file = os.path.basename(cfg_rm.Data.omics_file)
        pways_tag = pways_tag_dict.get(pways_file, '_NOPWAYMATCH')
        rnaseq_tag = rnaseq_tag_dict.get(rnaseq_file, '_NORNAMATCH')
        model_tag = model_tag + pways_tag + rnaseq_tag
    else:
        model_tag = model

    ## Complete data preprocessing, if needed
    if RUN_PREPROC:

        ## collect the active conda environment to be deployed into subprocess
        conda_env = re.sub('^.*envs/(.*)/bin/python.*', '\\1', sys.executable)
        conda_prefix = os.getenv('CONDA_PREFIX')

        ## determine the preprocessing bash script depending on the model in-use.
        if model == 'hvtsurv':
            bash_script_path = os.path.join(
                modules_dir, 
                'preprocessing/shell_deployers/hvtpreproc_script_launchers.sh')
        elif model == 'CLAM':
            bash_script_path = os.path.join(
                modules_dir, 
                'preprocessing/shell_deployers/CLAMpreproc_script_launchers.sh')
        elif model == 'globavg':
            bash_script_path = os.path.join(
                modules_dir, 
                'preprocessing/shell_deployers/globavgpreproc_script_launchers.sh')
        elif model == 'patchGCN':
            bash_script_path = os.path.join(
                modules_dir, 
                'preprocessing/shell_deployers/patchGCNpreproc_script_launchers.sh')
        elif 'CMTA' in model:
            bash_script_path = os.path.join( 
                modules_dir, 
                'preprocessing/shell_deployers/CLAMpreproc_script_launchers.sh')
        else:
            bash_script_path = None

        if bash_script_path is not None:
            ## Construct the command to activate the conda environment and run the script

            command = f"""{bash_script_path} {str(patch_size)} {encoder} {str(magnif)} {batch_name} {str(data_seed)} {f"{val_tr_frac:.3f}"} {str(n_bins)} {str(cfg_rm.Data.clin_file)} {str(cfg_rm.Data.omics_file)} {conda_env}"""

            print(command)

            subprocess.run(command, shell=True, executable="/bin/bash")

    looped_deployment(
        log_path_tag = os.path.join(
            'logs', model_tag + '_' + 
            batch_name + '_dseed' + str(data_seed) +
            '_patchsize' + str(patch_size) + 
            '_encoder' + str(encoder)
            ),
        yaml_path = active_yaml_path,
        clin_vars = clin_vars,
        lr = lr,
        alpha_vec = [alpha],
        test_no_clin = False, ## Feed empty clin var vector if dont want to test no clin.
        clin_tag = clin_tag,
        fold_idx_start=0,
        fold_idx_end=None,
        test_only=test_only
        )

### Model Specific One-Liner Deployers.
def update_yaml_and_deploy_CMTA(clin_vars, batch_name, fusion='concat'):
    update_yaml_preproc_deploy(
        data_parent_dir = DATA_PAR_DIR,
        epochs = CMTA_EPOCHS,
        patience = CMTA_PATIENCE,
        lr = CMTA_LR,
        clin_vars = clin_vars,
        encoder = ENCODER,
        patch_size = PATCH_SIZE,
        data_seed = BATCH_SPLITTING_SEED,
        clin_tag = CLIN_TAG,
        model = 'CMTA',
        batch_name = batch_name,
        magnif = MAGNIF,
        n_bins = N_BINS,
        val_tr_frac = VALID_PROP,
        test_only = TEST_ONLY,
        fusion = fusion,
    )

def update_yaml_and_deploy_CMTAGleason(clin_vars, batch_name, fusion='concat'):
    update_yaml_preproc_deploy(
        data_parent_dir = DATA_PAR_DIR,
        epochs = CMTA_EPOCHS,
        patience = CMTA_PATIENCE,
        lr = CMTA_LR,
        clin_vars = clin_vars,
        encoder = ENCODER,
        patch_size = PATCH_SIZE,
        data_seed = BATCH_SPLITTING_SEED,
        clin_tag = CLIN_TAG,
        model = 'CMTAGleason',
        batch_name = batch_name,
        magnif = MAGNIF,
        n_bins = N_BINS,
        val_tr_frac = VALID_PROP,
        test_only = TEST_ONLY,
        fusion = fusion,
    )

def update_yaml_and_deploy_CLAM(clin_vars, batch_name):
    update_yaml_preproc_deploy(
        data_parent_dir = DATA_PAR_DIR,
        epochs = CLAM_EPOCHS,
        patience = CLAM_PATIENCE,
        lr = CLAM_LR,
        clin_vars = clin_vars,
        encoder = ENCODER,
        patch_size = PATCH_SIZE,
        data_seed = BATCH_SPLITTING_SEED,
        clin_tag = CLIN_TAG,
        model = 'CLAM',
        batch_name = batch_name,
        magnif = MAGNIF,
        n_bins = N_BINS,
        val_tr_frac = VALID_PROP,
        test_only = TEST_ONLY,
    )

def update_yaml_and_deploy_patchGCN(clin_vars, batch_name):
    update_yaml_preproc_deploy(
        data_parent_dir = DATA_PAR_DIR,
        epochs = GCN_EPOCHS,
        patience = GCN_PATIENCE,
        lr = GCN_LR,
        clin_vars = clin_vars,
        encoder = ENCODER,
        patch_size = PATCH_SIZE,
        data_seed = BATCH_SPLITTING_SEED,
        clin_tag = CLIN_TAG,
        model = 'patchGCN',
        batch_name = batch_name,
        magnif = MAGNIF,
        n_bins = N_BINS,
        val_tr_frac = VALID_PROP,
        test_only = TEST_ONLY,
    )


def update_yaml_and_deploy_hvt(clin_vars, batch_name):
    update_yaml_preproc_deploy(
        data_parent_dir = DATA_PAR_DIR,
        epochs = HVT_EPOCHS,
        patience = HVT_PATIENCE,
        lr = HVT_LR,
        clin_vars = clin_vars,
        encoder = ENCODER,
        patch_size = PATCH_SIZE,
        data_seed = BATCH_SPLITTING_SEED,
        clin_tag = CLIN_TAG,
        model = 'hvtsurv',
        batch_name = batch_name,
        magnif = MAGNIF,
        n_bins = N_BINS,
        val_tr_frac = VALID_PROP,
        test_only = TEST_ONLY,
    )

def update_yaml_and_deploy_globavg(clin_vars, batch_name):
    update_yaml_preproc_deploy(
        data_parent_dir = DATA_PAR_DIR,
        epochs = FCMODEL_EPOCHS,
        patience = FCMODEL_PATIENCE,
        lr = FCMODEL_LR,
        clin_vars = clin_vars,
        encoder = ENCODER,
        patch_size = PATCH_SIZE,
        data_seed = BATCH_SPLITTING_SEED,
        clin_tag = CLIN_TAG,
        model = 'globavg',
        batch_name = batch_name,
        magnif = MAGNIF,
        n_bins = N_BINS,
        val_tr_frac = VALID_PROP, 
        test_only = TEST_ONLY,
    )


if __name__ == '__main__':
    ## setting up the clinical variable set for any late fusion models:
    CLIN_TAG = 'allclinical'
    clin_var_list = ALL_CLIN_VARS

    ## Deploy unimodal DPAI MIL models on stratified batch folds:

    update_yaml_and_deploy_globavg(
        clin_vars=[], 
        batch_name='batchstrat')

    update_yaml_and_deploy_CLAM(
        clin_vars=[], 
        batch_name='batchstrat')

    update_yaml_and_deploy_patchGCN(
        clin_vars=[], 
        batch_name='batchstrat')#

    update_yaml_and_deploy_hvt(
        clin_vars=[], 
        batch_name='batchstrat')

    ## Deploy model with early fusion genomics pathways:
    update_yaml_and_deploy_CMTA(
        clin_vars=[], 
        batch_name='batchstrat',
        )

    ## Deploy model with early fusion genomics pathways + Gleason:
    update_yaml_and_deploy_CMTAGleason(
        clin_vars=[], 
        batch_name='batchstrat',
        )

    ## Deploy model with late fusion tabular variables:    
    update_yaml_and_deploy_CLAM(
            clin_vars=clin_var_list, 
            batch_name='batchstrat')


    ## Deploy model with isolated batch folds:
    update_yaml_and_deploy_CLAM(
        clin_vars=[], 
        batch_name='batchiso')
