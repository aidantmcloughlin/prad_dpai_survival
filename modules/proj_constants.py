import os, sys
import yaml
import glob
from git import Repo

### Important constants for deciding what models will be run. =================

## Git TopLevel Path:
current_file_path = os.path.abspath(__file__)
repo = Repo(current_file_path, search_parent_directories = True)
GIT_ROOT = repo.git.rev_parse("--show-toplevel")

## WHERE TO RETRIEVE ALL MAIN DATA (subfolder):
ALL_DATA_PAR_DIR = os.path.join(
    GIT_ROOT,
    "data",)
    
COHORT_NAME = 'tcga_prad'

UNI_CKPT_DIR = os.path.join(ALL_DATA_PAR_DIR, "pretrained_models/uni/checkpointzip")

RUN_PREPROC = True ## run data preprocessing scripts, note these scripts have autoskip configured.
TEST_ONLY = False ## run training loops, or just recompute test results 

### Partition such that batches are close in UMAP space:
BATCHES_IDX_EVEN = [[1], [2, 3], [4, 5, 6], [7], [8, 9]]

BATCH_SPLITTING_SEED = 99999

BEST_ALPHA = 0.4 ## weighting hyperparameter for censored vs uncensored patients
VALID_PROP = 0.2 ## how much of train data to reserve for validation.
N_BINS = 4 ## number of bins to discretize the survival months into.

## Determining the feature set:
PATCH_SIZE = 1024
MAGNIF = 40
ENCODER = 'UNI' #options: "UNI", "imagenet"

### epochs/patience, lrs per model
FCMODEL_EPOCHS = 100
FCMODEL_PATIENCE = FCMODEL_EPOCHS
HVT_EPOCHS = 20
HVT_PATIENCE = HVT_EPOCHS
GCN_EPOCHS = 20
GCN_PATIENCE = GCN_EPOCHS
CLAM_EPOCHS = 20
CLAM_PATIENCE = CLAM_EPOCHS
CMTA_EPOCHS = 20
CMTA_PATIENCE = CMTA_EPOCHS

### Default learning rates for models
FCMODEL_LR = 1e-3
HVT_LR = 2e-4
CLAM_LR = 1e-4
GCN_LR = 5e-4 
CMTA_LR = 5e-5

## Clinical Variable Set:
ALL_CLIN_VARS = [
    'has_PSM', 'has_SVI', 'has_ECE', 'has_LNI',
    'gleason_grade', 'diagnosis_age',
]

## Genomic Set:
GENOMIC_SET_5 = [
    "PCBP1",
    "PABPN1", 
    "PTPRF", 
    "DANCR", 
    "MYC"
]

### Directories, Relative Directories

DATA_PAR_DIR = os.path.join(
    ALL_DATA_PAR_DIR,
    COHORT_NAME)

## Relative to the Git repository head
PROJ_REL_LOC = ''

## Important Subdirs, should in some way be dynamic to above root dirs.

SLIDES_DIR = os.path.join(
    DATA_PAR_DIR, 'wsi_files'
)

METADATA_PAR_DIR = os.path.join(
    DATA_PAR_DIR, 'tabular_data/metadata'
)

EMBED_ROOT_DIR = os.path.join(
    DATA_PAR_DIR,
    'histo_model_preproc',
)

LABEL_ROOT_DIR = os.path.join(
    DATA_PAR_DIR,
    "tabular_data/cross_val_tables"
)

## Exporting key path constants to a sourceable shell script
# Write a script to source these variables
dir_dict = {
    'GIT_ROOT': GIT_ROOT,
    'PROJ_REL_LOC': PROJ_REL_LOC,
    'DATA_PAR_DIR': DATA_PAR_DIR,
    'ALL_DATA_PAR_DIR': ALL_DATA_PAR_DIR,
    'SLIDES_DIR': SLIDES_DIR,
    'METADATA_PAR_DIR': METADATA_PAR_DIR,
    'EMBED_ROOT_DIR': EMBED_ROOT_DIR,
    'LABEL_ROOT_DIR': LABEL_ROOT_DIR,
    'UNI_CKPT_DIR': UNI_CKPT_DIR,
}
## Store the directories in accessible location for preprocessing bash scripts.
with open(os.path.join(
    GIT_ROOT,
    PROJ_REL_LOC,
    'modules/preprocessing/shell_deployers',
    'current_proj_dirs.yaml'), 'w'
    ) as f:
    yaml.dump(dir_dict, f)
