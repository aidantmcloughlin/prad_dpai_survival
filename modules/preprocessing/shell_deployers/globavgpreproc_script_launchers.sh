#!/bin/bash
set -euo pipefail
## This script includes commands for running globavg model preproc steps.

# Get the directory of the current script
script_dir=$(dirname "$(readlink -f "$0")")

# Read the arguments
PATCH_SIZE=$1
EXTRACTOR=$2
MAGNIF=$3

## For CV Splitting.
BATCH_NAME=$4
BATCH_SEED=$5
VAL_TR_FRAC=${6:-0.2}
N_BINS=${7:-4} ## number discretized bins 

CLIN_FILE=$8
OMICS_FILE=$9

CONDA_ENV=${10:-""}

## Collect directories 
dir_yaml_file=$script_dir/current_proj_dirs.yaml
# Load the constants from the text file located one directory up
GIT_ROOT=$(yq eval '.GIT_ROOT' $dir_yaml_file)
DATA_PAR_DIR=$(yq eval '.DATA_PAR_DIR' $dir_yaml_file)
PROJ_REL_LOC=$(yq eval '.PROJ_REL_LOC' $dir_yaml_file)

PROJ_ROOT="$GIT_ROOT/$PROJ_REL_LOC"

## activate conda environment, if needed:
if [[ -n "$CONDA_ENV" ]]; then

    if command -v conda >/dev/null 2>&1; then
        # Works on any install (conda/anaconda/miniconda)
        eval "$(conda shell.bash hook)"
    fi
    echo "Activating conda environment: $CONDA_ENV"

    conda activate "$CONDA_ENV"
else
    echo "No conda environment specified"
fi

## Patch Extraction, Fold Splits:
bash "$script_dir/patch_extract_cv_splits.sh" \
    $PATCH_SIZE $EXTRACTOR $MAGNIF $BATCH_NAME $BATCH_SEED \
    $VAL_TR_FRAC $N_BINS $CLIN_FILE $OMICS_FILE $CONDA_ENV

## Global Avg of Patch Embeddings:
echo "RUNNING GLOBAL AVERAGING OF PATCHES"
glob_avg_script="$PROJ_ROOT/modules/preprocessing/embeddings/wsi_level/global_avgs.py"
python3 $glob_avg_script --encoder $EXTRACTOR --patch_size $PATCH_SIZE \
     --magnif $MAGNIF --auto_skip True