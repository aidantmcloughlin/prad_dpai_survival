#!/bin/bash
set -euo pipefail
## This script includes commands for running HTVSurv preproc model steps.

# Get the directory of the current script
script_dir=$(dirname "$(readlink -f "$0")")

## Collect directories 
dir_yaml_file=$script_dir/current_proj_dirs.yaml
# Load the constants from the text file located one directory up
GIT_ROOT=$(yq eval '.GIT_ROOT' $dir_yaml_file)
DATA_PAR_DIR=$(yq eval '.DATA_PAR_DIR' $dir_yaml_file)
PROJ_REL_LOC=$(yq eval '.PROJ_REL_LOC' $dir_yaml_file)
EMBED_ROOT_DIR=$(yq eval '.EMBED_ROOT_DIR' $dir_yaml_file)

PROJ_ROOT="$GIT_ROOT/$PROJ_REL_LOC"

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

## Other Constants
# PATCH_SIZE=1024
# EXTRACTOR='imagenet'
WINDOW_SIZE=49
NUM_BAG=2
MASK_RATIO=0.5


## provide directory of the git repo root:
histo_data_root="$EMBED_ROOT_DIR"
feature_data_root="$histo_data_root/${PATCH_SIZE}_${MAGNIF}"
cross_val_root="$DATA_PAR_DIR/tabular_data/cross_val_tables/seed${BATCH_SEED}"


## Data and Output directories:
embed_out_dir="$feature_data_root/$EXTRACTOR/_2_features"
feat_rear_out_dir="$feature_data_root/$EXTRACTOR/_3_knn"
cv_split_out_dir="$cross_val_root/orig"
cv_split_rand_out_dir="$cross_val_root/random"
cv_split_batchiso_out_dir="$cross_val_root/batchiso"
cv_split_batchstrat_out_dir="$cross_val_root/batchstrat"
rand_mask_out_dir="$feature_data_root/$EXTRACTOR/_4_mask_window/seed${BATCH_SEED}"


## activate conda environment, if needed:
if [[ -n "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # Works on any install (conda/anaconda/miniconda)
        eval "$(conda shell.bash hook)"
    fi
    echo "Activating conda environment: $CONDA_ENV"
    source activate "$CONDA_ENV"
else
    echo "No conda environment specified"
fi


## Patch Extraction, Fold Splits:
bash "$script_dir/patch_extract_cv_splits.sh" \
    $PATCH_SIZE $EXTRACTOR $MAGNIF $BATCH_NAME $BATCH_SEED \
    $VAL_TR_FRAC $N_BINS $CLIN_FILE $OMICS_FILE $CONDA_ENV



## Feature Rearrangement:
echo "RUNNING FEATURE REARRANGEMENT"
feat_rear_script="$PROJ_ROOT/modules/preprocessing/embeddings/hvtsurv/use3_knn_position.py"
python3 $feat_rear_script \
    --h5-path $embed_out_dir/h5_files \
     --save-path $feat_rear_out_dir \
     --radius $WINDOW_SIZE --patch_size $PATCH_SIZE \
     --auto_skip True




## Random Window Masking: =====================================================
rand_wind_script="$PROJ_ROOT/modules/preprocessing/embeddings/hvtsurv/random_mask_window.py"

echo "RUNNING RANDOM WINDOW MASKING"

if [ "$BATCH_NAME" = "batchrandom" ]; then
    ## a: random CV splits:
    python3 $rand_wind_script --pt_dir $feat_rear_out_dir/ \
        --csv_dir $cv_split_rand_out_dir/ \
        --main_save_dir $rand_mask_out_dir \
        --window_size $WINDOW_SIZE --num_bag $NUM_BAG --masking_ratio $MASK_RATIO
elif [ "$BATCH_NAME" = "batchiso" ]; then
    ## b: batch iso CV splits:
    python3 $rand_wind_script --pt_dir $feat_rear_out_dir/ \
        --csv_dir $cv_split_batchiso_out_dir/ \
        --main_save_dir $rand_mask_out_dir \
        --window_size $WINDOW_SIZE --num_bag $NUM_BAG --masking_ratio $MASK_RATIO
elif [ "$BATCH_NAME" = "batchstrat" ]; then
    ## c: batch strat CV splits:
    python3 $rand_wind_script --pt_dir $feat_rear_out_dir/ \
        --csv_dir $cv_split_batchstrat_out_dir/ \
        --main_save_dir $rand_mask_out_dir \
        --window_size $WINDOW_SIZE --num_bag $NUM_BAG --masking_ratio $MASK_RATIO
else
    echo "Batch name variable is something unexpected"
fi




