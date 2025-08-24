#!/bin/bash
set -euo pipefail
## Extract Patches, Patch Features from WSIs ==================================

# Get the directory of the current script
script_dir=$(dirname "$(readlink -f "$0")")

## Collect directories 
dir_yaml_file=$script_dir/current_proj_dirs.yaml
# Load the constants from the text file located one directory up
GIT_ROOT=$(yq eval '.GIT_ROOT' $dir_yaml_file)
DATA_PAR_DIR=$(yq eval '.DATA_PAR_DIR' $dir_yaml_file)
ALL_DATA_PAR_DIR=$(yq eval '.ALL_DATA_PAR_DIR' $dir_yaml_file)
PROJ_REL_LOC=$(yq eval '.PROJ_REL_LOC' $dir_yaml_file)
LABEL_ROOT_DIR=$(yq eval '.LABEL_ROOT_DIR' $dir_yaml_file)
EMBED_ROOT_DIR=$(yq eval '.EMBED_ROOT_DIR' $dir_yaml_file)
UNI_CKPT_DIR=$(yq eval '.UNI_CKPT_DIR' $dir_yaml_file)

PROJ_ROOT="$GIT_ROOT/$PROJ_REL_LOC"

## Read the arguments

## For Patch Extraction:
PATCH_SIZE=${1:-1024}
EXTRACTOR=${2:-"UNI"}
MAGNIF=${3:-40}

## For CV Splitting.
BATCH_NAME=${4:-"batchstrat"}
BATCH_SEED=${5:-1}
VAL_TR_FRAC=${6:-0.2}
N_BINS=${7:-4} ## number discretized bins 

CLIN_FILE=$8
OMICS_FILE=$9

CONDA_ENV=${10:-""}


PATCH_EXT_BATCH_SIZE=20

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

## Data and Output directories:
slides_dir="$DATA_PAR_DIR/wsi_files/"
slide_ids_file="$DATA_PAR_DIR/tabular_data/metadata/wsi_svs_fnames.csv"
histo_data_root="$EMBED_ROOT_DIR"
feature_data_root="$histo_data_root/${PATCH_SIZE}_${MAGNIF}"
patch_out_dir="$feature_data_root/_1_tiles"
embed_out_dir="$feature_data_root/$EXTRACTOR/_2_features"
export UNI_CKPT_PATH="$UNI_CKPT_DIR"

## Extracting patches from WSIs: ==============================================
echo $slides_dir
echo $patch_out_dir
echo $UNI_CKPT_PATH
echo "RUNNING PATCH EXTRACTOR ============================================"
patch_script="$PROJ_ROOT/modules/preprocessing/embeddings/CLAM/create_patches_fp.py"
python $patch_script --source $slides_dir --step_size $PATCH_SIZE \
    --patch_size $PATCH_SIZE --magnif $MAGNIF \
    --patch --seg --stitch \
    --save_dir $patch_out_dir
    

## Tile-Level Feature Extraction: =============================================
if [ "$EXTRACTOR" == "UNI" ]; then
    echo "RUNNING TILE LEVEL FEATURE EXTRACTION: UNI =========================="
    MODEL_NAME='uni_v1'
elif [ "$EXTRACTOR" == "imagenet" ]; then
    echo "RUNNING TILE LEVEL FEATURE EXTRACTION: imagenet ====================="
    MODEL_NAME='resnet50_trunc'
else
    echo "Invalid extractor name provided for feature extraction module."
fi

embed_script="$PROJ_ROOT/modules/preprocessing/embeddings/CLAM/extract_features_fp.py"
    CUDA_VISIBLE_DEVICES=0 python $embed_script --data_h5_dir "$patch_out_dir" \
        --data_slide_dir $slides_dir \
        --slide_ext .svs --csv_path $slide_ids_file \
        --feat_dir $embed_out_dir --batch_size $PATCH_EXT_BATCH_SIZE \
        --model_name $MODEL_NAME


## Preparing any CV fold splits.

## Original [even fold sizing, randomg]: ======================================
cv_fold_script="$PROJ_ROOT/modules/preprocessing/labels_prep/orig_cv_splits.py"

fullrand_labels_out_dir="$LABEL_ROOT_DIR/seed$BATCH_SEED/fullrandom"
labels_in_csv="$DATA_PAR_DIR/tabular_data/clindata/outcomes.csv"

python3 $cv_fold_script --input_csv $labels_in_csv --output_dir $fullrand_labels_out_dir \
    --frac $VAL_TR_FRAC --n_bins $N_BINS --data_seed $BATCH_SEED

## Batch Informed CV Splits: ==================================================
if [ "$BATCH_NAME" == "fullrandom" ]; then
    echo "Fully Random Data Splits."
elif [ "$BATCH_NAME" == "batchrandom" ] || [ "$BATCH_NAME" == "batchiso" ] || [ "$BATCH_NAME" == "batchstrat" ]; then
    echo "Running Batch Splitting Script."  


    ## Batch Split Script:
    batch_split_script="$PROJ_ROOT/modules/preprocessing/labels_prep/batch_informed_cv_splits.py"
    python3 $batch_split_script --clin_file "$DATA_PAR_DIR/$CLIN_FILE" --omics_file "$DATA_PAR_DIR/$OMICS_FILE" --batch_split_seed $BATCH_SEED --valid_prop $VAL_TR_FRAC
else
    echo "Batch name variable is something unexpected"
fi
