import os, sys, shutil
import re
import random
import argparse
import pickle as pkl
from functools import reduce
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
from skimpy import clean_columns
import umap
import openslide
from git import Repo
from sklearn.model_selection import train_test_split

## Paths:
current_file_path = os.path.abspath(__file__)
repo = Repo(current_file_path, search_parent_directories = True)

GIT_ROOT = repo.git.rev_parse("--show-toplevel")

sys.path.append(os.path.join(GIT_ROOT, 'modules'))
from proj_constants import (
    BATCH_SPLITTING_SEED, 
    DATA_PAR_DIR, METADATA_PAR_DIR, LABEL_ROOT_DIR,
    SLIDES_DIR, 
    BATCHES_IDX_EVEN, 
    VALID_PROP
    )

from modeling.utils_survival import intersect


## overwrite some of the main args if specified:
parser = argparse.ArgumentParser(
    description="Accept optional instructions of which encoding to perform."
    )
parser.add_argument('--clin_file', type=str, required=True,)
parser.add_argument('--omics_file', type=str, required=True,)
parser.add_argument('--batch_split_seed', type=int, required=False,)
parser.add_argument('--valid_prop', type=float, required=False,)
args = parser.parse_args()

if args.batch_split_seed is None:
    args.batch_split_seed = BATCH_SPLITTING_SEED
if args.valid_prop is None:
    args.valid_prop = VALID_PROP

random.seed(args.batch_split_seed)
np.random.seed(args.batch_split_seed)

### Load Batch Metadata
tcga_prad_batch_meta = pd.read_csv(os.path.join(METADATA_PAR_DIR, 'slide_metadata.csv'))

## rename sample_id column to reflect plurality if needed
if 'sample_id' in tcga_prad_batch_meta.columns:
    tcga_prad_batch_meta.rename(columns={'sample_id': 'sample_ids'}, inplace=True)


### Collecting available labels in the style of the HVTSurv label data loader

### HVT Surv:
fullrand_cvsplits_dir = os.path.join(
    LABEL_ROOT_DIR, 'seed' + str(args.batch_split_seed),
    'fullrandom')

# load hvt surv example split info csv.
fold0_orig = pd.read_csv(os.path.join(fullrand_cvsplits_dir, 'fold0.csv'))


## Collect Full Set of Unique Data from the CVSplit files, 
all_data = fold0_orig.iloc[0:0, 0:5]
for f in os.listdir(fullrand_cvsplits_dir):
    fold_df_train = pd.read_csv(os.path.join(fullrand_cvsplits_dir, f)).iloc[:, 0:5]
    fold_df_valid = pd.read_csv(os.path.join(fullrand_cvsplits_dir, f)).iloc[:, 5:10].dropna()
    fold_df_test = pd.read_csv(os.path.join(fullrand_cvsplits_dir, f)).iloc[:, 10:].dropna()
    fold_df_valid.columns = [re.sub('^val_', 'train_', c) for c in fold_df_valid.columns]
    fold_df_test.columns = [re.sub('^test_', 'train_', c) for c in fold_df_valid.columns]

    all_data = pd.concat([all_data, fold_df_train, fold_df_valid], axis=0)

all_data.drop_duplicates(inplace=True)


## Find intersection of multiple 3D arrays
meta_case_ids = tcga_prad_batch_meta['case_id'].apply(lambda x: x.split(',')[0])
genomic_data = pd.read_csv(args.omics_file)
clin_data = pd.read_csv(args.clin_file)

## Subset outcome data based on the intersection
avail_label_data = all_data[all_data['train_case_id'].isin(
    reduce(intersect, [
        meta_case_ids, 
        clin_data['case_id'], 
        genomic_data.columns,
        all_data['train_case_id'],
    ])
)] 


tcga_prad_batch_meta['slide_id'] = [
    re.sub('.svs', '', s) for s in tcga_prad_batch_meta['file_name']
]

tcga_prad_batch_meta_has_label = tcga_prad_batch_meta[
    tcga_prad_batch_meta['slide_id'].isin(avail_label_data['train_slide_id'])]


## Collapsing patients with slides from multiple batches:
multiple_batches = tcga_prad_batch_meta_has_label.groupby('case_id')['batch_id'].nunique()

# Filter to find case_ids with more than one unique batch_id
case_ids_with_multiple_batches = multiple_batches[multiple_batches > 1]

## replace with mode
def replace_with_most_frequent(x):
    mode_batch_id = x['batch_id'].mode()[0]  # Find the mode (most frequent) batch_id
    x['batch_id'] = mode_batch_id  # Replace all batch_id values with the mode
    return x

multibatch_samples = tcga_prad_batch_meta_has_label[tcga_prad_batch_meta_has_label['case_id'].isin(
    list(case_ids_with_multiple_batches.index))
    ].groupby('case_id').apply(replace_with_most_frequent).reset_index(drop=True)

tcga_prad_batch_meta_has_label = pd.concat((
    tcga_prad_batch_meta_has_label[~tcga_prad_batch_meta_has_label['case_id'].isin(
    list(case_ids_with_multiple_batches.index))],
    multibatch_samples
    ), axis=0).reset_index()


tcga_prad_batch_meta_clean = tcga_prad_batch_meta_has_label.copy()

flat_batch_include_list = [item for sublist in BATCHES_IDX_EVEN for item in sublist]
tcga_prad_batch_meta_clean = tcga_prad_batch_meta_clean[tcga_prad_batch_meta_clean['batch_id'].isin(
    flat_batch_include_list)]


batches_even = BATCHES_IDX_EVEN

n_folds = len(batches_even)

## apportion back to the format on the basis of batch-guided splits
tcga_prad_batch_meta_hvt = tcga_prad_batch_meta_clean.copy()


## A: batch-isolated CV randomization
batch_iso_fold_test_slides = [None] * n_folds
batch_iso_fold_train_slides = [None] * n_folds
batch_iso_fold_valid_slides = [None] * n_folds
for i in range(len(batches_even)):
    b = batches_even[i]
    batch_iso_fold_test_slides[i] = tcga_prad_batch_meta_hvt[
        tcga_prad_batch_meta_hvt['batch_id'].isin(b)]['slide_id'].values.tolist()
    
    batch_iso_fold_nontest_cases = np.unique(tcga_prad_batch_meta_hvt[
        ~tcga_prad_batch_meta_hvt['batch_id'].isin(b)]['case_id'].values).tolist()
    batch_iso_fold_nontest_slides = np.unique(tcga_prad_batch_meta_hvt[
        ~tcga_prad_batch_meta_hvt['batch_id'].isin(b)]['slide_id'].values).tolist()

    ## separating valid samples on a Case ID basis:    
    num_valid_cases = int(len(batch_iso_fold_nontest_cases) * VALID_PROP)
    sampled_valid_cases = random.sample(batch_iso_fold_nontest_cases, num_valid_cases, )
    
    batch_iso_fold_valid_slides[i] = tcga_prad_batch_meta_hvt[
        tcga_prad_batch_meta_hvt['case_id'].isin(sampled_valid_cases)]['slide_id'].values.tolist()
    
    batch_iso_fold_train_slides[i] = np.setdiff1d(
        batch_iso_fold_nontest_slides, 
        batch_iso_fold_valid_slides[i]).tolist()
    

##  indicating we might want to write some code to spread this batch over each of 
##  stratified folds, whereas now it is apportioned into the first couple folds.
slide_per_batch_info = tcga_prad_batch_meta_hvt.groupby('batch_id').agg(
    num_rows=pd.NamedAgg(column='case_id', aggfunc='size'),  # Count of rows per group
    num_distinct=pd.NamedAgg(column='case_id', aggfunc=pd.Series.nunique)  # Count of distinct 'Values'
)

slide_per_batch_info.head(12)


## Store the number of case IDs per batch
test_set_case_sizes = []
for i in range(n_folds):
    test_meta_fold = tcga_prad_batch_meta_hvt[tcga_prad_batch_meta_hvt['slide_id'].isin(
        batch_iso_fold_test_slides[i])]
    test_cases_fold = len(np.unique(test_meta_fold['case_id'].values))
    test_set_case_sizes.append(test_cases_fold)


## B: batch-stratified CV randomization (conforming to batch-separated test set sizes)
max_batch_id = np.max(np.unique(tcga_prad_batch_meta_hvt['batch_id'].values))


batch_strat_fold_test_cases = [[]] * n_folds
batch_strat_fold_test_slides = [[]] * n_folds
batch_strat_fold_train_slides = [[]] * n_folds
batch_strat_fold_valid_slides = [[]] * n_folds

def update_b(b, max_batch_id):
    if b < max_batch_id:
            b += 1
    else:
        b = 0
    return b

### Get the batch-stratified test sets using the previous batch separated test set sizes.
## randomly rearrange the DF
tcga_prad_batch_meta_hvt_shuffle = tcga_prad_batch_meta_hvt.sample(frac=1).reset_index(drop=True)

remaining_df = tcga_prad_batch_meta_hvt_shuffle.loc[:, ['case_id', 'batch_id']].drop_duplicates()
## include censor status for stratification
remaining_df = pd.merge(
    remaining_df, 
    avail_label_data.loc[:, ['train_case_id', 'train_censorship']].rename(
        columns={'train_case_id': 'case_id', 'train_censorship': 'censorship'}).drop_duplicates()
        )


remaining_df['stratify_column'] = remaining_df['batch_id']
# Loop through each fold size and split
for i, test_size in enumerate(test_set_case_sizes):
    ## Create a fold name (e.g., fold1, fold2, ...)
    
    ## Perform a stratified split, ensuring stratification based on 'batch_id'
    ## set to batch if only 1 uncensored value
    strata_counts = remaining_df['stratify_column'].value_counts()
    rare_strata = list(strata_counts[strata_counts == 1].index)
    rare_strata_batches = [int(s.split("_")[0]) for s in rare_strata]
    for b in rare_strata_batches:
        remaining_df.loc[remaining_df['batch_id'] == b, 'stratify_column'] = str(b)
    ## stratified sampling one fold at a time.

    ## check if before last fold:
    if i < len(test_set_case_sizes) - 1:
        remaining_df, batch_strat_fold_test_cases_df = train_test_split(
            remaining_df, test_size=test_size, stratify=remaining_df['stratify_column']
            )
        
        batch_strat_fold_test_cases[i] = list(batch_strat_fold_test_cases_df['case_id'].to_numpy())
        
    else:
        ## at the last fold:
        batch_strat_fold_test_cases[i] = list(remaining_df['case_id'].drop_duplicates())

    ## assign corresponding slides:
    sampled_slides = tcga_prad_batch_meta_hvt[
            tcga_prad_batch_meta_hvt['case_id'].isin(batch_strat_fold_test_cases[i])]['slide_id']
    batch_strat_fold_test_slides[i] = sampled_slides



## Now apportion the train valid data in analogous way to before

for i in range(n_folds):
    
    batch_strat_fold_nontest_slides = np.setdiff1d(
        tcga_prad_batch_meta_hvt['slide_id'].values, 
        batch_strat_fold_test_slides[i]).tolist()

    batch_strat_fold_nontest_cases = np.unique(tcga_prad_batch_meta_hvt[
        tcga_prad_batch_meta_hvt['slide_id'].isin(batch_strat_fold_nontest_slides)
        ]['case_id'].values).tolist()

    num_valid_samples = int(len(batch_strat_fold_nontest_cases) * VALID_PROP)
    batch_strat_fold_valid_cases = random.sample(
        batch_strat_fold_nontest_cases, 
        num_valid_samples, )
    batch_strat_fold_valid_slides[i] = tcga_prad_batch_meta_hvt[
        tcga_prad_batch_meta_hvt['case_id'].isin(batch_strat_fold_valid_cases)
    ]['slide_id']
    batch_strat_fold_train_slides[i] = np.setdiff1d(
        batch_strat_fold_nontest_slides, 
        batch_strat_fold_valid_slides[i]).tolist()




## Function to recreate the HVTSurv CV Split DF given train/valid/test slideIDs:

def createHVTSurvCVDF(outcome_df, train_slides, valid_slides, test_slides):
    ## name standardization:
    outcome_df.columns = ['case_id', 'slide_id', 'survival_months', 'censorship', 'disc_label']

    n_folds = len(train_slides)
    fold_dfs = [None] * n_folds

    for i in range(n_folds):
        fold_train_df = outcome_df.loc[outcome_df['slide_id'].isin(train_slides[i]), :]
        fold_valid_df = outcome_df.loc[outcome_df['slide_id'].isin(valid_slides[i]), :]
        fold_test_df = outcome_df.loc[outcome_df['slide_id'].isin(test_slides[i]), :]

        fold_train_df.columns = 'train_' + fold_train_df.columns
        fold_valid_df.columns = 'val_' + fold_valid_df.columns
        fold_test_df.columns = 'test_' + fold_test_df.columns

        fold_train_df = fold_train_df.reset_index(drop=True)
        fold_valid_df = fold_valid_df.reset_index(drop=True)
        fold_test_df = fold_test_df.reset_index(drop=True)

        fold_dfs[i] = pd.concat([fold_train_df, fold_valid_df, fold_test_df], axis=1)
    
    return fold_dfs


## Batch Isolated
batch_iso_fold_dfs = createHVTSurvCVDF(
    avail_label_data, 
    batch_iso_fold_train_slides,
    batch_iso_fold_valid_slides,
    batch_iso_fold_test_slides,
    ) 

## Batch Random Stratified
batch_strat_fold_dfs = createHVTSurvCVDF(
    avail_label_data, 
    batch_strat_fold_train_slides,
    batch_strat_fold_valid_slides,
    batch_strat_fold_test_slides,
    ) 


## Function to check the separation of case IDs

def checkCaseIDIntersect(fold_df):
    train_case = fold_df['train_case_id'].dropna().values
    valid_case = fold_df['val_case_id'].dropna().values
    test_case = fold_df['test_case_id'].dropna().values

    n_overlap_case = (
        len(np.intersect1d(train_case, valid_case)) + 
        len(np.intersect1d(train_case, test_case)) + 
        len(np.intersect1d(valid_case, test_case)) 
    )

    return n_overlap_case

### Save each of the newly created data frames in the format for HVTSurv model training.
batch_iso_loc = os.path.join(LABEL_ROOT_DIR, 'seed'+str(args.batch_split_seed), 'batchiso')
batch_strat_loc = os.path.join(LABEL_ROOT_DIR, 'seed'+str(args.batch_split_seed), 'batchstrat')


if os.path.exists(batch_iso_loc):
    shutil.rmtree(batch_iso_loc)
if os.path.exists(batch_strat_loc):
    shutil.rmtree(batch_strat_loc)

os.makedirs(batch_iso_loc, exist_ok=True)
os.makedirs(batch_strat_loc, exist_ok=True)

for i in range(len(batch_iso_fold_dfs)):
    batch_iso_fold_dfs[i].to_csv(os.path.join(batch_iso_loc, "fold" + str(i) + ".csv"), index=False)
    batch_strat_fold_dfs[i].to_csv(os.path.join(batch_strat_loc, "fold" + str(i) + ".csv"), index=False)

