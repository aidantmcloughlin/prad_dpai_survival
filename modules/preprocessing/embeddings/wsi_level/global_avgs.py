
import os, sys
import re
import random
import numpy as np 
import pandas as pd
import argparse
from tqdm import tqdm

from git import Repo

import torch

## Git TopLevel Path:
from git import Repo
current_file_path = os.path.abspath(__file__)
repo = Repo(current_file_path, search_parent_directories = True)
GIT_ROOT = repo.git.rev_parse("--show-toplevel")

sys.path.append(os.path.join(GIT_ROOT, 'modules'))

from proj_constants import (
    DATA_PAR_DIR,
    EMBED_ROOT_DIR,
    LABEL_ROOT_DIR,
    ENCODER,
    PATCH_SIZE,
    MAGNIF,
)


## overwrite some of the main args if specified:
parser = argparse.ArgumentParser(
    description="Accept optional instructions of which encoding to perform."
    )

parser.add_argument('--encoder', type=str, required=False,)
parser.add_argument('--patch_size', type=int, required=False,)
parser.add_argument('--magnif', type=int, required=False)
parser.add_argument('--auto_skip', default=True)


args = parser.parse_args()

if args.encoder is None:
    args.encoder = ENCODER
if args.patch_size is None:
    args.patch_size = PATCH_SIZE
if args.magnif is None:
    args.magnif = MAGNIF
auto_skip = eval(str(args.auto_skip))    


wsi_feature_dir = os.path.join(
    EMBED_ROOT_DIR,
    str(args.patch_size) + '_' + str(args.magnif),
    args.encoder,
    )

pt_loc = os.path.join(wsi_feature_dir, '_2_features/pt_files')
glob_avg_out_loc = os.path.join(wsi_feature_dir, 'globavgs')


if not os.path.exists(glob_avg_out_loc):
    os.makedirs(glob_avg_out_loc)


pt_files = os.listdir(pt_loc)
slide_ids = [re.sub('.pt$', '', f) for f in pt_files]



## Computing the Global Averages over the patients:
for i in tqdm(range(len(slide_ids))):

    slide_globavg_savepath = os.path.join(glob_avg_out_loc, slide_ids[i] + ".pt")
    if auto_skip and os.path.isfile(slide_globavg_savepath):
        pass
    else:
        ### Load the patch feature vectors for a patient
        pt_features = torch.load(os.path.join(pt_loc, pt_files[i]))

        ## compute global mean over the patches
        patch_mean = torch.mean(pt_features, axis=0)
        ## save
        torch.save(patch_mean, slide_globavg_savepath)



