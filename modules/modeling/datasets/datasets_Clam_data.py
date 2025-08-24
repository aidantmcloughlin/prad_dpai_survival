import os, sys
import random
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import torch.utils.data as data
from torch.utils.data import dataloader
from sklearn.decomposition import PCA

## local modules:
from utils_survival import read_yaml, preprocess_clin_data

#---->Remove coordinate information
class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[:,2:]


class ClamData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.wsifeat_dir
        self.clin_file = self.dataset_cfg.clin_file
        self.csv_dir = os.path.join(self.dataset_cfg.label_dir, f'fold{self.fold}.csv')
        self.slide_data = pd.read_csv(self.csv_dir,)
        self.max_slides = self.dataset_cfg.max_slides

        self.clin_data = pd.read_csv(self.clin_file,)

        ## Get standardizing factors (sci-kit transformer) for the clin data
        _, self.train_preprocessor = preprocess_clin_data(
            self.clin_data.loc[self.clin_data['case_id'].isin(
                self.slide_data['train_case_id'].dropna()
                ), :],
                cols_keep=self.dataset_cfg.clin_vars,
            )

        ### learn PCA embedder, if specified
        if self.dataset_cfg.pc_dim is None or self.dataset_cfg.pc_dim == {}:
            pass
        else:
            ### Load the training data
            wsi_vec_list = []
            for slide_id in self.slide_data['train_slide_id'].dropna():
                full_path = Path(self.feature_dir) / f'{slide_id}.pt'
                wsi_vec_list.append(torch.load(full_path))

            wsi_vec_mat = torch.stack(wsi_vec_list).numpy()
            self.pca_model = PCA(n_components=self.dataset_cfg.pc_dim)
            self.pca_model.fit(wsi_vec_mat)

        #---->Split the dataset
        if state == 'train':
            self.data = self.slide_data['train_slide_id'].dropna()
            self.survival_months = self.slide_data['train_survival_months'].dropna()
            self.censorship = self.slide_data['train_censorship'].dropna()
            self.case_id = self.slide_data['train_case_id'].dropna()
            self.label = self.slide_data['train_disc_label'].dropna()

            ## preprocess clinical data:
            self.clin_model_data, _ = preprocess_clin_data(
                    self.clin_data.loc[self.clin_data['case_id'].isin(self.case_id)], 
                    cols_keep=self.dataset_cfg.clin_vars,
                    train_preprocessor=self.train_preprocessor,
                    )


        if state == 'val':
            self.data = self.slide_data['val_slide_id'].dropna()
            self.survival_months = self.slide_data['val_survival_months'].dropna()
            self.censorship = self.slide_data['val_censorship'].dropna()
            self.case_id = self.slide_data['val_case_id'].dropna()
            self.label = self.slide_data['val_disc_label'].dropna()

            ## preprocess clinical data:
            self.clin_model_data, _ = preprocess_clin_data(
                self.clin_data.loc[self.clin_data['case_id'].isin(self.case_id)], 
                cols_keep=self.dataset_cfg.clin_vars,
                train_preprocessor=self.train_preprocessor
                )

        if state == 'test':
            self.data = self.slide_data['test_slide_id'].dropna()
            self.survival_months = self.slide_data['test_survival_months'].dropna()
            self.censorship = self.slide_data['test_censorship'].dropna()
            self.case_id = self.slide_data['test_case_id'].dropna()
            self.label = self.slide_data['test_disc_label'].dropna()

            ## preprocess clinical data:
            self.clin_model_data, _ = preprocess_clin_data(
                self.clin_data.loc[self.clin_data['case_id'].isin(self.case_id)], 
                cols_keep=self.dataset_cfg.clin_vars,
                train_preprocessor=self.train_preprocessor
                )

        #---->Concat related information together
        splits = [self.data, self.survival_months, self.censorship, self.case_id, self.label]
        self.split_data = pd.concat(splits, ignore_index = True, axis=1)
        self.split_data.columns = [
            'slide_id', 
            'survival_months', 
            'censorship', 
            'case_id', 
            'disc_label']

        #---->get patient data
        self.patient_df = self.split_data.drop_duplicates(['case_id']).copy()
        self.patient_df.set_index(keys='case_id', drop=True, inplace=True)
        self.split_data.set_index(keys='case_id', drop=True, inplace=True)

        ## store clin var names
        self.clin_var_names = self.clin_model_data.columns

        #---->Establish a connection between patient_df and data
        self.patient_dict = {}
        for patient in self.patient_df.index:
            slide_ids = self.split_data.loc[patient, 'slide_id'] #get the case_id
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            self.patient_dict.update({patient:slide_ids}) #which WSIs are included in each patient

        self.patient_df.reset_index(inplace=True)
        ## get the patient-level labels
        self.survival_months = self.patient_df['survival_months']
        self.censorship = self.patient_df['censorship']
        self.case_id = self.patient_df['case_id']
        self.label = self.patient_df['disc_label']

        

    def __len__(self):
        return len(self.patient_df)

    def __getitem__(self, idx):
        case_id = self.case_id[idx]
        event_time = self.survival_months[idx]
        censorship = self.censorship[idx]
        label = self.label[idx]
        slide_ids = self.patient_dict[case_id].tolist()
        slide_ids_short = np.sort(slide_ids).tolist()[0:int(self.max_slides)]

        ## get the clin data:
        clin_features = self.clin_model_data.loc[case_id,:].values
        clin_features = torch.Tensor(clin_features)
        
        ## get the WSI data:
        wsi_features = []
        for slide_id in slide_ids_short:
            full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            try:
                wsi_features.append(torch.load(full_path))
            except:
                print(full_path)
        
        if len(wsi_features) > 1:
            print("MULTIPLE SLIDES FOR THE CASE")
            wsi_features_stack = torch.concatenate(wsi_features, dim=0)
        else:
            wsi_features_stack = wsi_features[0]

        # wsi_features_stack = torch.squeeze(wsi_features_stack, )

        slide_ids_short = [";".join(slide_ids_short)]

        clin_features_list = [clin_features, ]

        return (
            case_id, slide_ids_short, 
            wsi_features_stack, torch.zeros(1),
            clin_features_list, torch.zeros(1),
            label, event_time, censorship
            )


