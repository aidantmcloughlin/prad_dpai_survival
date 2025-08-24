import os, sys
import random
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import re
import torch.utils.data as data
from torch.utils.data import dataloader

## local modules:
from utils_survival import read_yaml, preprocess_clin_data

#---->Remove coordinate information
class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[:,2:]


class HvtData(data.Dataset):
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
        self.split_data.columns = ['slide_id', 'survival_months', 'censorship', 'case_id', 'disc_label']

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
        #---->！！！！！for the random mask strategy, get the patient-level label
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

        ## get the clin data:
        clin_features = self.clin_model_data.loc[case_id,:].values
        clin_features = torch.Tensor(clin_features)

        wsi_features = []

        ## limiting N WSIs ingested, if surpassing limit (helps reduce memory demands.)
        wsi_ids = [re.sub('(^.*)_\d$', '\\1', s) for s in slide_ids]
        n_wsi = len(np.unique(wsi_ids))
        n_sub_wsi = len(wsi_ids) / n_wsi
        slide_ids_short = np.sort(slide_ids).tolist()[0:int(self.max_slides * n_sub_wsi)]
        
        for slide_id in slide_ids_short:
            full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            try:
                wsi_features.append(torch.load(full_path))
            except:
                print(full_path)

        clin_features_list = [clin_features, ]
        
        return (
            case_id, slide_ids_short, 
            wsi_features, torch.zeros(1), 
            clin_features_list, torch.zeros(1),
            label, event_time, censorship
            )


from utils_survival import read_yaml
if __name__ == '__main__':
    cfg = read_yaml('BRCA/HVTSurv.yaml')
    Mydata = TcgarandomData(dataset_cfg=cfg.Data, state='train')
    dataloader = data.dataloader(Mydata)
    for i, data in enumerate(dataloader):
        pass
