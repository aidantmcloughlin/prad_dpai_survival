import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def add_disc_label(data, n_bins=4):
    
    label_col = 'survival_months'
    eps = 1e-6

    uncensored_data = data[data['censorship'] < 1].copy()
    disc_labels, q_bins = pd.qcut(
        uncensored_data[label_col], q=n_bins, retbins=True, 
        labels=False)
    q_bins[-1] = data[label_col].max() + eps
    q_bins[0] = data[label_col].min() - eps

    disc_labels, _ = pd.cut(
        data[label_col], bins=q_bins, retbins=True, 
        labels=False, right=False, include_lowest=True
        )
    data['disc_label'] = disc_labels.astype(int)

    return data


def prepare_dataset(input_csv, output_dir, n_splits, frac, n_bins, seed=42):

    data = pd.read_csv(input_csv)
    data = add_disc_label(data, n_bins)

    data_unique_patients = data.drop_duplicates(subset=['case_id'])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_index, test_index) in enumerate(
        skf.split(data_unique_patients, data_unique_patients['censorship'])):

        train_case_ids_ini, test_case_ids = (
            data_unique_patients.iloc[train_index]['case_id'], 
            data_unique_patients.iloc[test_index]['case_id']
            )
        train_data_ini = data[data['case_id'].isin(train_case_ids_ini)]
        test_data = data[data['case_id'].isin(test_case_ids)]
        
        train_case_ids, val_case_ids, _, _ = train_test_split(
            train_data_ini.drop_duplicates(subset=['case_id'])['case_id'],
            train_data_ini.drop_duplicates(subset=['case_id'])['censorship'],
            test_size=frac, random_state=seed,
            stratify=train_data_ini.drop_duplicates(subset=['case_id'])['censorship'])
        train_data = train_data_ini[train_data_ini['case_id'].isin(train_case_ids)]
        val_data = train_data_ini[train_data_ini['case_id'].isin(val_case_ids)]

        fold_filename = f'{output_dir}/fold{fold}.csv'

        train_set = pd.DataFrame(train_data.rename(columns={
            'case_id': 'train_case_id', 'slide_id': 'train_slide_id', 
            'survival_months': 'train_survival_months', 
            'censorship': 'train_censorship', 
            'disc_label':'train_disc_label'})).reset_index(drop=True)
        val_set = pd.DataFrame(val_data.rename(columns={
            'case_id': 'val_case_id', 'slide_id': 'val_slide_id', 
            'survival_months': 'val_survival_months', 
            'censorship': 'val_censorship', 
            'disc_label':'val_disc_label'})).reset_index(drop=True)
        test_set = pd.DataFrame(test_data.rename(columns={
            'case_id': 'test_case_id', 'slide_id': 'test_slide_id', 
            'survival_months': 'test_survival_months', 
            'censorship': 'test_censorship', 
            'disc_label':'test_disc_label'})).reset_index(drop=True)

        fold_data = pd.concat([train_set, val_set, test_set], axis=1)

        fold_data.to_csv(fold_filename, index=False)



def main():
    
    parser = argparse.ArgumentParser(description='Prepare data for Cross-Validation')
    parser.add_argument("--input_csv", type=str, help="Path to csv file with survival data")
    parser.add_argument("--output_dir", type=str, help="Directory to save csv for cross-validation")
    parser.add_argument("--n", default=5, type=int, help="Number of folds for cross-validation")
    parser.add_argument(
        "--frac", default=0.2, type=float, 
        help="Frac for valid set to split the train set into train and valid sets")
    parser.add_argument("--n_bins", default=4, type=int, help="Number of dicrete labels for survival_months")
    parser.add_argument("--data_seed", default=42, type=int, help="Random State for the Train-Valid-Test splits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prepare_dataset(
        args.input_csv, args.output_dir, args.n, 
        args.frac, args.n_bins, args.data_seed
        )

    print(f"Data preparation for {args.n} folds completed. See {args.output_dir} folder.")


if __name__ == '__main__':
    main()