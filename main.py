#!/bin/env python3
import argparse
from os.path import join, dirname, isfile

from sklearn.model_selection import StratifiedKFold

from mcipatterns.model_extraction import extract_roi
from mcipatterns.feature_extraction import compute_dataset_features
from mcipatterns.adnimerge import filter_adni_conv_time
from mcipatterns.classification import classify


def parse_args():
    parser = argparse.ArgumentParser(description='Brain pattern characterization for MCI to AD progression.')
    parser.add_argument('--groupfile',
                        help='File containing subjects to study.',
                        default='/home/ssilvari/Documents/temp/ADNI_temp/ADNI_FS_registered_flirt/groupfile.csv')
    parser.add_argument('--data',
                        help='Folder containing brain\'s data',
                        default=None)
    parser.add_argument('--out',
                        help='Results folder',
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Define parameters
    groupfile_csv = args.groupfile
    data_folder = dirname(groupfile_csv) if not args.out else args.out
    out_folder = join(dirname(groupfile_csv), 'brain_models') if not args.data else args.data
    sep = '=' * 15

    # Define times of conversion
    conversion_times = [36, 48, 60]

    # For each time perform the pipeline
    for time in conversion_times:
        print(f'{sep * 3}')
        print(f'{sep} TIME: {time} MONTHS {sep}')
        print(f'{sep * 3}')
        print(f'{sep} ROI EXTRACTION {sep}')

        # Define features filename
        features_file = join(out_folder, f'features_{time}_months.csv')

        # Filter ADNIMERGE
        adnimerge = filter_adni_conv_time(conversion_time=time)
        subjects = adnimerge.index
        labels = adnimerge['TARGET'].astype('category')
        print(subjects)

        skf = StratifiedKFold(n_splits=3, random_state=42)
        for train_index, test_index in skf.split(subjects, labels):
            subjects_train, labels_train = subjects[train_index], labels[train_index]
            subjects_test, labels_test = subjects[train_index], labels[train_index]

            print(f'Dataset size: {len(subjects)}')
            print(f'Subjects for training: {len(train_index)}')
            print(f'Subjects for testing: {len(test_index)}')
            print(f'Data folder: {data_folder}')
            print(f'Output folder: {out_folder}')

            if not isfile(features_file):
                # Extract ROI
                print(f'{sep} ROI EXTRACTION {sep}')
                roi_mask, diff_map = extract_roi(subjects=subjects_train,
                                                 data_folder=data_folder,
                                                 out_folder=out_folder)
                # Compute features
                print(f'{sep} FEATURE COMPUTING {sep}')
                compute_dataset_features(subjects=subjects,
                                         data_folder=data_folder,
                                         roi_mask=roi_mask,
                                         features_file=features_file)
            # Perform classification
            classify(features_file=features_file,
                     subj_train=subjects_train,
                     subj_test=subjects_test)
            print('Done!')
            print()
