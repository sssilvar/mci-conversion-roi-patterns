#!/bin/env python3
import os
import argparse
from os.path import join, dirname, basename

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from mcipatterns.model_extraction import extract_roi
from mcipatterns.feature_extraction import compute_dataset_features
from mcipatterns.adnimerge import filter_adni_conv_time
from mcipatterns.classification import classify

plt.switch_backend('agg')
plt.style.use('seaborn')


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

        skf = StratifiedKFold(n_splits=10, random_state=42)
        plt.figure(figsize=(19.2 * 0.75, 10.8 * 0.75), dpi=150)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)

        for train_index, test_index in skf.split(subjects, labels):
            subjects_train, labels_train = subjects[train_index], labels[train_index]
            subjects_test, labels_test = subjects[test_index], labels[test_index]

            print(f'Dataset size: {len(subjects)}')
            print(f'Subjects for training: {len(train_index)}')
            print(f'Subjects for testing: {len(test_index)}')
            print(f'Data folder: {data_folder}')
            print(f'Output folder: {out_folder}')
            print(f'Train intersection test: {set(subjects_train).intersection(set(subjects_test))}')

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
            metrics, fpr, tpr = classify(features_file=features_file,
                                         subj_train=subjects_train,
                                         subj_test=subjects_test)

            # Plot ROC and save it
            plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]}')

        # Final touches
        plt.legend()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')

        # Create folder with classification results
        clf_folder = join(out_folder, 'classification')
        os.makedirs(clf_folder, exist_ok=True)

        # Save figure to disk
        fig_file = basename(features_file).replace('csv', 'eps')
        fig_file = join(clf_folder, fig_file)
        plt.savefig(fig_file, bbox_inches='tight')
        print('Done!')
        print()
