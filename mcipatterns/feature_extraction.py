#!/bin/env python3
from tqdm import tqdm
from os.path import join, isfile, dirname, realpath

import pandas as pd
import numpy as np

from .adnimerge import load_adnimerge, get_subject_label


def compute_dataset_features(subjects, features_file, data_folder, roi_mask):
    features = pd.DataFrame()
    adnimerge = load_adnimerge()

    for subject_id in tqdm(subjects, ncols=150, desc='Subjects', unit='subj'):
        grad_file = join(data_folder, subject_id, 'gradients.npz')
        if not isfile(grad_file) or subject_id not in adnimerge.index:
            continue

        # Load edges per subject
        data = np.load(grad_file)
        g_mag, g_angle = data['g_mag'], data['g_angle']

        # Apply mask
        ix_mask = np.where(roi_mask)
        g_mag_vec = g_mag[ix_mask]
        g_angle_vec = g_angle[ix_mask]

        # Compute histogram of gradients
        hist_mag, bins_mag = np.histogram(g_mag_vec, bins=100, range=(0, 255))
        hist_angle, bins_angle = np.histogram(g_angle_vec, bins=100, range=(0, np.pi))

        feats = {
            'mag': zip(bins_mag, hist_mag),
            'angle': zip(bins_angle, hist_angle)
        }

        subj_features = pd.Series(name=subject_id)
        subj_features['label'] = get_subject_label(subject_id)

        for ax, hist_data in feats.items():
            for bin_name, bin_val in hist_data:
                subj_features[f'{ax}_{bin_name:.2f}'] = bin_val

        features = features.append(subj_features)
    # Label as first column
    arranged_cols = ['label'] + [col for col in features.columns if 'label' not in col]
    # Save feature matrix
    features[arranged_cols].to_csv(features_file)
