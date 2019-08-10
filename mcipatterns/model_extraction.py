#!/bin/env python3
import os
from os.path import join, dirname, realpath, isfile

import nibabel as nb
import numpy as np
import pandas as pd
from skimage import filters
import scipy.ndimage as ndi
from scipy.stats import describe

from .compute_gradient import compute_edge_space
from .adnimerge import load_adnimerge, get_subject_label

root = dirname(dirname(realpath(__file__)))


def load_subject_volume(subject_id, data_folder, space='intensity'):
    try:
        subj_label = get_subject_label(subject_id)
        nii_file = join(data_folder, subject_id, 'brainmask_reg.nii.gz')

        # Check if file exists
        if not isfile(nii_file):
            raise KeyError(f'File {nii_file} not found.')

        # Extract the requested space (intensity/edge)
        if space == 'intensity':
            return nb.load(nii_file), subj_label
        elif space == 'edge':
            mag, angle, affine = compute_edge_space(img_file=nii_file, return_affine=True)
            return mag, angle, affine, subj_label
    except KeyError:
        return None, None, None, None


def extract_roi(subjects, data_folder, out_folder, th_method='otsu'):
    print(f'Building model ...')
    stable_model = np.zeros(shape=(256, 256, 256))
    conversion_model = np.zeros_like(stable_model)
    n_stables = n_converters = 0  # Reset counters
    affine = np.eye(4)  # Default affine

    for subject in subjects:
        # Load subject volume
        mag, angle, affine, label = load_subject_volume(subject_id=subject,
                                                        data_folder=data_folder,
                                                        space='edge')
        if mag is None:
            continue  # Ignore subject

        # Add up to model
        if label == 'MCInc':
            stable_model += mag
            n_stables += 1
        elif label == 'MCIc':
            conversion_model += mag
            n_converters += 1
    # Compute average models
    stable_model /= n_stables
    conversion_model /= n_converters
    difference_map = np.abs(stable_model - conversion_model)

    # Thresholding
    if th_method == 'otsu':
        threshold = filters.threshold_otsu(difference_map[np.where(difference_map > 0)])
    else:
        threshold = filters.threshold_yen(difference_map)

    # Raw ROI and eroded
    roi_mask = (difference_map >= threshold).astype(np.int8)
    roi_mask_closed = ndi.binary_erosion(roi_mask).astype(np.int8)

    models_dict = {
        'MCInc': stable_model,
        'MCIc': conversion_model,
        'diff': difference_map,
        f'{th_method}_mask': roi_mask,
        'eroded_mask': roi_mask_closed
    }

    # Create output folder
    os.makedirs(out_folder, exist_ok=True)

    # Create and save models as NIFTI images
    print(f'Saving models ...')
    for label, vol in models_dict.items():
        nii_model = nb.Nifti1Image(vol, affine)
        nb.save(nii_model, join(out_folder, f'{label.lower()}_model.nii.gz'))
    print(f'\t- Stable subjects: {n_stables}')
    print(f'\t- Converter subjects: {n_converters}')
    print(f'\t- Threshold: {threshold:.2f} ({th_method})')
    print(f'\t- Stats: {describe(stable_model.ravel())}')
    return roi_mask, difference_map
