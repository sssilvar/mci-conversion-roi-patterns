from os.path import join, dirname, isfile

import numpy as np
import nibabel as nb


def compute_edge_space(img_file, method='gradient', out_folder=None, return_affine=False):
    # Define output filename
    if out_folder:
        out_file = join(out_folder, 'gradients.npz')
    else:
        out_file = join(dirname(img_file), 'gradients.npz')

    # Check if already computed
    if isfile(out_file):
        data = np.load(out_file)
        g_mag, g_angle, affine = data['g_mag'], data['g_angle'], data['affine']
    else:
        # Load image
        nii = nb.load(img_file)
        vol = nii.get_data().astype(np.float32)
        affine = nii.affine

        # Compute gradient
        if method == 'gradient':
            gx, gy, gz = np.gradient(vol)
            g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
            g_angle = np.nan_to_num(np.arctan2(np.sqrt(gx ** 2 + gy ** 2), gz))

            np.savez_compressed(out_file,
                                g_mag=g_mag,
                                g_angle=g_angle,
                                affine=affine)
    if return_affine:
        return g_mag, g_angle, affine
    else:
        return g_mag, g_angle
