# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains tools used for calibrationless reconstruction.
"""
# Third module import
import numpy as np
from sklearn.feature_extraction.image import extract_patches
from itertools import product
from modopt.signal.noise import thresh


def extract_patches_2d(images, patch_shape, overlapping_factor=1):
    """
    This function extracts overlapping and non-overlapping patches from 2D
    volume

    Parameters:
    -----------
    images: np.ndarray of shape [Nch, Nx, Ny]
        The 2D volume with the number of chanels first
    patch_shape: tuple
        The 2D patch_shape of type (patch_size_x, patch_size_y)
    overlapping_factor: int, (optional, default 1)
        The patches overlapping factor

    Returns:
    np.ndarray of shape [np, Nch, patch_size_x, patch_size_y] where np is the
    number of extracted patch
    """
    Nch, Nx, Ny = images.shape
    patch_size_x, patch_size_y = patch_shape

    patch_step_size = (Nch,
                       np.maximum(int(patch_size_x/overlapping_factor), 1),
                       np.maximum(int(patch_size_y/overlapping_factor), 1))

    if patch_size_x > Nx:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size_y > Ny:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    extracted_patches = extract_patches(images,
                                        patch_shape=(Nch,
                                                     patch_size_x,
                                                     patch_size_y),
                                        extraction_step=patch_step_size)

    return extracted_patches.reshape((-1, Nch, patch_size_x, patch_size_y))


def recombine_patches_2d(patches, img_shape, overlapping_factor=1):
    """
    This function reconstruct a 2D multichanel volume from the sets of patches
    given as input

    Parameters:
    -----------
    patches: np.ndarray of shape [np, Nch, patch_size_x, patch_size_y]
        The extracted patches from the image
    img_shape: tuple
        The target image shape [Nch, Nx, Ny]
    overlapping_factor: int, (optional, default 1)
        The patches overlapping factor

    Returns:
    np.ndarray of shape [Nch, Nx, Ny]
    """
    Nch, Nx, Ny = img_shape
    n_patch, Nch, patch_size_x, patch_size_y = patches.shape

    img = np.zeros(img_shape, dtype=patches.dtype)

    # Computes the true dimension of the patches array
    n_x = Nx - patch_size_x + 1
    n_y = Ny - patch_size_y + 1

    patch_step_size = (Nch,
                       np.maximum(int(patch_size_x/overlapping_factor), 1),
                       np.maximum(int(patch_size_y/overlapping_factor), 1))

    ratio_x = patch_size_x * 1.0 / patch_step_size[1]
    ratio_y = patch_size_y * 1.0 / patch_step_size[2]

    stop_x = int((ratio_x - 1) * patch_step_size[1])
    stop_y = int((ratio_y - 1) * patch_step_size[2])

    vect_n_x = np.arange(0, Nx - stop_x, patch_step_size[1])
    vect_n_y = np.arange(0, Ny - stop_y, patch_step_size[2])

    weights = np.zeros(img_shape)

    for p, (i, j) in zip(patches, product(vect_n_x, vect_n_y)):
        img[:, i:i + patch_size_x, j:j + patch_size_y] += p
        weights[:, i:i + patch_size_x, j:j + patch_size_y] += 1

    return img / weights


def _svd_thresh(patch, threshold, threshold_type='soft'):
    """
    This method computes the thresholding of the singular value decomposition
    of the input data

    Parameters:
    -----------
    patch: np.ndarray
        Input data of shape [nch, nx, ny, nz]
    threshold: float
        Threshold value
    threshold_type: str, {'hard', 'soft'}
        Threshold type (default is 'soft')

    Return:
    -------
    np.ndarray
        Results of the singular value thresholding
    """
    u, s, vh = np.linalg.svd(np.reshape(
        patch,
        (patch.shape[0], np.prod(patch.shape[1:]))),
        full_matrices=False)

    s = thresh(s, threshold, threshold_type=threshold_type)

    patch = np.reshape(
        np.dot(u * s, vh),
        patch.shape)
    return patch


def _svd_cost(patch, weights):
    """
    This computes the weighted sum of the singular value of the input data

    Parameters:
    -----------
    patch: np.ndarray
        Input data of shape [nch, nx, ny]
    weights: float
        The weights used

    Return:
    -------
    float
        The weighted sum of the singular value decomposition
    """
    s = np.linalg.svd(np.reshape(
        patch,
        (patch.shape[0], np.prod(patch.shape[1:]))),
        full_matrices=False,
        compute_uv=False)
    return np.sum(weights * np.abs(s.flatten()))
