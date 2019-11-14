# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2020                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

# Package import
from .utils import _svd_thresh, _svd_cost, extract_patches_2d, \
    recombine_patches_2d

# Third party import
import numpy as np
from modopt.opt.proximity import ProximityParent, LowRankMatrix
from joblib import Parallel, delayed


class PatchLowRank(ProximityParent):
    """The proximity of the patch based nuclear norm operator

    This class defines the nuclear norm proximity operator on a patch based
    method

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    patch_shape: tuple
        Shape of the patches extracted patches
    overlapping_factor: int, (optional, default 1)
        The patches overlapping_factor, for non-overlaped patches must be
        equal to 1
    thresh_type : str {'hard', 'soft'}, (optional, default 'soft')
        Threshold type (default is 'soft')
    num_cores: int
        Number of cores to run the decompositsion in parallel

    Notes:
    ------
    Only supports 2D images
    """
    def __init__(self, weights, patch_shape, overlapping_factor=1,
                 thresh_type='soft', num_cores=1):

        self.weights = weights
        self.patch_shape = patch_shape
        self.overlapping_factor = overlapping_factor
        self.thresh_type = thresh_type
        self.num_cores = num_cores
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, data, extra_factor=1.0):
        """Operator
        This method the input data after the singular values of the extracted
        patches have been thresholded

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray SVD thresholded data
        """
        patches = extract_patches_2d(
            images=data,
            patch_shape=self.patch_shape,
            overlapping_factor=self.overlapping_factor)

        patches = Parallel(n_jobs=self.num_cores, prefer="threads")(
            delayed(_svd_thresh)(
                        patch=patches[idx],
                        threshold=self.weights * extra_factor,
                        threshold_type=self.thresh_type)
            for idx in range(patches.shape[0]))

        return recombine_patches_2d(patches=np.asarray(patches),
                                    img_shape=data.shape,
                                    overlapping_factor=self.overlapping_factor)

    def _cost_method(self, data, extra_factor=1.0):
        """Cost
        This method computes the weighted sum of the patch based low-rank
        method

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        float, cost of the patch based nuclear norm
        """
        patches = extract_patches_2d(
            images=data,
            patch_shape=self.patch_shape,
            overlapping_factor=self.overlapping_factor
        )

        costs = Parallel(n_jobs=self.num_cores, prefer="threads")(
            delayed(_svd_cost)(
                        patch=patches[idx],
                        weights=np.copy(self.weights*extra_factor))
            for idx in range(patches.shape[0]))

        return np.sum(np.asarray(costs))
