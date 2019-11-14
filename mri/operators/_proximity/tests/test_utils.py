# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np
import unittest

# Package import
from mri.operators._proximity.utils import extract_patches_2d, \
    recombine_patches_2d, _svd_thresh, _svd_cost


class TestProximityUtils(unittest.TestCase):
    """ Test the utils methods used by the reconstructors.
    """

    def setUp(self):
        """ Generate fake data.
        """
        self.p_mri_data = np.arange(18).reshape(2, 3, 3)
        self.patches = np.array([[[[0, 1], [3, 4]], [[9, 10], [12, 13]]],
                                 [[[1, 2], [4, 5]], [[10, 11], [13, 14]]],
                                 [[[3, 4], [6, 7]], [[12, 13], [15, 16]]],
                                 [[[4, 5], [7, 8]], [[13, 14], [16, 17]]]])

    def test_patch_extraction(self):
        """ Test the extraction of the tests.
        """
        np.testing.assert_array_equal(extract_patches_2d(
            images=self.p_mri_data,
            patch_shape=(2, 2),
            overlapping_factor=2),
                                      self.patches,
                                      err_msg="Patches extraction fails")
        np.testing.assert_raises(ValueError,
                                 extract_patches_2d,
                                 self.p_mri_data,
                                 (4, 2))

        np.testing.assert_raises(ValueError,
                                 extract_patches_2d,
                                 self.p_mri_data,
                                 (2, 4))

    def test_patch_reconstruction(self):
        """ Test the extraction of the tests.
        """
        np.testing.assert_array_equal(recombine_patches_2d(
            patches=self.patches,
            img_shape=self.p_mri_data.shape,
            overlapping_factor=2),
                                      self.p_mri_data,
                                      err_msg="Patches extraction fails")

    def test_svd_threshold(self):
        """ Test the extraction of the tests.
        """
        np.testing.assert_almost_equal(_svd_thresh(
            patch=self.patches[0],
            threshold=0.0,
            threshold_type='soft'),
                                      self.patches[0],
                                      err_msg="SVD threshold fails")

    def test_svd_cost(self):
        """ Test the extraction of the tests.
        """
        np.testing.assert_almost_equal(_svd_cost(
            patch=self.patches[0],
            weights=1.0),
                                      25.176218853633724,
                                      err_msg="SVD cost fails")
