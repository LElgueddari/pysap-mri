# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2020                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

# Third party import
import numpy as np
from modopt.opt.proximity import ProximityParent
from joblib import Parallel, delayed


class GroupLASSO(ProximityParent):
    """This class implements the proximity operator of the group-lasso
    regularization, with groups dimension being the first dimension

    Attributes:
    ----------
    weights : np.ndarray
        Input array of weights
    """
    def __init__(self, weights):
        self.weights = weights

    def op(self, data, extra_factor=1.0):
        """ Operator
        This method returns the input data thresholded by the weights
        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor
        Returns
        -------
        DictionaryBase thresholded data
        """
        norm_2 = np.linalg.norm(data, axis=0)
        return data * np.maximum(0, 1.0 - self.weights*extra_factor /
                                 np.maximum(norm_2, np.finfo(np.float32).eps))

    def cost(self, data):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        return np.sum(self.weights * np.linalg.norm(data, axis=0))
