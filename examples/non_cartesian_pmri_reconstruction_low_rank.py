"""
Neuroimaging non-cartesian reconstruction
=========================================

Author: LElgueddari

In this tutorial we will reconstruct an MRI image from non-cartesian kspace
measurements.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquisition cartesian scheme.
"""

# Third party import
from modopt.math.metrics import ssim
import numpy as np

# Package import
from mri.operators import NonCartesianFFT
from mri.operators.utils import convert_locations_to_mask, \
    gridded_inverse_fourier_transform_nd
from mri.reconstructors import LowRankCalibrationlessReconstructor
import pysap
from pysap.data import get_sample_data

# Loading input data
image = get_sample_data('2d-pmri')
Il = image.data
image = pysap.Image(data=np.sqrt(np.sum(np.abs(image.data)**2, axis=0)))


# Obtain MRI non-cartesian mask
radial_mask = get_sample_data("mri-radial-samples")
kspace_loc = radial_mask.data
mask = pysap.Image(data=convert_locations_to_mask(kspace_loc, image.shape))

# View Input
# image.show()
# mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquisition mask, we retrospectively
# undersample the k-space using a radial acquisition mask
# We then reconstruct the zero order solution as a baseline

# Get the locations of the kspace samples and the associated observations
fourier_op = NonCartesianFFT(samples=kspace_loc, shape=image.shape,
                             implementation='cpu', n_coils=Il.shape[0])
kspace_obs = fourier_op.op(Il)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the reconstructor
reconstructor = LowRankCalibrationlessReconstructor(
    kspace_data=kspace_obs,
    kspace_loc=kspace_loc,
    uniform_data_shape=image.shape,
    n_coils=Il.shape[0],
    patch_shape=[64, 64],
    overlapping_factor=2,
    mu=1e-5,
    fourier_type='non-cartesian',
    nfft_implementation='cpu',
    lips_calc_max_iter=10,
    num_check_lips=10,
    lipschitz_cst=None,
    n_jobs=-1,
    verbose=0)

# Start Reconstruction
x_final, costs, metrics = reconstructor.reconstruct(num_iterations=200)
image_rec = pysap.Image(data=np.sqrt(np.sum(np.abs(x_final)**2, axis=0)))
image_rec.show()
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
