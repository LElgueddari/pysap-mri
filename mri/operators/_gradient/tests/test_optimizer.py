# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np
from scipy.fftpack import fftshift
import unittest


# Package import
from mri.reconstruct.fourier import FFT, NFFT
from mri.numerics.reconstruct import sparse_rec_fista, sparse_rec_condatvu,\
    sparse_rec_pogm
from mri.numerics.utils import convert_mask_to_locations
from mri.numerics.utils import generate_operators
from pysap.data import get_sample_data


class TestOptimizer(unittest.TestCase):
    """ Test the FISTA's gradient descent.
    """
    def setUp(self):
        """ Get the data from the server.
        """
        self.images = [get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [im.data.shape for im in self.images]))
        self.mask = get_sample_data("mri-mask").data
        # Test a wide variety of linear operators :
        # From WaveletN
        self.decimated_wavelets = ['sym8']
        # From WaveletUD2, tested only for analysis formulation
        self.undecimated_wavelets = [24]
        print("[info] Found {0} transformations.".
              format(len(self.decimated_wavelets)))
        self.nb_scales = [4]
        self.nb_iter = 100

    def test_reconstruction_fista_fft2(self):
        """ Test all the registered transformations.
        """
        print("Process test FFT FISTA::")
        for image in self.images:
            fourier = FFT(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                          shape=image.shape)

            data = fourier.op(image.data)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for name in self.decimated_wavelets:
                    print("    Transform: {0}".format(name))
                    gradient_op, linear_op, prox_op, cost_op = \
                        generate_operators(
                            data=data,
                            wavelet_name=name,
                            samples=convert_mask_to_locations(
                                fftshift(self.mask)),
                            mu=0,
                            nb_scales=4,
                            fourier_type='cartesian',
                            uniform_data_shape=image.shape,
                            gradient_space="synthesis")
                    x_final, costs, _ = sparse_rec_fista(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_op=prox_op,
                        cost_op=cost_op,
                        lambda_init=1.0,
                        max_nb_of_iter=self.nb_iter,
                        verbose=0)
                    fourier_0 = FFT(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                                    shape=image.shape)
                    data_0 = fourier_0.op(np.fft.fftshift(image.data))
                    np.testing.assert_allclose(x_final, np.fft.ifftshift(
                        fourier_0.adj_op(data_0)))

    def test_reconstruction_pogm_fft2(self):
        """ Test all the registered transformations.
        """
        print("Process test FFT POGM::")
        for image in self.images:
            fourier = FFT(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                          shape=image.shape)

            data = fourier.op(image.data)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for name in self.decimated_wavelets:
                    print("    Transform: {0}".format(name))
                    gradient_op, linear_op, prox_op, cost_op = \
                        generate_operators(
                            data=data,
                            wavelet_name=name,
                            samples=convert_mask_to_locations(
                                fftshift(self.mask)),
                            mu=0,
                            nb_scales=4,
                            fourier_type='cartesian',
                            uniform_data_shape=image.shape,
                            gradient_space="synthesis")
                    x_final, costs, _ = sparse_rec_pogm(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_op=prox_op,
                        cost_op=cost_op,
                        max_nb_of_iter=self.nb_iter,
                        verbose=0)
                    fourier_0 = FFT(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                                    shape=image.shape)
                    data_0 = fourier_0.op(np.fft.fftshift(image.data))
                    np.testing.assert_allclose(x_final, np.fft.ifftshift(
                        fourier_0.adj_op(data_0)))

    def test_reconstruction_condat_vu_fft2(self):
        """ Test all the registered transformations.
        """
        print("Process test FFT Condat Vu algorithm::")
        for image in self.images:
            fourier = FFT(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                          shape=image.shape)
            data = fourier.op(image.data)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                wavelets = np.append(self.decimated_wavelets,
                                     self.undecimated_wavelets)
                for name in wavelets:
                    print("    Transform: {0}".format(name))
                    gradient_op, linear_op, prox_dual_op, cost_op = \
                        generate_operators(
                            data=data,
                            wavelet_name=name,
                            samples=convert_mask_to_locations(
                                fftshift(self.mask)),
                            mu=0,
                            nb_scales=4,
                            fourier_type='cartesian',
                            uniform_data_shape=image.shape,
                            gradient_space="analysis")
                    x_final, costs, _, _ = sparse_rec_condatvu(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_dual_op=prox_dual_op,
                        cost_op=cost_op,
                        std_est=0.0,
                        std_est_method="dual",
                        std_thr=0,
                        tau=None,
                        sigma=None,
                        relaxation_factor=1.0,
                        nb_of_reweights=0,
                        max_nb_of_iter=self.nb_iter,
                        add_positivity=False,
                        atol=1e-4,
                        verbose=0)
                    fourier_0 = FFT(samples=convert_mask_to_locations(
                                            fftshift(self.mask)),
                                    shape=image.shape)
                    data_0 = fourier_0.op(np.fft.fftshift(image.data))
                    np.testing.assert_allclose(x_final, np.fft.ifftshift(
                        fourier_0.adj_op(data_0)))

    def test_reconstruction_fista_nfft2(self):
        """ Test all the registered transformations.
        """
        print("Process test NFFT FISTA::")
        for image in self.images:
            fourier = NFFT(samples=convert_mask_to_locations(
                                            self.mask),
                           shape=image.shape)
            data = fourier.op(image.data)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for name in self.decimated_wavelets:
                    print("    Transform: {0}".format(name))
                    gradient_op, linear_op, prox_op, cost_op = \
                        generate_operators(
                            data=data,
                            wavelet_name=name,
                            samples=convert_mask_to_locations(self.mask),
                            mu=0,
                            nb_scales=4,
                            fourier_type='non-cartesian',
                            uniform_data_shape=image.shape,
                            gradient_space="synthesis")
                    x_final, costs, _ = sparse_rec_fista(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_op=prox_op,
                        cost_op=cost_op,
                        lambda_init=1.0,
                        max_nb_of_iter=self.nb_iter,
                        verbose=0)
                    fourier_0 = FFT(samples=convert_mask_to_locations(
                                        self.mask),
                                    shape=image.shape)
                    data_0 = fourier_0.op(image.data)
                    np.testing.assert_allclose(x_final,
                                               fourier_0.adj_op(data_0))

    def test_reconstruction_pogm_nfft2(self):
        """ Test reconstruction with POGM.
        """
        print("Process test NFFT POGM::")
        for image in self.images:
            fourier = NFFT(samples=convert_mask_to_locations(
                                            self.mask),
                           shape=image.shape)
            data = fourier.op(image.data)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                for name in self.decimated_wavelets:
                    print("    Transform: {0}".format(name))
                    gradient_op, linear_op, prox_op, cost_op = \
                        generate_operators(
                            data=data,
                            wavelet_name=name,
                            samples=convert_mask_to_locations(self.mask),
                            mu=0,
                            nb_scales=4,
                            fourier_type='non-cartesian',
                            uniform_data_shape=image.shape,
                            gradient_space="synthesis")
                    x_final, costs, _ = sparse_rec_pogm(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_op=prox_op,
                        cost_op=cost_op,
                        max_nb_of_iter=self.nb_iter,
                        verbose=0)
                    fourier_0 = FFT(samples=convert_mask_to_locations(
                                            self.mask),
                                    shape=image.shape)
                    data_0 = fourier_0.op(image.data)
                    np.testing.assert_allclose(x_final,
                                               fourier_0.adj_op(data_0))

    def test_reconstruction_condat_vu_nfft2(self):
        """ Test all the registered transformations.
        """
        print("Process test NFFT Condat Vu algorithm::")
        for image in self.images:
            fourier = NFFT(samples=convert_mask_to_locations(
                                            self.mask),
                           shape=image.shape)
            data = fourier.op(image.data)
            print("Process test with image '{0}'...".format(
                image.metadata["path"]))
            for nb_scale in self.nb_scales:
                print("- Number of scales: {0}".format(nb_scale))
                wavelets = np.append(self.decimated_wavelets,
                                     self.undecimated_wavelets)
                for name in wavelets:
                    print("    Transform: {0}".format(name))
                    gradient_op, linear_op, prox_dual_op, cost_op = \
                        generate_operators(
                            data=data,
                            wavelet_name=name,
                            samples=convert_mask_to_locations(self.mask),
                            mu=0,
                            nb_scales=4,
                            fourier_type='non-cartesian',
                            uniform_data_shape=image.shape,
                            gradient_space="analysis")
                    x_final, costs, _, _ = sparse_rec_condatvu(
                        gradient_op=gradient_op,
                        linear_op=linear_op,
                        prox_dual_op=prox_dual_op,
                        cost_op=cost_op,
                        std_est=0.0,
                        std_est_method="dual",
                        std_thr=0,
                        tau=None,
                        sigma=None,
                        relaxation_factor=1.0,
                        nb_of_reweights=0,
                        max_nb_of_iter=self.nb_iter,
                        add_positivity=False,
                        atol=1e-4,
                        verbose=0)
                    fourier_0 = FFT(samples=convert_mask_to_locations(
                                            self.mask),
                                    shape=image.shape)
                    data_0 = fourier_0.op(image.data)
                    np.testing.assert_allclose(x_final,
                                               fourier_0.adj_op(data_0))


if __name__ == "__main__":
    unittest.main()