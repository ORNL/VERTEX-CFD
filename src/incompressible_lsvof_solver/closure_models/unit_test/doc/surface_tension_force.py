# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:58:39 2025

@author: ffn
"""
import numpy as np
import math

# NOTE:
# Python script to compute the expected values for the
# surface tension force term of the LSVOF model


def surftension_flux(sigma, identity, scalar_normal, mag_grad_alpha,
                     num_space_dim):

    fv_surftension_flux = sigma * mag_grad_alpha * (
        identity - np.outer(scalar_normal, scalar_normal))

    return fv_surftension_flux[0:num_space_dim]


# Coefficients
sigma = 0.7

# Number of phases
nphases = 3  # 2 gas phases + 1 liquid

# Identity tensors and alpha gradients for surface tension flux
grad_alpha_2d = np.array([[-0.75, 1.5], [-0.25, 0.5]])
grad_alpha_3d = np.array([[-0.75, 1.5, -2.25], [-0.25, 0.5, -0.75]])
identity_2d = np.array([[1.0, 0.0], [0.0, 1.0]])
identity_3d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Create expected values for every mesh dimension
dims = [2, 3]
for dim in dims:
    print("\n" + str(dim) + "-D case:")
    fv_Fs = 0.0

    for n in range(nphases - 1):
        if dim == 2:
            grad_alpha = grad_alpha_2d[n]
            identity = identity_2d
            mag_grad_alpha = math.sqrt(
                pow(grad_alpha[0], 2) + pow(grad_alpha[1], 2))
        else:
            grad_alpha = grad_alpha_3d[n]
            identity = identity_3d
            mag_grad_alpha = math.sqrt(
                pow(grad_alpha[0], 2) + pow(grad_alpha[1], 2) +
                pow(grad_alpha[2], 2))

        scalar_normal = grad_alpha / mag_grad_alpha

        fv_Fs += surftension_flux(sigma, identity, scalar_normal,
                                  mag_grad_alpha, dim)

    print("fv_Fs for surface tension force:", fv_Fs)
