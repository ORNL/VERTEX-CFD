# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:52:57 2024

@author: ffn
"""
import numpy as np
import math

# NOTE:
# Python script to compute the expected values for the artificial compression term
# of the LSVOF model


def interface_flux(Calpha, alpha, grad_alpha, magU, magGradalpha,
                   num_space_dim):

    fv_interface_flux = Calpha * alpha * (1.0 - alpha) * magU * grad_alpha / (
        magGradalpha)

    return fv_interface_flux[0:num_space_dim]


# Coefficients
Calpha = 20.0
rho = 1

# Alpha, velocity vector and alpha gradients
alpha = 0.3
velocity = np.array([0.25, 0.5, 0.125])
grad_alpha = np.array([-0.75, 1.5, -2.25])

# Create expected values for every mesh dimension
dims = [2, 3]
for dim in dims:
    print("\n" + str(dim) + "-D case:")

    if dim == 2:
        magU = math.sqrt(pow(velocity[0], 2) + pow(velocity[1], 2))
        magGradalpha = math.sqrt(pow(grad_alpha[0], 2) + pow(grad_alpha[1], 2))

    else:
        magU = math.sqrt(
            pow(velocity[0], 2) + pow(velocity[1], 2) + pow(velocity[2], 2))
        magGradalpha = math.sqrt(
            pow(grad_alpha[0], 2) + pow(grad_alpha[1], 2) +
            pow(grad_alpha[2], 2))

    fv_interface = interface_flux(Calpha, alpha, grad_alpha, magU,
                                  magGradalpha, dim)

    print("fv_interface for scalar interface flux:", fv_interface)
    print("magnitude of U:", magU)
    print("magnitude of grad_alpha:", magGradalpha)
