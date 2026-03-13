# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 2025

@author: ffn
"""
import numpy as np
import math

# NOTE:
# Python script to compute the expected values for the
# entropy viscosity terms of the LSVOF model


def max_entropy(cmax, magU, min_length):

    fv_max_entropy = cmax * min_length * magU

    return fv_max_entropy


def local_entropy(ce, min_length, resid_norm, entr_norm):

    fv_entropy = ce * (min_length**2) * resid_norm / entr_norm

    return fv_entropy


# Coefficients
cmax = 0.5
ce = 0.5

# grad alpha, velocity vector, element size and entropy values
resid_norm = 0.1
velocity = np.array([0.25, 0.5, 0.125])
element_size = np.array([1.0, 0.1, 0.01])
entr_norm = 0.1

# Create expected values for every mesh dimension
dims = [2, 3]
for dim in dims:
    print("\n" + str(dim) + "-D case:")

    if dim == 2:
        magU = math.sqrt(pow(velocity[0], 2) + pow(velocity[1], 2))
        min_length = min(element_size[0], element_size[1])
    else:
        magU = math.sqrt(
            pow(velocity[0], 2) + pow(velocity[1], 2) + pow(velocity[2], 2))
        min_length = np.min(element_size)

    fv_ceiling_entr = max_entropy(cmax, magU, min_length)

    fv_local_entr = local_entropy(ce, min_length, resid_norm, entr_norm)

    fv_entropy_visc = np.minimum(fv_ceiling_entr, fv_local_entr)

    print("Maximum entropy viscosity:", fv_ceiling_entr)
    print("Local entropy viscosity:", fv_local_entr)
    print("Entropy viscosity:", fv_entropy_visc)
