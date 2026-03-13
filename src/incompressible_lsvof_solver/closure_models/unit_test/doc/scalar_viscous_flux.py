# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:43:26 2025

@author: ffn
"""

import numpy as np
import math

# NOTE:
# Python script to compute the expected values for the
# viscous flux of the LSVOF eqn

# grad alpha and entropy value
entr_visc = 0.1
grad_alpha_3d = np.array([0.75, 1.5, 2.25])
grad_alpha_2d = np.array([0.75, 1.5])

# Create expected values for every mesh dimension
dims = [2, 3]
for dim in dims:
    print("\n" + str(dim) + "-D case:")

    if dim == 2:
        grad_alpha = grad_alpha_2d
    else:
        grad_alpha = grad_alpha_3d

    fv_visc_flux = entr_visc * grad_alpha

    print("Viscous flux of VOF eqn:", fv_visc_flux)
