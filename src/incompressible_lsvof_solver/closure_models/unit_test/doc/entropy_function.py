# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:52:57 2024

@author: ffn
"""
import numpy as np
import math

# NOTE:
# Python script to compute the expected values for the
# entropy term and entropy residual term of the LSVOF model

# grad alpha and velocity vector
velocity = np.array([0.25, 0.5, 0.125])
grad_alpha = np.array([0.75, 1.5, 2.25])

# alpha and dalpha_dt
alpha = 0.3
dadt = 0.01

# Create expected values for every mesh dimension
dims = [2, 3]
entropy_types = ['quadratic', 'log', 'biquadratic']
for dim in dims:
    for entropy_type in entropy_types:
        print("\n" + str(dim) + "-D case:")
        print("Entropy Type is", entropy_type)

        if dim == 2:
            entr_res = velocity[0] * grad_alpha[0] + velocity[1] * grad_alpha[1]
        else:
            entr_res = velocity[0] * grad_alpha[0] + velocity[1] * grad_alpha[
                1] + velocity[2] * grad_alpha[2]

        if entropy_type == 'quadratic':
            entr_res += dadt
            entr_res *= alpha
            entr_func = 0.5 * (alpha**2)
        elif entropy_type == 'log':
            entr_res *= math.log(alpha) - math.log(1.0 - alpha)
            entr_res += math.log(alpha / (1.0 - alpha)) * dadt
            entr_func = (alpha * math.log(alpha)) + (
                (1.0 - alpha) * math.log(1.0 - alpha))
        elif entropy_type == 'biquadratic':
            entr_res *= (2.0 * alpha) - (4.0 * (alpha**3))
            entr_res += 2.0 * alpha * (1.0 - (2.0 * (alpha**2))) * dadt
            entr_func = (alpha**2) - (alpha**4)

        print("Entropy term:", entr_func)
        print("Entropy residual term:", entr_res)
