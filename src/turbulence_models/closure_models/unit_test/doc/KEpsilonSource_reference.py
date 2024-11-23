import numpy as np
import math

# Turbulent quantities
nu_t = 3.0
k = 4.0
e = 5.0

# KEpsilon model constants
C_1 = 1.44
C_2 = 1.92

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

dims = [2, 3]

for dim in dims:
    print("Computing turbulence quantities in ", dim, "D\n")

    grad_vel = grad_vel_2D

    if (dim == 3):
        grad_vel = grad_vel_3D

    Sij_Sij = 0.0

    for i in range(0, dim):
        for j in range(0, dim):
            Sij_Sij += pow(0.5 * (grad_vel[i, j] + grad_vel[j, i]), 2.0)

    print("    Sij_Sij = ", Sij_Sij, "\n")

    k_prod = 2.0 * nu_t * Sij_Sij
    k_dest = -e
    k_source = k_prod + k_dest

    print("    k prod: ", k_prod)
    print("    k dest: ", k_dest)
    print("    k source: ", k_source, "\n")

    e_prod = C_1 * e / k * k_prod
    e_dest = -C_2 * e * e / k
    e_source = e_prod + e_dest

    print("    e prod: ", e_prod)
    print("    e dest: ", e_dest)
    print("    e source: ", e_source, "\n")
