import numpy as np
import math

# Turbulent quantities
nu_t = 3.0
k = 4.0
e = 5.0

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

# Flow and model constants
nu = 2.0
C_nu = 0.09
C_1 = 1.35
C_2 = 1.8
f1 = 1.0

# Flow variables
distance = 0.5
surface_area = 10000.0
tau = 4 / surface_area

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
    k_dest = -e - 2 * nu * k / distance**2
    k_source = k_prod + k_dest

    print("    k prod: ", k_prod)
    print("    k dest: ", k_dest)
    print("    k source: ", k_source, "\n")

    e_prod = C_1 * e / k * k_prod * f1
    gamma = e / k
    f2 = 1.0 - 0.22 * np.exp(-(k**2 / 6 / nu / e)**2.0)
    beta = C_2 * f2 * gamma + 2 * nu / distance**2 * np.exp(
        -0.5 * np.abs(tau) * distance / nu)
    e_dest = -beta * e
    e_source = e_prod + e_dest

    print("    e prod: ", e_prod)
    print("    e dest: ", e_dest)
    print("    e source: ", e_source, "\n")
