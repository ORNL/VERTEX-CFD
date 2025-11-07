import numpy as np
import math

# Turbulent quantities
k = 2.5
omegas = [1, 10]

# Wilcox k-w model constants
beta_star = 0.1  # Standard value is 0.09
C_lim = 0.88  # Standard value is 0.875 = 7.0 / 8.0

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

dim_list = [2, 3]

for dim in dim_list:
    print("Computing turbulence quantities in ", dim, "D\n")

    grad_vel = grad_vel_2D

    if (dim == 3):
        grad_vel = grad_vel_3D

    S2 = 0.0

    # COMMENT: This calculation ignores the divergence of u term
    # which will be required for compressible flows
    for i in range(0, dim):
        for j in range(0, dim):
            S2 += pow(0.5 * (grad_vel[i, j] + grad_vel[j, i]), 2.0)

    print("    S2 = ", S2, "\n")

    omega_lim = C_lim * math.sqrt(2.0 * S2 / beta_star)

    print("    Limiting value of omega: ", omega_lim)

    for omega in omegas:
        print("        Omega = ", omega)

        omega_tilda = max(omega, omega_lim)

        nu_t = k / omega_tilda

        print("        nu_t = ", nu_t, "\n")
