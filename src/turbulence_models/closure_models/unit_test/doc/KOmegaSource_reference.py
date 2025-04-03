import numpy as np
import math

# Wilcox k-w model constants
beta_star = 0.09
gamma = 0.52
beta_0 = 0.0708
sigma_d = 0.125

# Turbulent quantities
nu_t = 1.1
k = 0.1
w = 3.3

# Turbulent gradients
grad_k_3D = np.array([-0.75, 1.5, -2.25])
grad_w_3D = np.array([-1.25, 2.5, -3.75])

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

P_lim = 20 * beta_star * w * k
print("Limiting value of production term: ", P_lim)

dim_list = [2, 3]

for dim in dim_list:
    print("Computing turbulence quantities in ", dim, "D\n")

    grad_vel = grad_vel_2D

    if (dim == 3):
        grad_vel = grad_vel_3D

    grad_k = grad_k_3D[:dim]
    grad_w = grad_w_3D[:dim]

    P = 0.0
    cross = 0.0
    grad_k_grad_w = 0.0
    chi_w = 0.0

    # COMMENT: This calculation ignores the divergence of u term
    # which will be required for compressible flows
    for i in range(0, dim):
        grad_k_grad_w += grad_k[i] * grad_w[i]
        for j in range(0, dim):
            S_ij = 0.5 * (grad_vel[i, j] + grad_vel[j, i])
            P += 2.0 * nu_t * pow(S_ij, 2.0)
            for l in range(0, dim):
                chi_w += ((0.5 * (grad_vel[i, j] - grad_vel[j, i])) *
                          (0.5 * (grad_vel[j, l] - grad_vel[l, j])) *
                          (0.5 * (grad_vel[l, i] + grad_vel[i, l])))

    if (grad_k_grad_w > 0.0):
        cross = sigma_d * grad_k_grad_w / w

    chi_w = abs(chi_w / pow(beta_star * w, 3.0))

    print("    chi_w = ", chi_w, "\n")

    f_b = (1.0 + 85.0 * chi_w) / (1.0 + 100.0 * chi_w)
    beta = beta_0 * f_b

    print("    Unlimited k prod values:\n")

    k_prod = P
    k_dest = beta_star * w * k
    k_source = k_prod - k_dest

    print("        k prod: ", k_prod, "\n")
    print("        k dest: ", k_dest, "\n")
    print("        k source: ", k_source, "\n")

    print("    Limited k prod values:\n")

    k_prod = min(P, P_lim)
    k_source = k_prod - k_dest

    print("        k prod: ", k_prod, "\n")
    print("        k dest: ", k_dest, "\n")
    print("        k source: ", k_source, "\n")

    w_prod = gamma * w / k * P
    w_dest = beta * w * w
    w_cross = cross
    w_source = w_prod - w_dest + w_cross

    print("    Omega values (same for both cases):\n")

    print("        w prod: ", w_prod, "\n")
    print("        w dest: ", w_dest, "\n")
    print("        w cross: ", w_cross, "\n")
    print("        w source: ", w_source, "\n")
