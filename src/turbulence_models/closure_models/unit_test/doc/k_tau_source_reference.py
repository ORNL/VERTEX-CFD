import numpy as np
import math

# k-t model constants
beta_star = 0.09
gamma = 0.52
beta_0 = 0.0708
sigma_d = 0.125
sigma_t = 0.5

# Turbulent quantities
nu_t = 0.002
nu = 2e-6
k = 0.0001
t = 0.4

# Turbulent gradients
grad_k_3D = np.array([-0.75, 1.5, -2.25])
grad_t_3D = np.array([-1.25, 2.5, -3.75])

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

P_lim = 10 * beta_star * k / t
print("Limiting value of production term: ", P_lim)

dim_list = [2, 3]

for dim in dim_list:
    print("Computing turbulence quantities in ", dim, "D\n")

    grad_vel = grad_vel_2D

    if (dim == 3):
        grad_vel = grad_vel_3D

    grad_k = grad_k_3D[:dim]
    grad_t = grad_t_3D[:dim]

    P = 0.0
    cross = 0.0
    grad_k_grad_t = 0.0

    # COMMENT: This calculation ignores the divergence of u term
    # which will be required for compressible flows
    for i in range(0, dim):
        grad_k_grad_t += grad_k[i] * grad_t[i]
        for j in range(0, dim):
            S_ij = 0.5 * (grad_vel[i, j] + grad_vel[j, i])
            P += 2.0 * nu_t * pow(S_ij, 2.0)

    if (grad_k_grad_t < 0.0):
        cross = sigma_d * grad_k_grad_t * t

    print("    Unlimited k prod values:\n")

    k_prod = P
    k_dest = beta_star * k / t
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

    grad_tau_sq = np.dot(grad_t, grad_t)
    t_dest_1 = 2.0 * k * sigma_t * grad_tau_sq
    t_dest_2 = 2.0 * nu * grad_tau_sq / t

    t_prod = beta_0 - gamma * t * t * P / nu_t
    t_dest = t_dest_1 + t_dest_2
    t_cross = cross
    t_source = t_prod - t_dest + t_cross

    print("    Unlimited Tau dest values:\n")

    print("        t prod: ", t_prod, "\n")
    print("        t dest: ", t_dest, "\n")
    print("        t cross: ", t_cross, "\n")
    print("        t source: ", t_source, "\n")

    if t_dest_1 > t_dest_2:
        t_dest = t_dest_1 + min(4 / 3 * beta_0, t_dest_2)
    elif t_dest_2 >= t_dest_1:
        t_dest = min(4 / 3 * beta_0, t_dest_1 + t_dest_2)
    t_source = t_prod - t_dest + t_cross

    print("    Limited Tau dest values:\n")

    print("        t prod: ", t_prod, "\n")
    print("        t dest: ", t_dest, "\n")
    print("        t cross: ", t_cross, "\n")
    print("        t source: ", t_source, "\n")
