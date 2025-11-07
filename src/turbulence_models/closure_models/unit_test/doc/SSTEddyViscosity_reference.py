import numpy as np
import math

# Turbulent quantities
k = 2.5
omegas = [0.1, 10]
d = 3.0
nu = 1.0e-5
rho = 1.2

# Menter SST model constants
a_1 = 0.41
sigma_w2 = 0.856
beta_star = 0.09

# Turbulent gradients
grad_k_3D = np.array([-0.75, 1.5, -2.25])
grad_w_3D = np.array([-1.25, 2.5, -3.75])

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

    grad_k = grad_k_3D[:dim]
    grad_w = grad_w_3D[:dim]

    Wm = 0.0

    # Compute the vorticity magnitude
    for i in range(0, dim):
        for j in range(0, dim):
            Wm += 0.5 * (grad_vel[i, j] - grad_vel[j, i])**2
    Wm = math.sqrt(Wm)

    print("    Wm = ", Wm, "\n")

    # COMMENT: This calculation ignores the divergence of u term
    # which will be required for compressible flows
    S2 = 0.0
    grad_k_grad_w = 0.0
    for i in range(0, dim):
        grad_k_grad_w += grad_k[i] * grad_w[i]
        for j in range(0, dim):
            S2 += pow(0.5 * (grad_vel[i, j] + grad_vel[j, i]), 2.0)

    print("    S2 = ", S2, "\n")

    for omega in omegas:
        print("        Omega = ", omega)

        # Do the non-boundary calculation
        d2 = d * d

        term1 = math.sqrt(k) / (beta_star * omega * d)
        term2 = 500.0 * nu / (d2 * omega)

        # Eddy visocosity calculation
        arg_2 = max(2 * term1, term2)
        F_2 = math.tanh(arg_2**2)

        nu_t = a_1 * k / max(a_1 * omega, Wm * F_2)

        # Blending calculation
        CD_kw = max(2.0 * rho * sigma_w2 * grad_k_grad_w / omega, 1.0e-20)
        term3 = 4.0 * rho * sigma_w2 * k / (CD_kw * d2)
        arg_1 = min(max(term1, term2), term3)
        F_1 = math.tanh(arg_1**4)

        print("        nu_t = ", nu_t)
        print("  blending f = ", F_1, "\n")

        # The boundary calculation
        F_1 = 1.0
        F_2 = 1.0
        nu_t = a_1 * k / max(a_1 * omega, Wm * F_2)
        print("        nu_t (boundary) = ", nu_t)
        print("  blending f (boundary) = ", F_1, "\n")
