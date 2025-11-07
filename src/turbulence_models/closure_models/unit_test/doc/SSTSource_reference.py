import numpy as np
import math

# Turbulent quantities
k = 2.5
omegas = [0.1, 0.1, 10]
d = 3.0
limits = [True, False, True]
eddyvs = [0.25, 0.25, 0.25]
eddyvs3 = [0.25, 0.25, 0.25]
nu = 1.0e-5
rho = 1.2

# Menter SST model constants
sigma_w1 = 0.4  # standard value 0.5
sigma_w2 = 0.9  # standard value 0.856
beta_1 = 0.07  # standard value 0.075
beta_2 = 0.08  # standard value 0.0828
beta_star = 0.1  # standard value 0.09
kappa = 0.4  # standard value 0.41

gamma_1 = beta_1 / beta_star - sigma_w1 * kappa * kappa / math.sqrt(beta_star)
gamma_2 = beta_2 / beta_star - sigma_w2 * kappa * kappa / math.sqrt(beta_star)

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

    nu_ts = eddyvs
    if (dim == 3):
        grad_vel = grad_vel_3D
        nu_ts = eddyvs3

    grad_k = grad_k_3D[:dim]
    grad_w = grad_w_3D[:dim]

    # COMMENT: This calculation ignores the divergence of u term
    # which will be required for compressible flows
    S2 = 0.0
    grad_k_grad_w = 0.0
    for i in range(0, dim):
        grad_k_grad_w += grad_k[i] * grad_w[i]
        for j in range(0, dim):
            S2 += pow(0.5 * (grad_vel[i, j] + grad_vel[j, i]), 2.0)

    print("    S2 = ", S2, "\n")

    for omega, limit, nu_t in zip(omegas, limits, nu_ts):
        print("        Omega = ", omega, {
            True: '(with production limiter)',
            False: ''
        }[limit])

        d2 = d * d

        term1 = math.sqrt(k) / (beta_star * omega * d)
        term2 = 500.0 * nu / (d2 * omega)

        # Source term calculation, k equation
        Pk0 = 2 * nu_t * S2
        Dk = -beta_star * omega * k

        limited = False
        Pk = Pk0
        if limit:
            if Pk0 > -20 * Dk:
                limited = True
                Pk = -20 * Dk

        # Source term calculation, omega equation
        CD_kw = max(2.0 * rho * sigma_w2 * grad_k_grad_w / omega, 1.0e-20)
        term3 = 4.0 * rho * sigma_w2 * k / (CD_kw * d2)
        arg_1 = min(max(term1, term2), term3)
        F_1 = math.tanh(arg_1**4)

        gamma = gamma_1 * F_1 + gamma_2 * (1.0 - F_1)
        beta = beta_1 * F_1 + beta_2 * (1.0 - F_1)

        Pw = gamma * Pk0 / nu_t
        Dw = -beta * omega * omega
        Cw = 2.0 * (1.0 - F_1) * sigma_w2 * grad_k_grad_w / omega

        if limited:
            print("          Pk = ", Pk, "(limited)")
            print("               ", Pk0, "(original)")
        else:
            print("          Pk = ", Pk)
        print("          Dk = ", Dk)
        print("          Sk = ", Pk + Dk)
        print("          Pw = ", Pw)
        print("          Dw = ", Dw)
        print("          Cw = ", Cw)
        print("          Sw = ", Pw + Dw + Cw)
        print("        nu_t = ", nu_t)
        print("  blending f = ", F_1, "\n")
