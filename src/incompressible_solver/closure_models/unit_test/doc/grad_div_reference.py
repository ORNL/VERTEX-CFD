import numpy as np

# Coefficients
gamma = 0.5

# Velocity gradients and momentum flux
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.125, 0.25, -0.375]])

dims = [2, 3]
for dim in dims:
    momentum_flux = 2 * grad_vel
    divU = np.sum(np.diag(grad_vel[0:dim, 0:dim]))
    momentum_flux[0:dim, 0:dim] += np.identity(dim) * gamma * divU
    print(f"{dim}D Momentum Flux: ")
    print(momentum_flux[0:dim, 0:dim])
