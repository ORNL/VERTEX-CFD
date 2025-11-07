import numpy as np

# Coefficients
nu = 0.375

# 2-D case
print("\n2-D case:")
grad_vel = np.array([[-0.25, 0.5], [-0.5, 1.0]])

vorticity = np.array([0.0, 0.0])

vorticity[0] = grad_vel[1, 0] - grad_vel[0, 1]
vorticity[1] = 0.0

print("    Vorticity: ", vorticity)

enstrophy = nu * pow(vorticity[0], 2.0)

print("    Enstrophy: ", enstrophy)

divergence = 0.0

for i in range(len(vorticity)):
    divergence += grad_vel[i, i]

print("    Divergence: ", divergence)

# 3-D case
print("\n3-D case:")
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.725, 1.45, -2.175]])

vorticity = np.array([0.0, 0.0, 0.0])

vorticity[0] = grad_vel[2, 1] - grad_vel[1, 2]
vorticity[1] = grad_vel[0, 2] - grad_vel[2, 0]
vorticity[2] = grad_vel[1, 0] - grad_vel[0, 1]

print("    Vorticity: ", vorticity)

enstrophy = 0.0
divergence = 0.0

for i in range(len(vorticity)):
    enstrophy += nu * pow(vorticity[i], 2.0)
    divergence += grad_vel[i, i]

print("    Enstrophy: ", enstrophy)
print("    Divergence: ", divergence)
