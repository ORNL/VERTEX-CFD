import numpy as np

# Dependent values
u = np.array([-1.5, 2.0, -0.5])
h = np.array([0.25, 0.5, 0.75])
bateman = np.array([[-1.0, 0.0], [+1.0, -0.1]])
mic_cross_section = np.array([[-2.0, 1.0], [+2.0, -0.2]])
neutron_flux = 0.25
nu = 0.0576

# Bateman Case
dt = 1 / np.max(abs(np.diag(bateman)))
print("Bateman dt: ", dt)

# Transmutation Case
dt = 1 / np.max(abs(np.diag(mic_cross_section) * neutron_flux))
print("Transmutation dt: ", dt)

# Advection
eig_2D = np.sum(abs(u[:2]) / h[:2])
eig_3D = np.sum(abs(u) / h)
# 2-D case
dt = 1.0 / eig_2D
print("Advection 2-D dt: ", dt)

# 3-D case
dt = 1.0 / eig_3D
print("Advection 3-D dt: ", dt)

# Bateman Advection
# 2-D case
dt = np.minimum(1 / np.max(abs(np.diag(bateman))), 1.0 / eig_2D)
print("Bateman Advection 2-D dt: ", dt)

# 3-D case
dt = np.minimum(1 / np.max(abs(np.diag(bateman))), 1.0 / eig_3D)
print("Bateman Advection 3-D dt: ", dt)

# Diffusion
eig_dif_2D = np.sum(nu / h[:2] / h[:2])
eig_dif_3D = np.sum(nu / h / h)
# 2-D case
print("Diffusion 2-D dt: ", 1.0 / eig_dif_2D)

# 3-D case
print("Diffusion 3-D dt: ", 1.0 / eig_dif_3D)

# Bateman Advection Diffusion
# 2-D case
dt = np.min(
    [1 / np.max(abs(np.diag(bateman))), 1.0 / eig_2D, 1.0 / eig_dif_2D])
print("Bateman Advection Diffusion 2-D dt: ", dt)

# 3-D case
dt = np.min(
    [1 / np.max(abs(np.diag(bateman))), 1.0 / eig_3D, 1.0 / eig_dif_3D])
print("Bateman Advection Diffusion 3-D dt: ", dt)
