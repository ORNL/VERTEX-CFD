import numpy as np

rho = 1.0
Cp = 5.0
u = 0.25
v = 0.5
w = 0.125
p = 0.75  # lagrange pressure
T = u + v
grad_temp = np.array([-0.75, 1.5, -2.25])
beta = 2

print("\nFlux with scaled density:\n")
fc_continuity_AC = rho * np.array([u, v, w])
fc_continuity_EDAC = fc_continuity_AC + p * np.array([u, v, w]) / beta
print(fc_continuity_AC)
print(fc_continuity_EDAC)

fc_momentum_0 = rho * np.array([u * u + p, u * v, u * w])
print(fc_momentum_0)

fc_momentum_1 = rho * np.array([u * v, v * v + p, v * w])
print(fc_momentum_1)

fc_momentum_2 = rho * np.array([u * w, v * w, w * w + p])
print(fc_momentum_2)

fc_energy = rho * Cp * np.array([u * T, v * T, w * T])
print(fc_energy)

fc_source = rho * Cp * grad_temp * np.array([u, v, w])
print("source term in 2D", np.sum(fc_source[0:-1]))
print("source term in 3D", np.sum(fc_source))

print("\nFlux with unscaled density:\n")
rho = 3.0
fc_continuity_AC = rho * np.array([u, v, w])
fc_continuity_EDAC = fc_continuity_AC + p * np.array([u, v, w]) / beta
print(fc_continuity_AC)
print(fc_continuity_EDAC)

fc_momentum_0 = np.array([rho * u * u + p, rho * u * v, rho * u * w])
print(fc_momentum_0)

fc_momentum_1 = np.array([rho * u * v, rho * v * v + p, rho * v * w])
print(fc_momentum_1)

fc_momentum_2 = np.array([rho * u * w, rho * v * w, rho * w * w + p])
print(fc_momentum_2)

fc_energy = rho * Cp * np.array([u * T, v * T, w * T])
print(fc_energy)

fc_source = rho * Cp * grad_temp * np.array([u, v, w])
print("source term in 2D", np.sum(fc_source[0:-1]))
print("source term in 3D", np.sum(fc_source))
