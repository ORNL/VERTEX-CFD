import numpy as np

# Turbulent quantities
k = 40.0
e = 5.0

# Flow and model constants
nu = 0.0115
C_nu = 0.09

# Flow variables
distance = 0.5
surface_area = np.array([1.0, 0.0001])
tau = 4 / surface_area

eddy_visc = C_nu * k**2 / e * (1.0 - np.exp(-0.0115 * tau * distance / nu))
print(
    f"Eddy viscosity for unit area: {eddy_visc[0]}\nEddy viscosity for non unit area: {eddy_visc[1]}\n"
)
