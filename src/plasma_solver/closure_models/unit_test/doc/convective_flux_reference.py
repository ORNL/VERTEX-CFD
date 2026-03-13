import numpy as np

nd = 2.0
u = 4.0
v = 8.0
w = 12.0
rho = 1.5
p = 2.5
et = 3.5

fc_continuity = np.array([nd * u, nd * v, nd * w])
print("Number density equation:", fc_continuity)

fc_momentum_0 = np.array([rho * u * u + p, rho * u * v, rho * u * w])
print("X-momentum density equation:", fc_momentum_0)

fc_momentum_1 = np.array([rho * u * v, rho * v * v + p, rho * v * w])
print("Y-momentum equation:", fc_momentum_1)

fc_momentum_2 = np.array([rho * u * w, rho * v * w, rho * w * w + p])
print("Z-momentum equation:", fc_momentum_2)

fc_energy = np.array([(et + p) * u, (et + p) * v, (et + p) * w])
print("Energy density equation:", fc_energy)
