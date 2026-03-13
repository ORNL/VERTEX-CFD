import numpy as np

ms = 0.2
nd = 1.5
u = 0.25
v = 0.5
w = 0.75
T = 2.5
dnddt = -0.5
dudt = -1.5
dvdt = -3.0
dwdt = -4.5
dTdt = -3.0

# Momentum density
rho = ms * nd
print("Momentum density:", rho)
drhodt = ms * dnddt
print("Time derivative of momentum density:", drhodt)

# Pressure
p = nd * T
dpdt = nd * dTdt + dnddt * T
print("Pressure:", p)

# Energy density
vel2 = u**2 + v**2
dvel2dt = 2.0 * (u * dudt + v * dvdt)
print("2-D energy density:", 0.5 * 3 * p + 0.5 * rho * vel2)
print("2-D time derivative of energy density:",
      0.5 * 3 * dpdt + 0.5 * rho * dvel2dt + 0.5 * drhodt * vel2)

vel2 = u**2 + v**2 + w**2
dvel2dt = 2.0 * (u * dudt + v * dvdt + w * dwdt)
print("3-D energy density:", 0.5 * 3 * p + 0.5 * rho * vel2)
print("3-D time derivative of energy density:",
      0.5 * 3 * dpdt + 0.5 * rho * dvel2dt + 0.5 * drhodt * vel2)
