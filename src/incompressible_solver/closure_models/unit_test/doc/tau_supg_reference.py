import numpy as np
import math

## Momentum equation
# Coefficients
nu = 0.1

# 2-D case
vel = [0.5, 1.5]
h = [3.0, 6.0]

# Multi-D transient
alpha_cont = 3.0
alpha_mom = 0.5
dt = 0.2
u2_over_h2 = 0.0
h2 = 0.0
for v, l in zip(vel, h):
    u2_over_h2 += v * v / (l * l)
    h2 += l * l
tau = 1.0 / math.sqrt(4.0 /
                      (dt * dt) + 4.0 * u2_over_h2 + 9.0 * 16.0 * nu * nu /
                      (h2 * h2))
print("Multi-D transient tau continuity: ", alpha_cont * tau)
print("Multi-D transient tau momentum: ", alpha_mom * tau)

# 3-D case
# Multi-D steady
alpha_cont = 0.5
alpha_mom = 2.0
vel = [0.5, 1.5, 2.5]
h = [3.0, 6.0, 9.0]
u2_over_h2 = 0.0
h2 = 0.0
for v, l in zip(vel, h):
    u2_over_h2 += v * v / (l * l)
    h2 += l * l

tau = 1.0 / math.sqrt(4.0 * u2_over_h2 + 9.0 * 16.0 * nu * nu / (h2 * h2))
print("Multi-D steady tau continuity: ", alpha_cont * tau)
print("Multi-D steady tau momentum: ", alpha_mom * tau)

## Temperature equation
# Coefficients
D = 0.2
alpha = 0.5
tol = 1.0e-8

# 2-D case
vel = [0.5, 1.5]
h = [3.0, 6.0]

# Upwind
tau = alpha * h[0] / (vel[0] + tol)
print("Upwind tau: ", tau)

# Nodal
alpha = 2.0
Pe = 0.5 * vel[0] * h[0] / D
beta = math.cosh(Pe) / math.sinh(Pe) - 1.0 / Pe
tau = alpha * h[0] * beta / (vel[0] + tol)
print("Nodal tau: ", tau)

# Multi-D transient
alpha = 0.5
dt = 0.2
u2_over_h2 = 0.0
h2 = 0.0
for v, l in zip(vel, h):
    u2_over_h2 += v * v / (l * l)
    h2 += l * l
tau = alpha / math.sqrt(4.0 /
                        (dt * dt) + 4.0 * u2_over_h2 + 9.0 * 16.0 * D * D /
                        (h2 * h2))
print("Multi-D transient tau: ", tau)

# 3-D case
# Multi-D steady
rho = 3.0
D *= 1.0 / rho
vel = [0.5, 1.5, 2.5]
h = [3.0, 6.0, 9.0]
u2_over_h2 = 0.0
h2 = 0.0
for v, l in zip(vel, h):
    u2_over_h2 += v * v / (l * l)
    h2 += l * l

tau = alpha / math.sqrt(4.0 * u2_over_h2 + 9.0 * 16.0 * D * D / (h2 * h2))
print("Multi-D steady tau: ", tau)
