import numpy as np
import math

# Coefficients
D = 1.8
tol = 1.0e-10

# 2-D case
vel = [0.5, 1.5]
h = [3.0, 6.0]

# Multi-D steady
alpha = 0.2
u2_over_h2 = 0.0
h2 = 0.0
for v, l in zip(vel, h):
    u2_over_h2 += v * v / (l * l)
    h2 += l * l
tau = alpha / math.sqrt(4.0 * u2_over_h2 + 9.0 * 16.0 * D * D /
                        (h2 * h2) + tol)
print("Multi-D steady tau: ", tau)

# 3-D case
# Multi-D transient
alpha = 0.5
dt = 0.2
vel = [0.5, 1.5, 2.5]
h = [3.0, 6.0, 9.0]
u2_over_h2 = 0.0
h2 = 0.0
for v, l in zip(vel, h):
    u2_over_h2 += v * v / (l * l)
    h2 += l * l

tau = alpha / math.sqrt(4.0 /
                        (dt * dt) + 4.0 * u2_over_h2 + 9.0 * 16.0 * D * D /
                        (h2 * h2))
print("Multi-D transient tau: ", tau)
