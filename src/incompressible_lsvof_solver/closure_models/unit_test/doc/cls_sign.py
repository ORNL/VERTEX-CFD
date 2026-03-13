import numpy as np
from math import pi, sin

phi = np.array([-1.5, -0.5, 0.5, 1.5])
phi_star = np.array([-1.25, -0.25, 0.75, 1.75])
epsilon = 0.9

heaviside = np.array([0.0, 0.0, 0.0, 0.0])
sign = np.array([0.0, 0.0, 0.0, 0.0])
sign_star = np.array([0.0, 0.0, 0.0, 0.0])

for i in range(len(phi)):
    if phi[i] < -epsilon:
        heaviside[i] = 0.0
    elif phi[i] > epsilon:
        heaviside[i] = 1.0
    else:
        heaviside[i] = 0.5 * (1.0 + phi[i] / epsilon +
                              1.0 / pi * sin(pi * phi[i] / epsilon))

    sign[i] = 2.0 * heaviside[i] - 1.0

    if phi_star[i] < -epsilon:
        sign_star[i] = 0.0
    elif phi_star[i] > epsilon:
        sign_star[i] = 1.0
    else:
        sign_star[i] = 0.5 * (1.0 + phi_star[i] / epsilon +
                              1.0 / pi * sin(pi * phi_star[i] / epsilon))

    sign_star[i] = 2.0 * sign_star[i] - 1.0

print("heaviside: ", heaviside)
print("sign: ", sign)
print("sign star: ", sign_star)
