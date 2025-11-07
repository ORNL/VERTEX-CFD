from math import log, sin, pi

# IC parameters
x = 2.0
y = 3.0
z = -4.0

Lx = 13.0
Lz = 15.0
h = 5.0

nModes = 5

Re_tau = 125

nu = 1.5e-5

# Find bulk/average velocity
u_tau = Re_tau * nu / h

Re_b = pow(Re_tau / 0.09, 1.0 / 0.88)

U_b = Re_b * nu / (2.0 * h)

U_0 = 1.28 * pow(Re_b, -0.0116) * U_b

# Find velocity at point
u = 0
v = 0
w = 0

y_plus = u_tau * (h - y) / nu

if (y_plus < 11.06):
    u = y_plus * u_tau
else:
    u = u_tau * (log(y_plus) / 0.41 + 5.2)

# Add Fourier modes
for n in range(nModes):
    m = n + 1
    u += 0.1 * (pow(h, 2.0) - pow(y, 2.0)) * sin(4.0 * m * pi * z / Lz)
    w += 0.1 * U_0 * (pow(h, 2.0) - pow(y, 2.0)) * sin(4.0 * m * pi * x / Lx)

# Print values
print("\nTurbulent channel flow velocities:")
print("u: ", u)
print("v: ", v)
print("w: ", w, "\n")
