# Dependent values
rho = 2.75
Cp = 0.35
k = 1.75
h0 = 1.0
h1 = 2.0
h2 = 3.0

# 2-D case
dt = (h0 * h0 + h1 * h1) * (rho * Cp / (2.0 * k))
print("2-D dt: ", dt)

# 3-D case
dt = (h0 * h0 + h1 * h1 + h2 * h2) * (rho * Cp / (2.0 * k))
print("3-D dt: ", dt)
