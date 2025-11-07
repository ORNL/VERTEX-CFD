import math

# Dependent values
b_0 = 1.1
b_1 = -1.2
b_2 = 1.3

h0 = 0.25
h1 = 0.5
h2 = 0.75

# 2-D case, c_h = 0.1
c_h = 0.1
dt = 1.0 / (max(c_h, abs(b_0)) / h0 + max(c_h, abs(b_1)) / h1)
print("2-D dt, c_h = 0.1: ", dt)

# 3-D case, large c_h
c_h = 5.0
dt = 1.0 / (max(c_h, abs(b_0)) / h0 + max(c_h, abs(b_1)) / h1 +
            max(c_h, abs(b_2)) / h2)
print("3-D dt, c_h = 5.0: ", dt)

# 3-D case, no c_h contribution
# no convective speeds, dt set to constant
dt = 1.0e6
print("3-D dt, no c_h: ", dt)
