from math import *

x = 0.2
y = 0.3
r2 = x * x + y * y
B_magn = 1.3

print("Toroidal: ")
print("\tBx: ", -y * B_magn / r2)
print("\tBy: ", x * B_magn / r2)
print("\tBz: ", 0.0)

B_magn = [1.0, 2.0, 3.0]
curve_coef = 1.0
offset = 0.5
print("X-dependent: ")
print("\tBx: ", B_magn[0] * (1 - tanh(-1 * (x / curve_coef - offset))) / 2)
print("\tBy: ", B_magn[1] * (1 - tanh(-1 * (x / curve_coef - offset))) / 2)
print("\tBz: ", B_magn[2] * (1 - tanh(-1 * (x / curve_coef - offset))) / 2)
