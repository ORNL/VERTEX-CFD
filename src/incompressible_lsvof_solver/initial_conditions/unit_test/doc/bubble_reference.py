import math

# Bubble parameters
xc = 0.47
yc = 0.92
zc = 0.95
r = 0.55

# Node coordinates
x_list_2D = [0.0, 1.0, 1.0, 0.0]
y_list_2D = [0.0, 0.0, 1.0, 1.0]

x_list_3D = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
y_list_3D = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
z_list_3D = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def bub_2d(x, y):
    return r - pow(pow(xc - x, 2.0) + pow(yc - y, 2.0), 0.5)


def bub_3d(x, y, z):
    return r - pow(pow(xc - x, 2.0) + pow(yc - y, 2.0) + pow(zc - z, 2.0), 0.5)


# 2D CLS solution
phi_list_2D = []

for x, y in zip(x_list_2D, y_list_2D):
    phi = bub_2d(x, y)
    phi_list_2D.append(phi)

print("2D CLS solution: ", phi_list_2D)

# 3D CLS solution
phi_list_3D = []

for x, y, z in zip(x_list_3D, y_list_3D, z_list_3D):
    phi = bub_3d(x, y, z)
    phi_list_3D.append(phi)

print("3D CLS solution: ", phi_list_3D)
