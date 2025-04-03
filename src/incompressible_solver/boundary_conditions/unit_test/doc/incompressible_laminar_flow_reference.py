# Functions
def ic2d(y, origin_coordinate, radius, vel_avg):
    coeff = 3.0 / 2.0
    r = y - origin_coordinate[1]
    u = vel_avg * coeff * (1.0 - r * r / (radius * radius))
    return u, 0.0


def ic3d(y, z, origin_coordinate, radius, vel_avg):
    coeff = 2.0
    r2 = (y - origin_coordinate[1])**2
    r2 += (z - origin_coordinate[2])**2
    u = vel_avg * coeff * (1.0 - r2 / (radius * radius))
    return u, 0.0, 0.0


# IC parameters
origin_coordinate = [0, 0.1, 0]
radius = 2.1
h_min = -2.0
h_max = 2.2
vel_avg = 3.0

# Compute ic values in 2D
y = 0.7375
u, v = ic2d(y, origin_coordinate[0:2], radius, vel_avg)

# Print ic values
print("\n2D laminar flow:")
print("phi: ", 0.0)
print("u: ", u)
print("v: ", v)

# Compute ic values in 3D
z = 0.9775
u, v, w = ic3d(y, z, origin_coordinate, radius, vel_avg)

# Print ic values
print("\n3D laminar flow:")
print("phi: ", 0.0)
print("u: ", u)
print("v: ", v)
print("w: ", w)
