# Functions
def ic2d(y, h_min, h_max, vel_avg):
    coeff = 3.0 / 2.0
    H = 0.5 * (h_max - h_min)
    y0 = h_min + H
    u = vel_avg * coeff * (1.0 - pow(y - y0, 2.0) / (H * H))
    return u, 0.0


def ic3d(y, z, h_min, h_max, vel_avg):
    coeff = 2.0
    H = 0.5 * (h_max - h_min)
    y0 = h_min + H
    r2 = pow(y - y0, 2.0) + z * z
    u = vel_avg * coeff * (1.0 - r2 / (H * H))
    return u, 0.0, 0.0


# IC parameters
h_min = -2.0
h_max = 2.2
vel_avg = 3.0

# Compute ic values in 2D
num_points = 4
y_list = [(basis + 1) * 0.125 for basis in range(0, num_points)]
u_list = []
v_list = []
for y in y_list:
    u, v = ic2d(y, h_max, h_min, vel_avg)
    u_list.append(u)
    v_list.append(v)

# Print ic values
print("\n2D laminar flow:")
print("phi: ", 0.0)
print("u: ", u_list)
print("v: ", v_list)

# Compute ic values in 3D
num_points = 8
y_list = [(basis + 1) * 0.125 for basis in range(0, num_points)]
z_list = [h_min + y for y in y_list]
u_list = []
v_list = []
w_list = []
for y, z in zip(y_list, z_list):
    u, v, w = ic3d(y, z, h_min, h_max, vel_avg)
    u_list.append(u)
    v_list.append(v)
    w_list.append(w)

# Print ic values
print("\n3D laminar flow:")
print("phi: ", 0.0)
print("u: ", u_list)
print("v: ", v_list)
print("w: ", w_list)
