import math

# Node coordinates (set in the unit test)
x_list = [-0.15, -0.05, 0.05, 0.15]
y_list = [-0.05, 0.15, 0.35, 0.55]
z_list = [0.05, 0.35, 0.65, 0.95]


# Function
def ic2d(x, y):
    u = math.cos(x) * math.sin(y)
    v = -math.sin(x) * math.cos(y)
    p = -0.25 * (math.cos(2.0 * x) + math.cos(2.0 * y))
    return u, v, p


def ic3d(x, y, z):
    u, v, p = ic2d(x, y)
    u = u * math.cos(z)
    v = v * math.cos(z)
    w = 0.0
    p = p * 0.25 * math.cos(2.0 * z + 2.0)
    return u, v, w, p


# Compute ic values
u_list = []
v_list = []
w_list = []
p_list = []

# 2D
print("2D values:")
for x, y in zip(x_list, y_list):
    u, v, p = ic2d(x, y)
    p_list.append(p)
    u_list.append(u)
    v_list.append(v)

# Print ic values
print("    phi: ", p_list)
print("    u: ", u_list)
print("    v: ", v_list)

# 3D
u_list = []
v_list = []
w_list = []
p_list = []

x_list = [-0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
y_list = [-0.05, 0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35]
z_list = [0.05, 0.35, 0.65, 0.95, 1.25, 1.55, 1.85, 2.15]

print("3D values:")
for x, y, z in zip(x_list, y_list, z_list):
    u, v, w, p = ic3d(x, y, z)
    p_list.append(p)
    u_list.append(u)
    v_list.append(v)
    w_list.append(w)

# Print ic values
print("    phi: ", p_list)
print("    u: ", u_list)
print("    v: ", v_list)
print("    w: ", w_list)
