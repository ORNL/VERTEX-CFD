import numpy as np

# 2-D case
ep_grad = np.array([3.0, 6.0])
normals = np.array([0.1, 0.2])
B = np.array([0.0, 0.0, 0.3])
v_bc = np.array([0.6, 1.4, 0.0])
v = np.array([0.5, 1.5, 0.0])

J_dot_n = np.dot(ep_grad - np.cross(v, B)[:-1], normals)

print("\n2-D case:")
print("J_n: ", J_dot_n)
print(
    "x-boundary value: ", ep_grad[0] + np.cross(v, B)[0] -
    np.cross(v_bc, B)[0] - J_dot_n * normals[0])
print(
    "y-boundary value: ", ep_grad[1] + np.cross(v, B)[1] -
    np.cross(v_bc, B)[1] - J_dot_n * normals[1])

# 3-D case
ep_grad = np.array([3.0, 6.0, 9.0])
normals = np.array([0.1, 0.2, 0.3])
B = np.array([1.1, 2.0, -0.3])
v_bc = np.array([0.6, 1.4, 3.0])
v = np.array([0.5, 1.5, 2.5])

J_dot_n = np.dot(ep_grad - np.cross(v, B), normals)

print("\n3-D case:")
print("J_n: ", J_dot_n)
print(
    "x-boundary value: ", ep_grad[0] + np.cross(v, B)[0] -
    np.cross(v_bc, B)[0] - J_dot_n * normals[0])
print(
    "y-boundary value: ", ep_grad[1] + np.cross(v, B)[1] -
    np.cross(v_bc, B)[1] - J_dot_n * normals[1])
print(
    "z-boundary value: ", ep_grad[2] + np.cross(v, B)[2] -
    np.cross(v_bc, B)[2] - J_dot_n * normals[2])
