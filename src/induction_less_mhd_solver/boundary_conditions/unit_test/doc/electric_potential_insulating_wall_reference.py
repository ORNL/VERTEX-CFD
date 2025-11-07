import numpy as np

# 2-D case
normals = np.array([3.0, 6.0])
ep_grad = np.array([0.1, 0.2])

ep_dot_n = np.dot(normals, ep_grad)
print(ep_dot_n)

print("\n2-D case:")
print("x-boundary value: ", ep_grad[0] - ep_dot_n * normals[0])
print("y-boundary value: ", ep_grad[1] - ep_dot_n * normals[1])

# 3-D case
normals = np.array([3.0, 6.0, 9.0])
ep_grad = np.array([0.1, 0.2, 0.3])

ep_dot_n = np.dot(normals, ep_grad)

print("\n3-D case:")
print("x-boundary value: ", ep_grad[0] - ep_dot_n * normals[0])
print("y-boundary value: ", ep_grad[1] - ep_dot_n * normals[1])
print("z-boundary value: ", ep_grad[2] - ep_dot_n * normals[2])
