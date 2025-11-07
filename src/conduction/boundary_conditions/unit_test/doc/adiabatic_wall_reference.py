import numpy as np

temperature = 2.0
normals = np.array([-0.6, 1.2, -1.8])
grad_temp = np.array([0.75, -1.5, 2.25])

dim_list = [2, 3]
for dim in dim_list:
    print("Dimension is", dim)
    boundary_temp = temperature
    grad_T_dot_n = np.dot(grad_temp[:dim], normals[:dim])
    boundary_grad_temp = grad_temp[:dim] - grad_T_dot_n * normals[:dim]

    print("Boundary Temperature: ", boundary_temp)
    print("Boundary Temperature Gradient: ", boundary_grad_temp)
    print("\n")
