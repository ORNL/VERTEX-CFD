import numpy as np

# Wall function quantities
u_tau = 2.0
y_plus = 13.0
nu_t = 5.0

# Flow quantities
rho = 1.0
nu = 2.5
vel = np.array([1.0, -2.0, 3.0])

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, -0.5], [0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, -0.5, -0.125], [0.5, 1.0, 0.25],
                        [-0.75, -1.5, -0.375]])

grad_temp = np.array([1.0, -2.0, 3.0])

grad_press = np.array([4.2, 1.7, -3.5])

normals = [-0.02, 0.04, -0.06]

dim_list = [2, 3]

for dim in dim_list:
    print("Computing boundary velocity gradient in ", dim, "D\n")

    grad_vel = grad_vel_2D
    if (dim == 3):
        grad_vel = grad_vel_3D

    vel_dim = vel[:dim]
    grad_temp_dim = grad_temp[:dim]
    grad_press_dim = grad_press[:dim]
    normals_dim = normals[:dim]

    boundary_vel = vel_dim - np.array(np.dot(vel_dim,
                                             normals_dim)) * normals_dim
    print("Boundary velocity: ", boundary_vel)

    boundary_grad_temp = grad_temp_dim - np.array(
        np.dot(grad_temp_dim, normals_dim)) * normals_dim

    print("Boundary temperature gradient: ", boundary_grad_temp)

    boundary_grad_press = grad_press_dim - np.array(
        np.dot(grad_press_dim, normals_dim)) * normals_dim

    print("Boundary pressure gradient: ", boundary_grad_press)

    for d in range(dim):
        boundary_grad_u = grad_vel[d]
        if d == 0:
            boundary_grad_u[1] = u_tau / y_plus * vel_dim[d] / rho / (nu +
                                                                      nu_t)
        print("Boundary gradient of component ", d, " is: ", boundary_grad_u,
              "\n")
    print("\n")
