import numpy as np

# Flow quantities
phi = 1.5
nu = 0.375
vel = np.array([0.25, 0.5, 0.125])
normals = np.array([-0.75, 1.5, -2.25])
grad_vel_0 = np.array([-0.25, 0.5, -0.75])
grad_vel_1 = np.array([-0.5, 1.0, -1.5])
grad_vel_2 = np.array([-0.125, 0.25, -0.375])
grad_pressure = np.array([-0.75, 1.5, -2.25])

dim_list = [2, 3]
continuity_models = ['AC', 'EDAC']
for continuity_model in continuity_models:
    for dim in dim_list:
        print("Flow variables for", continuity_model, "model:")
        print("Dimension is", dim)
        boundary_lagrange_pres = phi
        if (continuity_model == 'EDAC'):
            grad_lagrange_pres_dot_n = np.dot(grad_pressure[:dim],
                                              normals[:dim])
            boundary_grad_lagrange_pres = grad_pressure[:
                                                        dim] - grad_lagrange_pres_dot_n * normals[:
                                                                                                  dim]
        print("Boundary Lagrange pressure: ", boundary_lagrange_pres)

        if (continuity_model == 'EDAC'):
            print("Boundary Grad Lagrange pressure: ",
                  boundary_grad_lagrange_pres)

        vel_dot_n = np.dot(vel[:dim], normals[:dim])
        boundary_vel = vel[:dim] - vel_dot_n * normals[:dim]
        print("Boundary velocity: ", boundary_vel)

        grad_vel_0_dot_n = np.dot(grad_vel_0[:dim], normals[:dim])
        boundary_grad_vel_0 = grad_vel_0[:
                                         dim] - grad_vel_0_dot_n * normals[:dim]
        print("Boundary Grad velocity 0: ", boundary_grad_vel_0)

        grad_vel_1_dot_n = np.dot(grad_vel_1[:dim], normals[:dim])
        boundary_grad_vel_1 = grad_vel_1[:
                                         dim] - grad_vel_1_dot_n * normals[:dim]
        print("Boundary Grad velocity 1: ", boundary_grad_vel_1)

        if (dim == 3):
            grad_vel_2_dot_n = np.dot(grad_vel_2[:dim], normals[:dim])
            boundary_grad_vel_2 = grad_vel_2[:
                                             dim] - grad_vel_2_dot_n * normals[:
                                                                               dim]
            print("Boundary Grad velocity 2: ", boundary_grad_vel_2)

        print("\n")
