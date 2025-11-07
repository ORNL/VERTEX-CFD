import numpy as np

# Flow quantities
phi = 1.5
nu = 0.375
vel = np.array([0.25, 0.5, 0.125])
normals = np.array([-0.75, 1.5, -2.25])
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.125, 0.25, -0.375]])
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
        boundary_vel = np.zeros(dim)
        boundary_grad_vel = grad_vel[:dim, :dim]

        print("Boundary Lagrangre Pressure: ", boundary_lagrange_pres)
        if (continuity_model == 'EDAC'):
            print("Boundary Lagrangre Pressure Gradient: ",
                  boundary_grad_lagrange_pres)
        print("Boundary Velocity: ", boundary_vel)
        print("Boundary Velocity Gradient: ", boundary_grad_vel)
        print("\n")
