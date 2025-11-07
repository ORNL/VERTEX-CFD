import numpy as np

# Flow quantities
phi = 1.5
set_phi = 0
nu = 0.375
vel = np.array([0.25, 0.5, 0.125])
normals = np.array([-0.75, 1.5, -2.25])
temperature = 0.75
Twall = 0.125
wallflux = 2.0
kappa = 0.5
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.125, 0.25, -0.375]])
grad_temp = np.array([-0.75, 1.5, -2.25])
grad_pressure = np.array([-0.75, 1.5, -2.25])
set_grad_pressure = np.zeros(3)

dim_list = [2, 3]
temperature_profiles = ['adiabatic', 'isothermal', 'flux']
continuity_models = ['AC', 'EDAC']
for continuity_model in continuity_models:
    for dim in dim_list:
        for temperature_profile in temperature_profiles:
            print("Flow variables for", continuity_model, "model:")
            print("Dimension is", dim)
            print("Temperature profile is", temperature_profile)
            boundary_lagrange_pres = phi
            boundary_lagrange_pres_set_lagrange = set_phi
            if (continuity_model == 'EDAC'):
                grad_lagrange_pres_dot_n = np.dot(grad_pressure[:dim],
                                                  normals[:dim])
                boundary_grad_lagrange_pres = grad_pressure[:
                                                            dim] - grad_lagrange_pres_dot_n * normals[:
                                                                                                      dim]
            boundary_vel = np.zeros(dim)
            boundary_grad_vel = grad_vel[:dim, :dim]
            if temperature_profile == 'isothermal':
                boundary_grad_temp = grad_temp[:dim]
                boundary_temp = Twall
            elif temperature_profile == 'adiabatic':
                boundary_temp = temperature
                boundary_grad_temp = grad_temp[:dim]
                grad_T_dot_n = np.dot(grad_temp[:dim], normals[:dim])
                boundary_grad_temp = boundary_grad_temp - grad_T_dot_n * normals[:
                                                                                 dim]
            elif temperature_profile == 'flux':
                boundary_temp = temperature
                boundary_grad_temp = grad_temp[:dim]
                grad_T_dot_n = np.dot(grad_temp[:dim], normals[:dim])
                boundary_grad_temp = boundary_grad_temp - (
                    grad_T_dot_n - (wallflux / kappa)) * normals[:dim]

            print("Boundary Lagrangre Pressure: ", boundary_lagrange_pres)
            print("Defined Boundary Lagrangre Pressure: ",
                  boundary_lagrange_pres_set_lagrange)
            if (continuity_model == 'EDAC'):
                print("Boundary Lagrangre Pressure Gradient: ",
                      boundary_grad_lagrange_pres)
            print("Boundary Velocity: ", boundary_vel)
            print("Boundary Velocity Gradient: ", boundary_grad_vel)
            print("Boundary Temperature: ", boundary_temp)
            print("Boundary Temperature Gradient: ", boundary_grad_temp)
            print("\n")
