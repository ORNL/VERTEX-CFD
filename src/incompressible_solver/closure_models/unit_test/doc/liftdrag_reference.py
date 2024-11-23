import numpy as np

# NOTE:
# - The lift and drag calculations assume that the drag is
#   in x-direction and lift is in y-direction.
# - Currently, lift/drag calculation ignores the turbulence model.


def twosym(t):
    return t + t.T


def dev(t):
    return t - (1 / t.shape[0]) * np.eye(t.shape[0]) * np.trace(t)


def lift_drag(rho,
              nu,
              lagrange_pressure,
              normals,
              grad_velocity,
              dim,
              is_compressible=False):
    # Calculate shear stress with grad.U + transpose(grad.U)
    tau = twosym(grad_velocity[0:dim, 0:dim])
    if is_compressible: tau = dev(tau)
    viscous_force = -rho * nu * np.dot(tau, normals[0:dim])
    pressure_force = lagrange_pressure * normals[0:dim]
    total_force = viscous_force + pressure_force

    return viscous_force, pressure_force, total_force


# Coefficients
# rho is given within the for loop for unscaled_density case
nu = 0.375

# Velocity vector, velocity and temperature gradients for viscous flux
velocity = np.array([0.25, 0.5, 0.125])
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.125, 0.25, -0.375]])
normals = np.array([-0.75, 1.5, -2.25])
lagrange_pres = 0.75

# Create expected values for every mesh dimension and
# Scaled/Unscaled density cases
dims = [2, 3]
density_type = ['Scaled', 'Unscaled']
calculation_types = ['Incompressible', 'Compressible']
for cal_type in calculation_types:
    print("\n" + cal_type + " case:")
    for dim in dims:
        print("\n" + str(dim) + "-D case:")
        for den_type in density_type:
            print("\n" + den_type + " Density:")
            if den_type == 'Scaled': rho = 1.0
            if den_type == 'Unscaled': rho = 3.0

            if cal_type == 'Incompressible':
                viscous_force, pressure_force, total_force = lift_drag(
                    rho, nu, lagrange_pres, normals, grad_vel, dim)
            if cal_type == 'Compressible':
                viscous_force, pressure_force, total_force = lift_drag(
                    rho, nu, lagrange_pres, normals, grad_vel, dim, True)
            print("Viscous Force:", viscous_force)
            print("Pressure Force:", pressure_force)
            print("Total Force:", total_force)
