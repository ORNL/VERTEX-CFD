import numpy as np

# NOTE:
# - all momentum fluxes have to be multiplied by `nu` for
#   laminar case and `nu + nu_t` for turbulent cases.
# - with unscaled density, all viscous fluxes are to be
#   multiplied by rho
# - energy flux is multiplied by `k` for laminar case and
#   `k + k_t` for turbulent cases.


def continuity_flux(rho, nu, nu_t, gamma, kappa, cp, beta, grad_lagrange_pres,
                    build_turbulence_model, ndim):
    fv_continuity_flux_isothermal = rho * nu * grad_lagrange_pres / beta
    fv_continuity_flux = gamma * kappa * grad_lagrange_pres / (rho * cp * beta)

    return fv_continuity_flux_isothermal[0:ndim], fv_continuity_flux[0:ndim]


def momentum_flux(rho, nu, nu_t, grad_vel, build_turbulence_model,
                  num_space_dim):
    fv_momentum_flux = rho * nu * grad_vel
    if build_turbulence_model: fv_momentum_flux = rho * (nu + nu_t) * grad_vel

    return fv_momentum_flux[0:num_space_dim]


def energy_flux(kappa, kappa_t, grad_temp, build_turbulence_model,
                num_space_dim):
    fv_energy = kappa * grad_temp
    if build_turbulence_model: fv_energy = (kappa + kappa_t) * grad_temp

    return fv_energy[0:num_space_dim]


# Coefficients
# rho and kappa_t is given within the for loop for unscaled_density case
kappa = 0.5
nu = 0.375
nu_t = 4.0
cp = 0.2
gamma = 3.0
Pr_t = 0.8
beta = 2.0

# Velocity vector, velocity and temperature gradients for viscous flux
velocity = np.array([0.25, 0.5, 0.125])
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.125, 0.25, -0.375]])
grad_temp = np.array([-0.75, 1.5, -2.25])
grad_lagrange_pres = np.array([-0.75, 1.5, -2.25])

# Create expected values for every mesh dimension and
# Scaled/Unscaled density cases with/without turbulence model
dims = [2, 3]
density_type = ['Scaled', 'Unscaled']
for dim in dims:
    print("\n" + str(dim) + "-D case:")
    for den_type in density_type:
        print("\n" + den_type + " Density:")
        for build_turbulence_model in [True, False]:
            if den_type == 'Scaled': rho = 1.0
            if den_type == 'Unscaled': rho = 3.0
            kappa_t = rho * nu_t * cp / Pr_t

            print("\nTurbulence Model is " + str(build_turbulence_model) +
                  "\n")

            fv_continuity_isothermal, fv_continuity = continuity_flux(
                rho, nu, nu_t, gamma, kappa, cp, beta, grad_lagrange_pres,
                build_turbulence_model, dim)
            print("fv_continuity for AC Model:", np.zeros([dim]))
            print("fv_continuity for EDAC Model without temperature equation:",
                  fv_continuity_isothermal)
            print("fv_continuity for EDAC Model with temperature equation:",
                  fv_continuity)

            fv_momentum_0 = momentum_flux(rho, nu, nu_t, grad_vel[0],
                                          build_turbulence_model, dim)
            print("fv_momentum_0:", fv_momentum_0)

            fv_momentum_1 = momentum_flux(rho, nu, nu_t, grad_vel[1],
                                          build_turbulence_model, dim)
            print("fv_momentum_1:", fv_momentum_1)

            fv_momentum_2 = momentum_flux(rho, nu, nu_t, grad_vel[2],
                                          build_turbulence_model, dim)
            if dim > 2: print("fv_momentum_2:", fv_momentum_2)

            fv_energy = energy_flux(kappa, kappa_t, grad_temp,
                                    build_turbulence_model, dim)
            print("fv_energy:", fv_energy)
