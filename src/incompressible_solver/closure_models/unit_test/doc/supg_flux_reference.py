import numpy as np


## Function
def compute_flux(dxdt_vel, vel, grad_vel, grad_p, mom_source, rho, tau_cont,
                 tau_mom, dim):
    for j in range(0, dim):
        res_vel = dxdt_vel[j]
        for v, gv in zip(vel, grad_vel[j]):
            res_vel += v * gv
        res_vel *= rho
        res_vel += grad_p[j]
        res_vel -= mom_source[j]

        print("\n\tContinuity SUPG flux " + str(j) + ":",
              tau_cont * res_vel / rho)
        print("\tMomentum " + str(j) + " SUPG flux 0:",
              tau_mom * res_vel * vel[0])
        print("\tMomentum " + str(j) + " SUPG flux 1:",
              tau_mom * res_vel * vel[1])
        if (dim == 3):
            print("\tMomentum " + str(j) + " SUPG flux 2:",
                  tau_mom * res_vel * vel[2])


## Momentum and continuity equations
# Coefficients
tau_cont = 3.5
tau_mom = 3.6
rho = 1.0

# 2-D case
dxdt_vel = [1.1, 1.2]
vel = [1.5, 2.5]
grad_vel = [[-2.1, 4.2], [-2.2, 4.4]]
grad_p = [-3.1, 6.2]
mom_source = [4.1, 4.2]

print("\n2-D case for NS equations:")
compute_flux(dxdt_vel, vel, grad_vel, grad_p, mom_source, rho, tau_cont,
             tau_mom, 2)

# 3-D case
dxdt_vel = [1.1, 1.2, 1.3]
vel = [1.5, 2.5, 3.5]
grad_vel = [[-2.1, 4.2, -6.3], [-2.2, 4.4, -6.6], [-2.3, 4.6, -6.9]]
grad_p = [-3.1, 6.2, -9.3]
mom_source = [4.1, 4.2, 4.3]

print("\n3-D case for NS equations:")
compute_flux(dxdt_vel, vel, grad_vel, grad_p, mom_source, rho, tau_cont,
             tau_mom, 3)

## Temperature equation
# Coefficients
dxdt_temp = 4.0
energy_source = 6.0
tau_ene = 7.0
rhoCp = 10.0

# 2-D case
vel = [1.5, 2.5]
grad_temp = [-8.0, 16.0]

print("\n2-D case for temperature equation:")
res_temp = rhoCp * dxdt_temp - energy_source
for v, gt in zip(vel, grad_temp):
    res_temp += rhoCp * v * gt

print("Energy SUPG flux 0: ", tau_ene * res_temp * vel[0])
print("Energy SUPG flux 1: ", tau_ene * res_temp * vel[1])

# 3-D case
vel = [1.5, 2.5, 3.5]
grad_temp = [-8.0, 16.0, -24.0]

print("\n3-D case for temperature equation:")
res_temp = rhoCp * dxdt_temp - energy_source
for v, gt in zip(vel, grad_temp):
    res_temp += rhoCp * v * gt

print("Energy SUPG flux 0: ", tau_ene * res_temp * vel[0])
print("Energy SUPG flux 1: ", tau_ene * res_temp * vel[1])
print("Energy SUPG flux 2: ", tau_ene * res_temp * vel[2])
