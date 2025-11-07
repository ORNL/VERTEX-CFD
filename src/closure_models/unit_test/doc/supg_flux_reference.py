import numpy as np

# Coefficients
dxdt_var = 4.0
var_source = 6.0
tau_var = 7.0
diff_var_flux = [-9.0, 18.0, -27.0]

# 2-D case
vel = [1.5, 2.5]
grad_var = [-8.0, 16.0]

print("\n2-D case:")
res_var = dxdt_var - var_source
for v, gt in zip(vel, grad_var):
    res_var += v * gt

print("Scalar variable SUPG flux 0: ",
      diff_var_flux[0] + tau_var * res_var * vel[0])
print("Scalar variable SUPG flux 1: ",
      diff_var_flux[1] + tau_var * res_var * vel[1])

# 3-D case
vel = [1.5, 2.5, 3.5]
grad_var = [-8.0, 16.0, -24.0]

print("\n3-D case:")
res_var = dxdt_var - var_source
for v, gt in zip(vel, grad_var):
    res_var += v * gt

print("Scalar variable SUPG flux 0: ",
      diff_var_flux[0] + tau_var * res_var * vel[0])
print("Scalar variable SUPG flux 1: ",
      diff_var_flux[1] + tau_var * res_var * vel[1])
print("Scalar variable SUPG flux 2: ",
      diff_var_flux[2] + tau_var * res_var * vel[2])
