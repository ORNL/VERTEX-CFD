import numpy as np

time = 5.0
ramp_time = 10
omega_i = 100
omega_f = 100000
grad_k = [0.02, 0.04, 0.06]
normals = [0.33, 0.66, 0.99]

dim_list = [2, 3]

f = time / ramp_time
omega = pow(omega_i, 1.0 - f) * pow(omega_f, f)

print("Omega ramp value: ", omega, "\n")

for dim in dim_list:
    print("Computing boundary variable gradient in ", dim, "D\n")

    normals_dim = normals[:dim]
    grad_k_dim = grad_k[:dim]

    boundary_grad_k = grad_k_dim - np.array(np.dot(grad_k_dim,
                                                   normals_dim)) * normals_dim

    for d in range(dim):
        print("Boundary gradient component ", d, " = ", boundary_grad_k[d],
              "\n")

    print("\n")
