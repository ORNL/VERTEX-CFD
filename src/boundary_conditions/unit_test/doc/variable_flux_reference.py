import numpy as np

# Test variables
normals = np.array([-0.75, 1.5, -2.25])
variable = 0.75
wallflux = 2.0
dependent_list = [0.5, 1.0]
grad_variable = np.array([-0.75, 1.5, -2.25])

dim_list = [2, 3]
for dependent in dependent_list:
    for dim in dim_list:
        print("dependent variable is", dependent)
        print("Dimension is", dim)

        boundary_variable = variable
        boundary_grad_variable = grad_variable[:dim]
        grad_variable_dot_n = np.dot(grad_variable[:dim], normals[:dim])
        boundary_grad_variable = boundary_grad_variable - (
            grad_variable_dot_n - (wallflux / dependent)) * normals[:dim]

        print("Boundary Variable: ", boundary_variable)
        print("Boundary Variable Gradient: ", boundary_grad_variable)
        print("\n")
