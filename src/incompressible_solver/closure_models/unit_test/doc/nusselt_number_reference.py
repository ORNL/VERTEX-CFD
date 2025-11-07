import numpy as np

# Define normal vector and temperature gradient
normals = np.array([-0.75, 1.5, -2.25])
grad_temp = np.array([-1.5, 3.0, -4.5])

# Loop for possible dimension options
dims = [2, 3]
for dim in dims:
    print("Nusselt number for " + str(dim) + " case: ",
          np.dot(grad_temp[:dim], normals[:dim]))
