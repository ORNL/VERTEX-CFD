import numpy as np

# electrical conductivity
sigma = 0.5

# 2D
J = np.array([0.1, -0.2])

joule_heating = np.dot(J, J) / sigma
print("2-D JH:", joule_heating)

# 3D
J = np.array([0.1, -0.2, 0.3])

joule_heating = np.dot(J, J) / sigma
print("3-D JH:", joule_heating)
