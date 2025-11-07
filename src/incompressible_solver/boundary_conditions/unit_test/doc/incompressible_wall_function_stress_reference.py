import numpy as np

# Wall function quantities
u_tau = 5.1
y_plus = 11.06

# Flow quantities
vel = np.array([2.4, 3.6, 4.8])

print("Wall function shear stress components:\n")

for vel_d in vel:
    print(u_tau / y_plus * vel_d, "\n")
