import math

# Values
grad_phi = [1.0, -2.0, 3.0]

print("2D:")

mag_grad_phi = pow(5.0, 0.5)

print("    q[0] = ", 1.0 / mag_grad_phi)
print("    q[1] = ", -2.0 / mag_grad_phi)

print("")
print("3D:")

mag_grad_phi = pow(14.0, 0.5)

print("    q[0] = ", 1.0 / mag_grad_phi)
print("    q[1] = ", -2.0 / mag_grad_phi)
print("    q[2] = ", 3.0 / mag_grad_phi)
