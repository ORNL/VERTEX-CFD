import math

# computational/mms solution
comp = [0.1, 0.3, 0.4, 0.5, 0.8]
exact = [0.2, 0.7, 0.9, 1.1, 1.2]

var_id = range(len(comp))

# L1 error norm
print("L1 norm (phi, vec{u}, T):")
for i in var_id:
    print(abs(comp[i] - exact[i]))

# L2 error norm
print("L2 norm (phi, vec{u}, T):")
for i in var_id:
    print((comp[i] - exact[i])**2.0)
