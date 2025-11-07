import numpy as np

num_species = 3
flux_amp = 0.25
xs = 0.8
avagadro = 6.02214076e23

gamma = 6.02214076e23 * np.array([1, 2, 4])

print("Exact species values:")
for num in range(num_species):
    exact_species = flux_amp * gamma[num] * xs / avagadro
    print(exact_species)
