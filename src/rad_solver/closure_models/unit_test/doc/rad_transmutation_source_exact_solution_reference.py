import numpy as np

num_species = 2
flux_amp = 1.2
a = 0.5
b = 1.0
kappa = 0.3
beta = 0.7
time = 1.0
mic_cross_section = np.array([[-2.0, 1.0], [3.0, -5.1]])

initial_amp = np.array([1, 0])

flux = flux_amp * (((np.tanh(a * time - b) + 1) / 2) +
                   (-(np.tanh(kappa * time - beta) + 1) / 2))
print("Neutron Flux: ", flux)

val_t = np.log(np.cosh(a * time - b)) - np.log(np.cosh(0.3 * time - 0.7))
val_t0 = np.log(np.cosh(-b)) - np.log(np.cosh(-beta))
phi = (flux_amp / 4.0) * (val_t - val_t0)
A = mic_cross_section * phi
A2 = A @ A
A3 = A2 @ A
A4 = A3 @ A

sol = (np.eye(num_species) + A + A2 / 2 + A3 / 6 + A4 / 24) @ initial_amp
print("Exact species_0 values: ", sol[0])
print("Exact species_1 values: ", sol[1])
