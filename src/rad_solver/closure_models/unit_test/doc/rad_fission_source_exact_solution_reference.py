import numpy as np

_num_species = 2
flux_amp = 1.2
xs = 0.8
avagadro = 6.02214076e23
a = 0.5
b = 1.0
kappa = 0.3
beta = 0.7
time = 1.0

_gamma = 6.02214076e23 * np.array([1, 2])
_initial_amp = [0, 0]

_flux = flux_amp * (((np.tanh(a * time - b) + 1) / 2) +
                    (-(np.tanh(kappa * time - beta) + 1) / 2))
print("Neutron Flux: ", _flux)
print("Exact species values:")
for num in range(_num_species):
    const_s = flux_amp * _gamma[num] * xs / avagadro
    const_c = (-const_s * np.log(np.cosh(b)) / (2 * a) +
               const_s * np.log(np.cosh(-beta)) / (2 * kappa) -
               _initial_amp[num])
    _exact_species = (const_s * np.log(np.cosh(b - a * time)) / (2 * a) -
                      const_s * np.log(np.cosh(kappa * time - beta)) /
                      (2 * kappa) + const_c)
    print(_exact_species)
