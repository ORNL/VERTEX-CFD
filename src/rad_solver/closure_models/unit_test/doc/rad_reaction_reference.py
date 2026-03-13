import numpy as np

flux_amp = 0.25

bateman_matrix = np.array([[-1.0, 0.0], [1.0, -0.1]])
mic_cross_section = np.array([[-2.0, 1.0], [3.0, -5.1]])

species = np.array([1.25, 2.5])
print("Bateman-Transmutation species: ",
      (flux_amp * mic_cross_section + bateman_matrix) @ species)
print("Bateman species: ", bateman_matrix @ species)
print("Transmutation species: ", (flux_amp * mic_cross_section) @ species)
