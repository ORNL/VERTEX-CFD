import numpy as np

multiplier = np.array([2.5, 4.0])
source = 0.25
species = np.array([1.25, 2.5])

print("Vapor removal source: ", multiplier * source * species)
