import numpy as np

m_dot_target = 0.85
m_dot_inlet = 1.0
wall_shear_stress = 4.0
bottom_wall_area = 0.7
inlet_wall_area = 1.2
lorentz_force = 2.0

source = m_dot_target / m_dot_inlet * (2.0 *
                                       (wall_shear_stress / bottom_wall_area) /
                                       inlet_wall_area)
print(f"Source term for constant volumetric flow rate: {source}")

source -= m_dot_target / m_dot_inlet * (lorentz_force / inlet_wall_area)
print(f"Source term for constant volumetric flow rate with MHD: {source}")

print(f"Source term for variable volumetric flow rate: {0.1}, {0.2}, {0.3}")
