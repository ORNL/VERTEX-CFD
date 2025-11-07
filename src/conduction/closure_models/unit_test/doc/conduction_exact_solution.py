import math

# Dependent values
q = 10.0
k = 2.0
T_right = 300.0
x = 0.1

# Temperature
T = T_right * math.exp(q / (6.0 * k) * (1.0 - x * x * x))
print("temperature = ", T)
