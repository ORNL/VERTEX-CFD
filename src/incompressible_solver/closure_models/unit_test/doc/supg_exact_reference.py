import math

# Node coordinates (set in the unit test)
x_list = [-0.15, -0.05, 0.05, 0.15]


# Function
def supg(x, u, D):
    t = (x - (1.0 - math.exp(u * x / D)) / (1.0 - math.exp(u / D))) / u
    return t


# Compute exact values
u = 5.0
D = 3.0 / (2.0 * 4.0)
t_list = []
for x in x_list:
    t = supg(x, u, D)
    t_list.append(t)

# Print exact values
print("t: ", t_list)
