import math

# Values
x = 0.625
y = 0.750

u = math.sin(2.0 * math.pi * y) * pow(math.sin(math.pi * x), 2.0)
v = -1.0 * math.sin(2.0 * math.pi * x) * pow(math.sin(math.pi * y), 2.0)

print("Forward vortex velocity components:")
print("u = ", u)
print("v = ", v)
print("Multiply by -1 for reverse values.")
