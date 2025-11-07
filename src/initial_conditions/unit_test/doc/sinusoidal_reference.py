from decimal import *
import math

places = Decimal(10)**-16


def show(d):
    print(d.quantize(places))


zero = Decimal(0)
half = Decimal(0.5)
one = Decimal(1)
two = Decimal(2)
twohalf = Decimal(2.5)
three = Decimal(3)
twopi = two * Decimal(math.pi)
third = one / three
twothird = two * third

amplitude = two
phase = twohalf

r0 = zero
r1 = one
r2 = two
r3 = three


# Sinusoidal
def dimResult(a, p, x):
    return a * Decimal(math.sin(x + p))


# two-dimensional mesh
print("\n2D case:")
result0 = dimResult(amplitude, phase, r0)
result1 = dimResult(amplitude, phase, r1)
result2 = dimResult(amplitude, phase, r2)
result3 = dimResult(amplitude, phase, r1)

show(result0)
show(result1)
show(result2)
show(result3)

# three-dimensional mesh
print("\n3D case:")
result0 = dimResult(amplitude, phase, r0)
result1 = dimResult(amplitude, phase, r1)
result2 = dimResult(amplitude, phase, r2)
result3 = dimResult(amplitude, phase, r1)
result4 = dimResult(amplitude, phase, r1)
result5 = dimResult(amplitude, phase, r2)
result6 = dimResult(amplitude, phase, r3)
result7 = dimResult(amplitude, phase, r2)

show(result0)
show(result1)
show(result2)
show(result3)
show(result4)
show(result5)
show(result6)
show(result7)
