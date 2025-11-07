from decimal import *
import math

places = Decimal(10)**-16


def show(d):
    print(d.quantize(places))


zero = Decimal(0)
half = Decimal(0.5)
one = Decimal(1)
two = Decimal(2)
three = Decimal(3)
twopi = two * Decimal(math.pi)
third = one / three
twothird = two * third

center0 = third
center1 = half
center2 = twothird

sigma0 = half
sigma1 = two
sigma2 = three

base = third

a0 = one / (twopi.sqrt() * sigma0)
a1 = one / (twopi.sqrt() * sigma1)
a2 = one / (twopi.sqrt() * sigma2)

b0 = center0
b1 = center1
b2 = center2

c0 = half / (sigma0 * sigma0)
c1 = half / (sigma1 * sigma1)
c2 = half / (sigma2 * sigma2)

r0 = zero
r1 = one
r2 = one


# Gaussian
def dimResult(a, b, c, x):
    return a * Decimal(math.exp((b - x) * (x - b) * c))


# one-dimensional mesh
print("1D case:")
print("Gaussian")
result0 = dimResult(a0, b0, c0, r0) + base
result1 = dimResult(a0, b0, c0, r1) + base

show(result0)
show(result1)

print("Inverse Gaussian")
show(one / result0)
show(one / result1)

# two-dimensional mesh
print("\n2D case:")
print("Gaussian: Unscaled")
result0 = dimResult(a0, b0, c0, r0) * dimResult(a1, b1, c1, r0) + base
result1 = dimResult(a0, b0, c0, r1) * dimResult(a1, b1, c1, r0) + base
result2 = dimResult(a0, b0, c0, r1) * dimResult(a1, b1, c1, r1) + base
result3 = dimResult(a0, b0, c0, r0) * dimResult(a1, b1, c1, r1) + base

show(result0)
show(result1)
show(result2)
show(result3)

print("Gaussian: Scaled")
result0 = dimResult(a0, b0, c0, r0) / a0 * dimResult(a1, b1, c1,
                                                     r0) / a1 + base
result1 = dimResult(a0, b0, c0, r1) / a0 * dimResult(a1, b1, c1,
                                                     r0) / a1 + base
result2 = dimResult(a0, b0, c0, r1) / a0 * dimResult(a1, b1, c1,
                                                     r1) / a1 + base
result3 = dimResult(a0, b0, c0, r0) / a0 * dimResult(a1, b1, c1,
                                                     r1) / a1 + base

show(result0)
show(result1)
show(result2)
show(result3)

# Inverse Gaussian
print("Inverse Gaussian")
show(one / result0)
show(one / result1)
show(one / result2)
show(one / result3)

# three-dimensional mesh
print("\n3D case:")
print("Gaussian: Unscaled")
result0 = dimResult(a0, b0, c0, r0) * dimResult(a1, b1, c1, r0) * dimResult(
    a2, b2, c2, r0) + base
result1 = dimResult(a0, b0, c0, r1) * dimResult(a1, b1, c1, r0) * dimResult(
    a2, b2, c2, r0) + base
result2 = dimResult(a0, b0, c0, r1) * dimResult(a1, b1, c1, r1) * dimResult(
    a2, b2, c2, r0) + base
result3 = dimResult(a0, b0, c0, r0) * dimResult(a1, b1, c1, r1) * dimResult(
    a2, b2, c2, r0) + base
result4 = dimResult(a0, b0, c0, r0) * dimResult(a1, b1, c1, r0) * dimResult(
    a2, b2, c2, r2) + base
result5 = dimResult(a0, b0, c0, r1) * dimResult(a1, b1, c1, r0) * dimResult(
    a2, b2, c2, r2) + base
result6 = dimResult(a0, b0, c0, r1) * dimResult(a1, b1, c1, r1) * dimResult(
    a2, b2, c2, r1) + base
result7 = dimResult(a0, b0, c0, r0) * dimResult(a1, b1, c1, r1) * dimResult(
    a2, b2, c2, r1) + base

show(result0)
show(result1)
show(result2)
show(result3)
show(result4)
show(result5)
show(result6)
show(result7)

print("Gaussian: Scaled")
result0 = dimResult(a0, b0, c0, r0) / a0 * dimResult(
    a1, b1, c1, r0) / a1 * dimResult(a2, b2, c2, r0) / a2 + base
result1 = dimResult(a0, b0, c0, r1) / a0 * dimResult(
    a1, b1, c1, r0) / a1 * dimResult(a2, b2, c2, r0) / a2 + base
result2 = dimResult(a0, b0, c0, r1) / a0 * dimResult(
    a1, b1, c1, r1) / a1 * dimResult(a2, b2, c2, r0) / a2 + base
result3 = dimResult(a0, b0, c0, r0) / a0 * dimResult(
    a1, b1, c1, r1) / a1 * dimResult(a2, b2, c2, r0) / a2 + base
result4 = dimResult(a0, b0, c0, r0) / a0 * dimResult(
    a1, b1, c1, r0) / a1 * dimResult(a2, b2, c2, r2) / a2 + base
result5 = dimResult(a0, b0, c0, r1) / a0 * dimResult(
    a1, b1, c1, r0) / a1 * dimResult(a2, b2, c2, r2) / a2 + base
result6 = dimResult(a0, b0, c0, r1) / a0 * dimResult(
    a1, b1, c1, r1) / a1 * dimResult(a2, b2, c2, r1) / a2 + base
result7 = dimResult(a0, b0, c0, r0) / a0 * dimResult(
    a1, b1, c1, r1) / a1 * dimResult(a2, b2, c2, r1) / a2 + base

show(result0)
show(result1)
show(result2)
show(result3)
show(result4)
show(result5)
show(result6)
show(result7)

print("Inverse Gaussian")
show(one / result0)
show(one / result1)
show(one / result2)
show(one / result3)
show(one / result4)
show(one / result5)
show(one / result6)
show(one / result7)
