from decimal import *

getcontext().prec = 60
places = Decimal(10)**-16


# Print value
def show(name, d):
    print(name + " = ", str(d.quantize(places)).rstrip('0'))


# Function to calculate boundary values
def calculate_boundary_values(omega,
                              omega_init,
                              x,
                              y,
                              dim,
                              time,
                              time_init=Decimal(0.0),
                              time_final=Decimal(10)**-6):
    # Compute time transient coefficients
    if (time < time_init): time = time_init
    if (time > time_final): time = time_final

    dt = time_final - time_init
    a = (omega - omega_init) / dt
    b = omega_init - a * time_init

    # Compute velocity components
    u = -(a * time + b) * y
    v = (a * time + b) * x
    w = Decimal(0.0)  # the z-component of the velocity is always 0.0

    # Compute boundary pressure gradient
    dqp = np.arange(3) + 1 + dim
    grad_pressure = 1.0 * dqp
    normals = 0.1 * dqp
    grad_lagrange_pres_dot_n = np.dot(grad_pressure[:dim], normals[:dim])
    boundary_grad_lagrange_pres = grad_pressure[:
                                                dim] - grad_lagrange_pres_dot_n * normals[:
                                                                                          dim]
    # Show values
    show("u", u)
    show("v", v)
    if (dim == 3): show("w", w)
    print("pressure gradient = ", boundary_grad_lagrange_pres)


# Constant input arguments for all tests
omega = Decimal(2.0)
x = Decimal('0.7375')
y = Decimal('0.9775')

# 2-D cases
# Steady case (no transient)
dim = 2
print("\n2-D test case 'steady")
omega_init = omega

time = Decimal(3.0)

calculate_boundary_values(omega, omega_init, x, y, dim, time)

# time > time_final (same result as steady-state case)
print("\n2-D test case 'time > time_final'")
omega_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal(3.5)

calculate_boundary_values(omega, omega_init, x, y, dim, time, time_init,
                          time_final)

# time < time_init
print("\n2-D test case 'time < time_init'")
omega_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal('0.2')

calculate_boundary_values(omega, omega_init, x, y, dim, time, time_init,
                          time_final)

# time_init < time < time_final
print("\n2-D test case 'time_init < time < time_final'")
omega_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal(1.5)

calculate_boundary_values(omega, omega_init, x, y, dim, time, time_init,
                          time_final)

# 3-D case
dim = 3
# time_init < time < time_final
print("\n3-D test case 'time_init < time < time_final'")
omega_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal(1.5)

calculate_boundary_values(omega, omega_init, x, y, dim, time, time_init,
                          time_final)
