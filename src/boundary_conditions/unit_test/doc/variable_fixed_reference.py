from decimal import *

getcontext().prec = 60
places = Decimal(10)**-16


# Print value
def show(name, d):
    print(name + " = ", str(d.quantize(places)).rstrip('0'))


# Function to calculate boundary values
def calculate_boundary_values(value,
                              value_init,
                              grad_val,
                              time,
                              time_init=Decimal(0.0),
                              time_final=Decimal(10)**-6):
    # Compute time transient coefficients
    if (time < time_init): time = time_init
    if (time > time_final): time = time_final

    dt = time_final - time_init
    a = (value - value_init) / dt
    b = value_init - a * time_init

    # Compute value
    boundary_val = a * time + b

    # Compute boundary value gradient
    boundary_grad_val = grad_val

    # Show values
    show("boundary_val", boundary_val)
    show("boundary_grad_val", grad_val)


# Constant input arguments for all tests
value = Decimal(2.0)
grad_val = Decimal(3.0)

# 2-D cases
# Steady case (no transient)
print("\nTest case 'steady")
value_init = value

time = Decimal(3.0)

calculate_boundary_values(value, value_init, grad_val, time)

# time > time_final (same result as steady-state case)
print("\nTest case 'time > time_final'")
value_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal(3.5)

calculate_boundary_values(value, value_init, grad_val, time, time_init,
                          time_final)

# time < time_init
print("\nTest case 'time < time_init'")
value_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal('0.2')

calculate_boundary_values(value, value_init, grad_val, time, time_init,
                          time_final)

# time_init < time < time_final
print("\nTest case 'time_init < time < time_final'")
value_init = Decimal(1.0)

time_init = Decimal(0.5)
time_final = Decimal(3.0)
time = Decimal(1.5)

calculate_boundary_values(value, value_init, grad_val, time, time_init,
                          time_final)
