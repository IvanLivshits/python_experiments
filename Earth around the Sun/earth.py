'''
Problem Statement
-----------------
- Create a simulation to track the orbit of the Earth around the Sun for a period of 1 year;
- Use Ueler and Runge - Kutta method of 4th order (RK4) for this task;
- Find the distance from Earth to the Sun at Apogee using Euler and RK4 methods and compare it with the original;

Given Equations
---------------
--> Acceleration of the Earth: a = -G * M * r / r^3

--> ODE for Position of the Earth: dr/dt = v

--> ODE for Velocity of the Earth: dv/dt = a

Initinal Condition
------------------
--> Earth is at its Perihelion (closest to the Sun) at the start of the simulation;
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np

# Constants
G = 6.67430e-11 # m^3 kg^-1 s^-2
M_sun = 1.9891e30 # kg

# Initial Position and Velocity
r_0 = np.array([147.1e9, 0]) # m
v_0 = np.array([0, -30.29e3]) # m/s

# Time steps and total time for simulation
dt = 3600 # seconds
t_max = 365 * 24 * 3600 # 1 year in seconds

# Time array to be used in numerical solution 
t = np.arange(0, t_max, dt)

# Initialising arrays to store the positions and velocities at all the time steps
r = np.empty(shape=(len(t), 2))
v = np.empty(shape=(len(t), 2))

# Set the Initial conditions for postition and velocity
r[0], v[0] = r_0, v_0

# Define the functions that gets us the acceleration vector when passed in the position vector
def acceleration(r):
    return -G * M_sun * r / np.linalg.norm(r) ** 3

# Implementing the Euler method
def euler_method(r, v, acceleration, dt):
    '''
    Equatons for Euler Method:
    ODE for Position of the Earth: 
    --> dr/dt = v
    --> r_new = r_old + v_old * dt

    ODE for Velocity of the Earth:
    --> dv/dt = a
    --> v_new = v_old + acceleration(r_old) * dt

    Parameters
    ----------
    r: empty array for position of size t
    v: empty array for velocity of size t
    acceleration: function to calculate the acceleration at a given position
    dt: time step for the simulation

    This function will update the empty arrays r and v with the new position and velocity at each time step 
    '''

    for i in range(1, len(t)):
        r[i] = r[i - 1] + v[i - 1] * dt
        v[i] = v[i - 1] + acceleration(r[i - 1]) * dt

# Apply the Euler Integration on the given initial conditions
euler_method(r, v, acceleration, dt)

# Find the point at which the Earth is at its Apogee
sizes = np.array([np.linalg.norm(position) for position in r])
pos_at_apogee = np.max(sizes)
arg_max_size = np.argmax(sizes)
vel_at_apogee = np.linalg.norm(v[arg_max_size])

print(f"Apogee Position: {pos_at_apogee/1e9} million km, Apogee Velocity: {vel_at_apogee/1e3} km/s")