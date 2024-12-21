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
print(t)

# Initialising arrays to store the positions and velocities at all the time steps
r = np.empty(shape=(len(t), 2))
v = np.empty(shape=(len(t), 2))

# Set the Initial conditions for postition and velocity
r[0], v[0] = r_0, v_0

# Define the functions that gets us the acceleration vector when passed in the position vector
def acceleration(r):
    return -G * M_sun * r / np.linalg.norm(r) ** 3

print(acceleration(r[0]))