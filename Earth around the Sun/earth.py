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

You can symply check the facts about the Earth's orbit around the Sun at this website: 
https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np

# Constants
G = 6.67430e-11 # m^3 kg^-1 s^-2
M_sun = 1.989e30 # kg

# Initial Position and Velocity
r_0 = np.array([147.095e9, 0]) # m
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

# # Apply the Euler Integration on the given initial conditions
# euler_method(r, v, acceleration, dt)

# # Find the point at which the Earth is at its Apogee
# sizes = np.array([np.linalg.norm(position) for position in r])
# pos_at_apogee = np.max(sizes)
# arg_max_size = np.argmax(sizes)
# vel_at_apogee = np.linalg.norm(v[arg_max_size])

# print(f"Apogee Position: {pos_at_apogee/1e9} million km, Apogee Velocity: {vel_at_apogee/1e3} km/s")

# RK4 Integration
def rk4_method(r, v, acceleration, dt):
    '''
    Equatons for Euler Method:
    ODE for Position of the Earth: 
    --> dr/dt = v
    --> r_new = r_old + dt / 6 (k1r + 2k2r + 2k3r + k4r)

    ODE for Velocity of the Earth:
    --> dv/dt = a
    --> v_new = v_old + dt / 6 (k1v + 2k2v + 2k3v + k4v)

    Method to calculate the k values
    -------------------------------
    Step 1:
    k1v = acceleration(r_old)
    k1r = v_old

    Step 2: dt/2 using step 1
    k2v = acceleration(r_old + k1r * dt / 2)
    k2r = v_old + k1v * dt / 2

    Step 3: dt/2 using step 2
    k3v = acceleration(r_old + k2r * dt / 2)
    k3r = v_old + k2v * dt / 2

    Step 4: dt/2 using step 3
    k4v = acceleration(r_old + k3r * dt)
    k4r = v_old + k3v * dt

    Parameters
    ----------
    r: empty array for position of size t
    v: empty array for velocity of size t
    acceleration: function to calculate the acceleration at a given position
    dt: time step for the simulation

    This function will update the empty arrays r and v with the new position and velocity at each time step 
    '''

    for i in range(1, len(r)):
        # Step 1
        k1v = acceleration(r[i - 1])
        k1r = v[i - 1]

        # Step 2    
        k2v = acceleration(r[i - 1] + k1r * dt / 2)
        k2r = v[i - 1] + k1v * dt / 2

        # Step 3
        k3v = acceleration(r[i - 1] + k2r * dt / 2)
        k3r = v[i - 1] + k2v * dt / 2

        # Step 4
        k4v = acceleration(r[i - 1] + k3r * dt)
        k4r = v[i - 1] + k3v * dt

        # Update the position and velocity arrays
        r[i] = r[i - 1] + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v[i] = v[i - 1] + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

# Apply the RK4 Integration on the given initial conditions
rk4_method(r, v, acceleration, dt)

# Find the point at which the Earth is at its Apogee
sizes = np.array([np.linalg.norm(position) for position in r])
pos_at_apogee = np.max(sizes)
arg_max_size = np.argmax(sizes)
vel_at_apogee = np.linalg.norm(v[arg_max_size])

print(f"Apogee Position: {pos_at_apogee/1e9} million km, Apogee Velocity: {vel_at_apogee/1e3} km/s")