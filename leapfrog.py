import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate
import random
import multiprocessing
from IPython.display import clear_output
import time
from shutil import copytree



# constants in pc-M_sol-Ma units
G = 0.004491

def distance(p_1, p_2):
    """vector from p_1 towards p_2:"""
    return p_2 - p_1


def absolute(v):
    """Absolute of a 3-vector."""
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def grav_acc(m_j, d_ij, softening = 0):
    """Gives the gravitational acceleration between particles i and j."""
    return G * m_j * d_ij / ((absolute(d_ij) + softening)**3)


def distribute(n,p):
    """Finds out which processor needs to compute which section of the particle population"""
    base, extra = divmod(n,p)
    distribution = np.array([base + (i < extra) for i in range(p)])
    distribution = np.concatenate([np.array([0]), distribution])
    distribution = np.concatenate([distribution, np.array([0])])
    return distribution.flatten()
    
    
def single_thread(i, n, masses, positions, softening=0):
    """This version of single thread just lets the starmap do more of the work"""
        
    accelerations_on_i = np.zeros((3,n)) # stores all the contributions to the acc on i
    for j in range(i):
        # j-loop 1: get half of the accelerations from the j particle on the i particle
        d_ij = distance(positions[:,i], positions[:,j])
        accelerations_on_i[:,j] = grav_acc(masses[j], d_ij, softening)
    for j in range(i+1, n):
        # j-loop 2: get the other half to exclude i==j
        d_ij = distance(positions[:,i], positions[:,j])
        accelerations_on_i[:,j] = grav_acc(masses[j], d_ij, softening)

    total_acceleration_on_i = np.sum(accelerations_on_i, axis=1)
    return total_acceleration_on_i
    

def pooled_acceleration(n, p, masses, positions, softening = 0):
    """Pools the full particle population to the individual CPUs"""
    input_array = np.empty(n, dtype=object) # array that stores the arguments passed to single thread
    
    for i in range(n):
        # every i in this array is for one particle
        input_array_i = np.empty(5, dtype=object)
        input_array_i[0] = i # ID of the particle
        input_array_i[1] = n # total number of particles scalar
        input_array_i[2] = masses # array of all particle masses
        input_array_i[3] = positions # array of positions
        input_array_i[4] = softening # softening scalar
        
        input_array[i] = input_array_i


    with multiprocessing.Pool(p) as pool:
        accelerations_array = pool.starmap(single_thread, input_array)
        pool.close()

    return np.transpose(accelerations_array)
    
    
def get_epsilon(kick_velocity):
    """returns the parameter epsilon from a given kick velocity"""
    c = 306000 # pc/Ma
    return kick_velocity / (kick_velocity + c)
    

def get_kick_velocity(epsilon):
    """returns a kick velocity for a given epsilon"""
    c = 306000 # pc/Ma
    return epsilon * c / (1 - epsilon)
    
    
def new_mass(epsilon):
    """returns the ratio of m_daughter to m_mother"""
    return np.sqrt(1 - 2 * epsilon)
    
    
def random_direction():
    """Generates a random direction in spherical coordinates"""
    theta = np.random.uniform(low = 0.0, high = np.pi)
    phi = np.random.uniform(low = 0.0, high = 2 * np.pi)
    return theta, phi
    

def relative_cart_kick(kick_velocity):
    """Generates a velocity Kick in a random direction and outputs it in cartesian coordinates"""
    theta, phi = random_direction()
    x_velocity = kick_velocity * np.cos(phi) * np.sin(theta)
    y_velocity = kick_velocity * np.sin(phi) * np.sin(theta)
    z_velocity = kick_velocity * np.cos(theta)
    return x_velocity, y_velocity, z_velocity
    
    
def decay(decay_rate, kick_velocity, mass_array, velocity_array, time_step):
    """changes the mass and velocity array according to random decay"""
    epsilon = get_epsilon(kick_velocity)
    daughter_mass_ratio = new_mass(epsilon)
    P = time_step * decay_rate # probability for a particle to decay during dt
    # generate a random number between 0 and 1 for each particle
    R = np.random.uniform(low=0.0, high=1.0, size=(len(mass_array)))
    for particle, decay_prob in enumerate(R):
        if decay_prob < P: # in this case decay happens
            #print('boom')
            mass_array[particle] = daughter_mass_ratio * mass_array[particle]
            x_kick, y_kick, z_kick = relative_cart_kick(kick_velocity)
            velocity_array[:, particle] = velocity_array[:, particle] + np.array([x_kick, y_kick, z_kick])
    return mass_array, velocity_array
            


if __name__ == "__main__":

    start = time.time()
    
    # number of particles
    n = 10000
    # number of processors
    p = 56
    # number of timesteps
    T = 2000
    # length of one time step (units!)
    t = 5 #Ma
    # accelerations
    accelerations = np.zeros((3, n))
    # masses
    M = 1e15
    masses = np.loadtxt(f'Initial_Setup/initial_masses_{n}_particles_M_{M}.txt')
    # softening parameter (units!)
    softening = 1e4 #pc
    
    # decay parameters
    Gamma = 0.0001 #Ma
    epsilon = 0.0001
    v_kick = get_kick_velocity(epsilon) #pc/Ma
    
    # path where the output should be stored
    filepath = f'/home/billich/LeapfrogNbody/DCDM/Gamma00001_epsilon00001'

    ref_path = f'/home/billich/LeapfrogNbody/DCDM/template_dir' # reference directory with all the needed folders
    copytree(ref_path, filepath) # copys the reference directory and names it according to the output given under filepath

    # initial setup of the system (units!):
    # all txt files contain (3, n)-arrays where n gives the number of particles
    # positions
    positions = np.loadtxt(f'Initial_Setup/relaxed_halo_positions_{n}_particles.txt')
    # velocities
    velocities = np.loadtxt(f'Initial_Setup/relaxed_halo_velocities_{n}_particles.txt')
    
    
    # 0-step velocities, positions and masses
    np.savetxt(f'{filepath}/Velocities/k0velocities.txt', velocities)
    np.savetxt(f'{filepath}/Positions/k0positions.txt', positions)
    np.savetxt(f'{filepath}/Masses/k0masses.txt', masses)
    
    # 0-step accelerations:
    
    accelerations = pooled_acceleration(n, p, masses, positions, softening)
    np.savetxt(f'{filepath}/Accelerations/k0accelerations.txt', accelerations)
    
    # outer loop for every time step:
    for k in range(1, T):
        # update the position for all particles
        positions += t * velocities + .5 * t**2 * accelerations
        # save the new positions array
        np.savetxt(f'{filepath}/Positions/k{k}positions.txt', positions)
    
        #remember the last acceleration
        accelerations_prior = accelerations
    
        # accelerations using pooled workers
        accelerations = pooled_acceleration(n, p, masses, positions, softening)
    
        # save the new accelerations array
        np.savetxt(f'{filepath}/Accelerations/k{k}accelerations.txt', accelerations)
    
        # update the velocities for all particles
        velocities += .5 * t * (accelerations_prior + accelerations)
        # save the new velocities array
        np.savetxt(f'{filepath}/Velocities/k{k}velocities.txt', velocities)
    
        # decay process
        masses, velocities = decay(Gamma, v_kick, masses, velocities, t)
        np.savetxt(f'{filepath}/Masses/k{k}masses.txt', masses)

    
        # progress update
        clear_output(wait=True)
        print(f'progress: {round((k+1)/(T) * 100, 3)}%')
    
    end = time.time()
    
    print(f'Simulation finished. It took {end - start} seconds')
    
