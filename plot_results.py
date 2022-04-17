import numpy as np
import matplotlib.pyplot as plt
import math
import gif
import multiprocessing
from IPython.display import clear_output

G = 0.004491


# Plot the Results projected into the x-y-plane
@gif.frame
def plot_positions(i, positions):
    """Create a frame of the positions projected into the x-y-plane at one time"""
    x_positions = positions[0, :]
    y_positions = positions[1, :]
    z_positions = positions[2, :]
    plt.figure(figsize=(20,20))
    plt.scatter(x_positions, y_positions, s=8)
    plt.xlim((-2e6, 2e6))
    plt.ylim((-2e6, 2e6))
    

@gif.frame
def plot_density(n, positions, masses, bins):
    """Create a frame of the density profile at one point in time"""

    origin = np.array([0,0,0])
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = absolute(distance(origin, positions[:,i]))
        
    rho = density(n, radii, masses, bins)
    
    plt.figure(figsize=(10,8))
    plt.xscale('log');plt.yscale('log')
    plt.plot(bins[1:], rho[1:])
    plt.show()
    
    
def distance(p_1, p_2):
    """vector from p_1 towards p_2:"""
    return p_2 - p_1


def absolute(v):
    """Absolute of a 3-vector."""
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def potential_energy(m_i, m_j, d_ij, softening = 0):
    """Gives the gravitational potential energy between particles i and j."""
    return G * m_i * m_j / (absolute(d_ij) + softening)


def single_thread(i, n, masses, positions, softening=0):
    """This version of single thread just lets the starmap do more of the work"""
        
    potential_of_i = np.zeros(n) # stores all the contributions to the potential of i
    for j in range(i+1, n):
        d_ij = distance(positions[:,i], positions[:,j])
        potential_of_i[j] = potential_energy(masses[i], masses[j], d_ij, softening)


    total_potential_of_i = np.sum(potential_of_i)
    return total_potential_of_i
    
    
def pooled_potential_energy(n, p, masses, positions, softening = 0):
    """Pools the full particle population to the individual CPUs"""
    
    input_array = np.empty(n, dtype=object) # array that stores the arguments passed tosingle  thread
    
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
        potential_array = pool.starmap(single_thread, input_array)
        pool.close()
        
    return np.sum(potential_array)
    

def kinetic_energy(masses, velocities):
    """gives the total kinetic energy"""
    T_array = np.zeros(n)
    
    for i in range(n):
        T_array[i] = 0.5 * masses[i] * absolute(velocities[:,i])**2
    return np.sum(T_array)


def partial_volume(r_1, r_2):
    """Volume of a partial sphere between R_1 and R_2"""
    return 4/3 * np.pi * (r_2 - r_1)
    
    
def density(n, radii, masses, bins):
    """Computes the total mass of all objects within single volume bins"""
     # stores the bin-number for each particle
    #print(np.max(digits))

    mass_per_cell = np.zeros(len(bins))
    volumes = np.zeros(len(bins))
    bins = np.concatenate([np.array([0]), bins])
    digits = np.digitize(radii, bins, right=False)
    for i in range(len(volumes)):
        volumes[i] = partial_volume(bins[i], bins[i+1])
        
    for j in range(n):
        mass_per_cell[digits[j]] = mass_per_cell[digits[j]] + masses[j]
        
    density = mass_per_cell / volumes
        
    return density
    
    
def NFW_profile(r, rho_0, r_s):
    return rho_0 / (r/r_s * (1 + r/r_s)**2)
    

if __name__ == "__main__":
    
    plot_energies = False
    make_position_gif = False
    make_density_gif = True
    plot_the_density = False
    
    filepath = 'DCDM/Gamma01_epsilon01'

    T = 2000
    fraction = 50 # fraction of timesteps to plot
    n = 10000
    p = 56
    softening = 1e4
    M = 1e15
    
    if plot_energies == True:
        """This section plots the energies (takes a long time)"""
        U_total = np.zeros(int(T/fraction))
        T_total = np.zeros(int(T/fraction))
        
        for t in range(int(T/fraction)):
            Positions = np.loadtxt(f'{filepath}/Positions/k{t*fraction}positions.txt')
            Velocities= np.loadtxt(f'{filepath}/Velocities/k{t*fraction}velocities.txt')
            Masses = np.loadtxt(f'{filepath}/Masses/k{t*fraction}masses.txt')
    
            Utotal_t = pooled_potential_energy(n, p, Masses, Positions, softening)
            U_total[t] = Utotal_t
            T_total[t] = kinetic_energy(Masses, Velocities)
        
            print(t)
        
        print(U_total)
        print(T_total)
        np.savetxt(f'{filepath}/potential_energy.txt', U_total)
        np.savetxt(f'{filepath}/kinetic_energy.txt', T_total)
        
        timesteps = np.array(range(int(T/fraction))) * fraction
        plt.figure(figsize=(8,10))
        plt.plot(timesteps, U_total, label='U')
        plt.plot(timesteps, 2 * T_total, label='2T')
        plt.savefig(f'{filepath}/Output/energies.pdf')
    
    if make_position_gif == True:
        """creates a nice little gif of the simulation evolving over time"""
        fraction = 10
        frames = []
        for t in range(int(T/fraction)):
            Positions = np.loadtxt(f'{filepath}/Positions/k{t*fraction}positions.txt')
    
            frame_pos = plot_positions(t*fraction, Positions)
            frames.append(frame_pos)
            clear_output(wait=True)
            print(f'progress: {round((t+1)*fraction/(T) * 100, 3)}%')
        
        gif.save(frames, f'{filepath}/Output/n_body_system.gif', duration=30, unit='s', between='startend')
    
    if make_density_gif == True:
        fraction = 100
        
        bins = np.logspace(3.5, 11, 100)
        frames = []
        for t in range(int(T/fraction)):
            Positions = np.loadtxt(f'{filepath}/Positions/k{t*fraction}positions.txt')
            Masses = np.loadtxt(f'{filepath}/Masses/k{t*fraction}masses.txt')
        
            frame_dens = plot_density(n, Positions, Masses, bins)
            frames.append(frame_dens)
            clear_output(wait=True)
            print(f'progress: {round((t+1)*fraction/(T) * 100, 3)}%')
        
        gif.save(frames, f'{filepath}/Output/density_evolution.gif', duration=20, unit='s', between='startend')
    
    if plot_the_density == True:
        """Plots the density against radius and adds some generic NFW profiles to it for comparison"""
        bins = np.logspace(3.5, 11, 100)
        rho_0 = 1e7
        R_s_MW = 14.4e3
        profiles = []
        for R_s in np.logspace(5, 9, 10):
            NFW = NFW_profile(bins, rho_0, R_s)
            profiles.append(NFW)
        
        Positions = np.loadtxt(f'{filepath}/Positions/k1999positions.txt')
        Masses = np.loadtxt(f'{filepath}/Masses/k1999masses.txt')
        origin = np.array([0,0,0])
        radii = np.zeros(n)
        for i in range(n):
            radii[i] = absolute(distance(origin, Positions[:,i]))
        relaxed_density = density(n, radii, Masses, bins)
        
        plt.figure(figsize=(10,8))
        plt.xscale('log');plt.yscale('log')
        plt.plot(bins, relaxed_density)
        for i in range(len(profiles)):
            plt.plot(bins, profiles[i], label=f'log(rho_0) = 7, log(R_s) = {np.log10(np.logspace(5,     10, 10)[i])}')
        plt.legend()
        plt.savefig(f'{filepath}/Output/density_profile.pdf')

        
        #U_total = np.loadtxt(f'{filepath}/potential_energy.txt')
        #T_total = np.loadtxt(f'{filepath}/kinetic_energy.txt')
        #timesteps = np.array(range(int(T/fraction))) * fraction
        #plt.figure(figsize=(8,10))
        #plt.plot(timesteps, U_total, label='U')
        #plt.savefig(f'{filepath}/Output/energies.pdf')
       

    
    
