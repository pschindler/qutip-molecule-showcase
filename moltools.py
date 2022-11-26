import numpy as np
import scipy as sp
import scipy.constants as const
import pylab as pl
from scipy.special import factorial


"""Helper functions for molecular constants and coupling strengths
"""

d = 0.5 #Dipole moment for v5 in acetylene
#I = 1e18 #Frequency comb peak power (from Didis NJP)
di_mom = 0.6 #Dipole moment for strong transition
I = 600e-9/0.2e-12*(1/10e-6)**2 #600nJ energy @ 200fs duration
#I = 5e8*(1/10e-6)**2 #Frequency comb peak power (from Didis NJP)
#I = 0.1 * 1e7 # 0.1 W @ rep rate of 1MHz, 1ps pulse duration
delta = 10e12*2*np.pi #10 THz detuning
w0 = 45e12 * 2*np.pi # 1600cm-1
waist = 10e-6 # 10 um spot size
duration = 200e-15
polarizability = 63 / (1e6/(4*np.pi*const.epsilon_0)) * const.physical_constants['Bohr radius'][0]


def clebsch(j1, j2, j3, m1, m2, m3):
    """From qutip
    Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Parameters
    ----------
    j1 : float
        Total angular momentum 1.

    j2 : float
        Total angular momentum 2.

    j3 : float
        Total angular momentum 3.

    m1 : float
        z-component of angular momentum 1.

    m2 : float
        z-component of angular momentum 2.

    m3 : float
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.

    """
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
                factorial(j3 + m3) * factorial(j3 - m3) /
                (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) * factorial(j1 + m1) *
                factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C

def hoenl_london(J, L, is_p=True):
    if is_p == False:
        J = J+1
    return (J+1+L)*(J+1-L)/(J+1)
    
def cm_to_hz(wavenumber):
    """Convert wavenumber to Hz"""
    freq = wavenumber * const.c *1e2
    return freq

def rot_energy(B,D,J):
    """Rotational energy of J+2 -> J"""
    E = 6.0*B*((1+6.*D/B)+4*J*(1+15*D/B)+4*D/B*(9.*J**2+2*J**3))
    return E

def thermal_occupation(B,T,Jmax=1000):
    """Thermal occupation of state J"""
    h = const.h
    kb = const.k
    p_list = np.zeros(Jmax)
    for J in range(Jmax):
        p_list[J] = (2*J+1)*np.exp(-h*B*J*(J+1)/(kb*T))
    return p_list/np.sum(p_list)

def plot_rotations(B,D,T,Jmax=1000,cutoff=1e-3):
    J = np.arange(Jmax)
    E = rot_energy(B,D,J)
    P = thermal_occupation(B,T,Jmax)
    index = np.where(P>cutoff)[0]
    pl.bar(E[index]/1e9,P[index], width=0.1)
    pl.show()
    
def comb_resonances(rep_rate, detuning, no_teeth, start_tooth=1):
    freq_list = []
    for i in no_teeth:
        freq_list.append(rep_rate*(i+start_tooth)+detuning)
        freq_list.append(rep_rate*(i+start_tooth)-detuning)
    return np.array(freq_list)

def get_rabi_freq(I,di_mom0):
    """Calculate Rabi frequency from:
    I: Laser intensity in W/m
    d: dipole moment in debye
    """
    E = np.sqrt(2*I/(const.c*const.epsilon_0))
    d = di_mom0 * const.elementary_charge * const.physical_constants['Bohr radius'][0]
    return d*E/const.hbar

def get_excitation_single_pulse(I_max, t_pulse, di_mom0):
    """Calculates the rotation angle of a resonant pulse in units of pi
    I_Max: Peak laser intensity in W/m
    t_pulse: Pulse length in s
    di_mom0: Dipole moment in Debye
    """
    omega_max = get_rabi_freq(I_max, di_mom0)
    theta = omega_max*np.sqrt(np.pi/np.log(4))*t_pulse
    return theta/np.pi*2 #times two because a complete spin flip is for sin(pi/2)^2

def get_tweezer_frequency(I, di_mom0, waist, delta, freq, mass=40.):
    """Calculate the trap frequency of an optical tweezer in Hz
    freq, delta in cm-1"""
    # EQ 12 in [Grimm 99]
    omega = 2*np.pi*cm_to_hz(freq)
    delta = 2*np.pi*cm_to_hz(delta)
    gamma = get_decay_rate(di_mom0, freq)
    m = mass * const.atomic_mass
    #U_dip = 3*np.pi*const.speed_of_light**2/(2*omega**2) * (gamma/delta - gamma/delta)
    U_dip =  3*np.pi*const.speed_of_light**2/(2*omega**3) * gamma/delta * I
    F_sc =   3*np.pi*const.speed_of_light**2/(2*const.hbar*omega**3) * (gamma/delta)**2 * I
    # FRequency: EQ Page 15 in Grimm 99
    omega_r = np.sqrt(4*U_dip/(m*waist**2))
    return omega_r/(2*np.pi), F_sc/(2*np.pi)

def get_decay_rate(di_mom0, freq):
    """Calculate Gamma for a given dipole moment"""
    #EQ 9 in [Grimm 99]
    freq = 2 * np.pi * cm_to_hz(freq)
    d = di_mom0 * const.elementary_charge * 1e-10 #* const.physical_constants['Bohr radius'][0]
    return freq ** 3 / (3 * np.pi * const.epsilon_0 * const.hbar * const.speed_of_light ** 3) * d ** 2

def get_dipole_force_stark(omega, delta, waist):
    """Calculate dipole force for a given Stark shift given a Rabi frqeuency (in s-1)"""
    force = const.hbar * omega**2/delta * 1/waist
    return force

def get_dipole_force_pol(polarizability, I, waist, w, w0):
    """Calculate dipole dorce given the polarizability"""
    alpha = polarizability * w0**2/(w0**2-w**2)
    F = 1/(2*const.c*const.epsilon_0) * I/waist * alpha
    return F

def get_dipole_force_sat(delta, omega, phi, wavelength):
    sat = 1/2.*(omega/delta)**2
    F = const.hbar*delta*sat*np.cos(phi)/(4+(2*sat*np.sin(phi)))
    return F/wavelength

def calc_force_sat():
    wave = 12e-6
    omega = get_rabi_freq(I,d)
    phi = 0
    F = get_dipole_force_sat(delta,omega,phi, wave)
    print("Rabi freq: ",omega  /1e12 /2 /np.pi)
    print("Detuning: ",delta  /1e12 /2 /np.pi)
    print("Force: ",F)
    print("Momentum: ",F * duration)
    print("Required momentum: 1e-27")
    print("Excitation: ",omega**2/(delta**2+omega**2))
    
def calc_force_pol():
    #omega = get_rabi_freq(I,d)
    force = get_dipole_force_pol(polarizability,I,waist,w0-delta,w0)
    #print("Rabi freq: ",omega/1e6)
    print(("Force: ",force))
    momentum = force * duration
    print(("Momentum: ",momentum))

def calc_force_stark():
    omega = get_rabi_freq(I,d)
    F = get_dipole_force_stark(omega,delta,waist)
    print("Rabi freq: ",omega  /1e12 /2 /np.pi)
    print("Detuning: ",delta  /1e12 /2 /np.pi)
    print("Force: ",F)
    print("Momentum: ",F * duration)
    print("Required momentum: 1e-27")
    print("Excitation: ",omega**2/(delta**2+omega**2))

if __name__ == '__main__':
    print("Stark shift force")
    calc_force_stark()
    print("\n\nSaturation parameter force")
    calc_force_sat()
    print(f"\n\ndecay rate (us): {get_decay_rate(0.2, 3000)/(2*np.pi*1e6)}")
    #trap_freq, F_sc = get_tweezer_frequency(1e11, np.sqrt(0.2), 5e-6, 100, 3000)
    trap_freq, F_sc = get_tweezer_frequency(1e11, np.sqrt(2.8/42.), 5e-6, 100, 6000)
    # I assume that the unit for d is in atomic units
    # Units from nwchem: [atomic units] [(debye/angs)**2] [(KM/mol)] [arbitrary]
    # Do we need to take the sqrt of these?
    print(f"\n\ntrap freq: {trap_freq}")
    print(f"scattering rate: {F_sc}")
    print(get_decay_rate(1.5,23696)/1e8)
    # Test dacay rate from: https://arxiv.org/pdf/physics/0202029.pdf
    # Expect 2.2e8
    omega = get_rabi_freq(1e15, 0.1)
    #omega = get_rabi_freq(1e4, 3)/1e6/2/np.pi
    print(f"-------")
    print(f"Rabi Frequency: {omega}")
    print(f"Rabi Frequency * pulse_length: {omega*200e-15}")
    theta = get_excitation_single_pulse(7e14,200e-15,0.1)
    print(f"-------")
    print(f"Pulse angle in units of pi: {theta}")
