from qutip import *
# from pylab import *
import scipy.constants as const
import numpy as np

try:
    from scipy.special import eval_genlaguerre as laguerre
except ImportError:
    from scipy.special.orthogonal import eval_genlaguerre as laguerre

"""
Simulates generation and detection efficiency of a cat state outside of the Lamb Dicke regime
including a phenomenological model for decoherence effects.

The dynamics of creating the cat state is simulated by qutip with custom destroy and create operators
that take the reduction of the coupling strength with increasing phonon number into account. 

The functions from this script are called from a script that reads parameters from DFT simulations.
"""


def cm_to_hz(wavenumber):
    """Convert wavenumber to Hz"""
    freq = wavenumber * const.c * 1e2
    return freq


def simulate_cat_state(eta_ir, eta_logic, name, w_t=1. * 2 * np.pi, Omega_car=0.3 * 2 * np.pi,
                       angle_list=np.linspace(0, np.pi / 2 - np.pi / 40, 20), N_motional_states=5000,
                       Rh_list=[1e-5, 1e-6, 1e-7], do_save=True):
    """
    Simulate a cat state detection for following parameters:
    eta_ir: Lamb-Dicke Parameter for the molecular transition
    eta_logic: Lamb-Dicke Parameter for the atomic transition
    name: Identifier
    w_t: Trap frequency in 1/us (angular frequency)
    Omega_car: Carrier Rabi frequency in 1/us
    angle_list: Angle of incidence for atomic laser beam
    N_motional_states: Number of simulated motional states
    Rh_list: List of simulated heating rates in phonons/us
    do_save: Save results in npy files
    """
    # Calculate the effective Lamb-Dicke parameters for all given beam angles
    eta_list = eta_logic * np.cos(angle_list)

    def get_coherence(alpha, time, heating_rate):
        """Calculate remaining coherence for a given state preparation"""
        phi2 = 8 * heating_rate * alpha ** 2 * 2 * time / 3.
        return np.exp(-phi2 / 2)

    def get_det(alpha, time, heating_rate):
        """Calculate effective detection efficiency by reducing the ideal detection efficiency by the coherence"""
        coh = get_coherence(alpha, time, heating_rate)
        det_eff = coh * np.sin(2 * alpha * eta_ir) ** 2
        return det_eff

    def exp_p(t, args):
        """Positive rotating term of bichro laser field"""
        return np.exp(1j * w_t * t)

    def exp_m(t, args):
        """Negative rotating term of bichro laser field"""
        return np.exp(-1j * w_t * t)

    # Define the standard operators for motional and electronic states
    sx = tensor(sigmax(), qeye(N_motional_states))
    sp = tensor(sigmap(), qeye(N_motional_states))
    a = tensor(qeye(2), destroy(N_motional_states))
    x = 1 / np.sqrt(2.) * (a + a.dag())
    ex1 = 1 / np.sqrt(2.) * tensor(sigmax(), (destroy(N_motional_states) + destroy(N_motional_states).dag()))

    # Define the empty lists to store the results
    c_list = []
    max_list = []

    # Loop over given Lamb Dicke parameters
    for eta in eta_list:
        Omega = Omega_car * eta
        # Define the coupling strength beyond the Lamb Dicke approximation
        coupling_func = lambda n: np.exp(-1. / 2 * eta ** 2) * (1. / (n + 1.)) ** 0.5 * laguerre(n, 1, eta ** 2)
        aa = destroy(N_motional_states)
        # Define an extended destroy operator with the extended coupling strength
        for i in range(N_motional_states - 1):
            aa.data[i, i + 1] = coupling_func(i)
        aa = tensor(qeye(2), aa)
        # Define initial state
        psi0 = tensor([1 / np.sqrt(2) * (basis(2, 0) + basis(2, 1)), fock(N_motional_states, 2)])

        # Define Hamilonian with state dependent force
        H_i = Omega / 2 * sx * (aa + aa.dag())
        # H_c = Omega/2/eta * tensor(sigmap(),qeye(N_motional_states))
        times = np.linspace(0, 3e3, 100)
        #    res0 = mesolve([H_i, [H_c, exp_m], [H_c.dag(), exp_p]], psi0, times, [], [], options=Options(nsteps=1e4))
        # Run simulation
        res0 = mesolve(H_i, psi0, times, [], [], options=Options(nsteps=1e4))
        states = res0.states  # + res.states
        # Calculate expectation values
        c = [expect(1 / np.sqrt(2) * tensor(destroy(N_motional_states).dag() - destroy(N_motional_states)), p.ptrace(1))
             for p in states]
        c_list.append(np.abs(c))

        # Add phenomenological Heating rate model and find maximum detection efficiency
        for Rh in Rh_list:
            d = []
            for alpha, t in zip(np.abs(c), times):
                d.append(get_det(alpha, t, Rh))
            max_det = np.max(d)
            max_disp = np.max(np.abs(c))
            max_time = times[np.argmax(np.abs(c))]
            max_list.append([Omega, eta, Rh, max_det, max_time, max_disp])
    if do_save:
        np.save('../data/max_list_nocar_ld' + name, max_list)
        np.save('../data/c_list_nocar_ld' + name, c_list)
    else:
        return max_list


if __name__ == '__main__':
    eta_logic = 0.1  # LD parameter for the logic ion
    eta_ir = 0.01  # LD parameter for the IR transition
    angle_list = [0]  # Angle of incidence for the logic lasers
    heating_rates = [1e-6]  # Heating rate in us
    print('Calculating cat state. This might take a while')
    simulation_results = simulate_cat_state(eta_ir, eta_logic, 'test_run', angle_list=angle_list,
                                            do_save=False, Rh_list=heating_rates)

    print(f'Maximum detection efficiency: {simulation_results}')
