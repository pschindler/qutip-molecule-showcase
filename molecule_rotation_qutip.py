import numpy as np
from numpy import pi
import scipy
import scipy.sparse as sp
from qutip.qobj import Qobj
from qutip.fastsparse import fast_csr_matrix, fast_identity
from qutip import *
from scipy.sparse import (_sparsetools, isspmatrix, isspmatrix_csr, diags,
                          csr_matrix, coo_matrix, csc_matrix, dia_matrix)
import pylab as pl
import matplotlib.pyplot as plt
import scipy.constants as const
import copy

import seaborn as sns

sns.set_context("talk")

"""Quantum simulation of rotational molecules
All units in s^-1 
"""


def my_hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def rot_destroy(N, offset=0):
    '''Destruction (lowering) operator for molecular rotation.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Qobj for lowering operator.

    Examples
    --------

    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    # data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    j = np.arange(offset + 1, N + offset, dtype=complex)
    data = np.sqrt(j * (j + 1))
    ind = np.arange(1, N, dtype=np.int32)
    ptr = np.arange(N + 1, dtype=np.int32)
    ptr[-1] = N - 1
    return Qobj(fast_csr_matrix((data, ind, ptr), shape=(N, N)), isherm=False)


def rot_create(N, offset=0):
    '''Creation (raising) operator for molecular rotation.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    qo = rot_destroy(N, offset=offset)  # create operator using destroy function
    return qo.dag()


# old B1: 2*pi*4.564e9
class QMolecule:
    def __init__(self, B0=2 * pi * 4.550e9, B1=2 * pi * 4.673e9, deltaJ=2 * pi * 1e6, N_rot=60,
                 omega_echo=2 * pi * 2.5e9, T=20, wavenum_vib=226, timesteps=10000):
        # Internally we use the rotational constant as unit of time and frequency
        self.B0 = B0
        self.B_ratio = B1 / B0
        self.rel_dJ = deltaJ / B0
        self.N_rot = N_rot
        self.omega_echo = omega_echo / B0
        self.omega_vib = self.cm_to_rot_freq(wavenum_vib)
        self.distortion_const = 4 * self.B0 ** 3 / self.omega_vib ** 2
        self.T = T
        self.timesteps = timesteps
        self.sx0 = tensor(sigmax(), qeye(self.N_rot))

    def cm_to_rot_freq(self, wavenumber):
        freq = wavenumber * const.c * 1e2
        return freq * 2 * pi

    def get_sim_time(self, real_time):
        return real_time * self.B0

    def get_real_time(self, sim_time):
        if type(sim_time) == list:
            sim_time = np.array(sim_time)
        return sim_time / self.B0

    def get_real_freq(self, sim_freq):
        return sim_freq * self.B0 / (2 * pi)

    def thermal_occupation(self):
        """Thermal occupation of state J"""
        h = const.h
        kb = const.k
        c = const.speed_of_light
        p_list = np.zeros(self.N_rot)
        for J in range(self.N_rot):
            p_list[J] = (2 * J + 1) * np.exp(-h * self.B0 * J * (J + 1) / (kb * self.T))
        return p_list / np.sum(p_list)

    def rot_energy(self):
        j = np.arange(self.N_rot)
        return Qobj(diags(j * (j + 1)), isherm=False)

    def rot_difference(self):
        j = np.arange(self.N_rot) * (1 - self.B_ratio)
        return Qobj(diags(j * (j + 1)), isherm=False)

    def free_evol_hamil(self):
        sz = tensor(sigmaz(), self.rot_difference())
        return sz

    def echo_hamil(self):
        H_i = self.omega_echo * self.sx0 + self.free_evol_hamil()
        return H_i

    def solve_concatenated_hamil(self, hamil_list, times_list, rho):
        rho0 = copy.deepcopy(rho)
        start_time = 0
        final_times = []
        final_states = []
        for hamil, times in zip(hamil_list, times_list):
            res0 = mesolve(hamil, rho0, times, [], [], options=Options(nsteps=1e4))
            final_times += (times + start_time).tolist()
            final_states += res0.states
            rho0 = res0.states[-1]
            start_time += times[-1]
        return final_times, final_states

    def get_expectation_value(self, state_list, exp_operator):
        e_z = expect(tensor(exp_operator, qeye(self.N_rot)), state_list)
        return e_z

    def get_sy_exp(self, state_list):
        e_z = self.get_expectation_value(state_list, sigmay())
        return e_z

    def get_all_sy_exp(self, state_list):
        e_z_list = []
        for i in range(self.N_rot):
            op = Qobj(csr_matrix(([1], ([i], [i])), shape=(self.N_rot, self.N_rot)), isherm=False)
            e_z = expect(tensor(sigmay(), op), state_list)
            e_z_list += [e_z]
        return e_z_list

    def get_branch_frequencies(self):
        J = np.arange(self.N_rot)
        P = -(1 + self.B_ratio) * J + (1 - self.B_ratio) * J ** 2
        Q = (1 - self.B_ratio) * J * (J + 1)
        R = (1 + self.B_ratio) * (J + 1) + (1 - self.B_ratio) * (J + 1) ** 2
        return P, Q, R

    def get_raman_frequencies(self):
        J = np.arange(self.N_rot)
        P = - 6 * self.B0 - 4 * self.B0 * J
        Q = 0
        R = 6 * self.B0 + 4 * self.B0 * J
        return P, R

    def plot_raman(self):
        branch_list = self.get_raman_frequencies()
        int_list = self.thermal_occupation()
        #print(np.argmax(int_list))
        plt.figure()
        clr = ['r', 'g', 'b', 'k']
        q_max_idx = np.argmax(int_list)
        q_max_freq = self.get_real_freq(branch_list[1][q_max_idx]) / 1e9
        omega = self.get_real_freq(self.omega_echo) / 1e9
        for idx, branch in enumerate(branch_list):
            sum_all = 0
            for idx1, freq in enumerate(branch):
                try:
                    d = int_list[idx1 + 2 * idx]
                except IndexError:
                    d = 0
                d = d * (-1) ** idx
                #print(idx, idx1, idx1 + 2 * idx, d)
                freq = np.abs(freq / (2 * pi * 1e9))  # + idx * 5
                plt.plot([freq, freq], [0, d], clr[idx], linewidth=4)
                sum_all += d * omega ** 2 / (omega ** 2 + (freq - q_max_freq) ** 2)
            #print(sum_all)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Occupation')
        plt.xlim([0, 700])
        plt.tight_layout()
        plt.savefig('rotational_raman.pdf')
        plt.show()

    def plot_branches(self):
        branch_list = self.get_branch_frequencies()
        int_list = self.thermal_occupation()
        plt.figure()
        clr = ['r', 'g', 'b', 'k']
        q_max_idx = np.argmax(int_list)
        freq = branch_list[1][q_max_idx] * 2
        q_max_freq = self.get_real_freq(freq) / 1e9
        omega = self.get_real_freq(self.omega_echo) / 1e9
        for idx, branch in enumerate(branch_list):
            sum_all = 0
            for idx1, freq in enumerate(branch):
                d = int_list[idx1]
                freq = self.get_real_freq(freq) / 1e9
                plt.plot([freq, freq], [0, d], clr[idx], linewidth=4)
                sum_all += d * omega ** 2 / (omega ** 2 + (freq - q_max_freq) ** 2)
            print(sum_all)
        x_freq = np.linspace(min(branch_list[0]), max(branch_list[2]), 1000)
        # plt.plot(x_freq,  0.05 * omega**2/(omega**2+(x_freq-q_max_freq)**2),clr[-1])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Occupation')
        plt.xlim([-300, 300])
        plt.tight_layout()
        plt.savefig('rotational_branches.pdf')
        plt.show()

    def get_thermal_rho(self, superposition=True, even_rot_states=False):
        if even_rot_states:
            rot_occupation = np.ones(self.N_rot)
        else:
            rot_occupation = self.thermal_occupation()
        rho_rot = Qobj(diags(rot_occupation))
        rho = tensor(ket2dm(basis(2, 0)), rho_rot)
        if superposition:
            Ux = (1j * np.pi / 4 * my_mol.sx0).expm()
            rho = Ux * rho * Ux.dag()
        return rho

    def get_echo_evolution(self, wait_time, timing_error=0):
        timing_error = self.get_sim_time(timing_error)
        times_deph = np.linspace(0, self.get_sim_time(wait_time) / 2 - timing_error / 2, self.timesteps)
        times_deph1 = np.linspace(0, self.get_sim_time(wait_time) / 2 + timing_error / 2, self.timesteps)
        times_echo = np.linspace(0, 1 / 2. * np.pi / self.omega_echo, self.timesteps)
        ham_list = [self.free_evol_hamil(), self.echo_hamil(), self.free_evol_hamil()]
        times_list = [times_deph, times_echo, times_deph1]
        rho0 = self.get_thermal_rho()
        final_times, final_states = self.solve_concatenated_hamil(ham_list, times_list, rho0)
        exp_y = self.get_sy_exp(final_states)

        rho0 = self.get_thermal_rho(superposition=False)
        final_times, final_states = self.solve_concatenated_hamil(ham_list, times_list, rho0)
        exp_z = self.get_expectation_value(final_states, sigmaz())
        fid = 1 - ((exp_y[-1] + exp_z[-1]) + 2) / 4

        return self.get_real_time(final_times), exp_y, exp_z, fid

    def get_no_echo_evolution(self, wait_time):
        times_deph = np.linspace(0, self.get_sim_time(wait_time), self.timesteps)
        ham_list = [self.free_evol_hamil()]
        times_list = [times_deph]
        rho0 = self.get_thermal_rho()
        final_times, final_states = self.solve_concatenated_hamil(ham_list, times_list, rho0)
        exp_y = self.get_sy_exp(final_states)
        return self.get_real_time(final_times), exp_y

    def get_echo_pulse(self):
        times_echo = np.linspace(0, 1 / 2. * np.pi / self.omega_echo, self.timesteps)
        ham_list = [self.echo_hamil()]
        times_list = [times_echo]
        rho0 = self.get_thermal_rho(even_rot_states=False)
        final_times, final_states = self.solve_concatenated_hamil(ham_list, times_list, rho0)
        exp_y = self.get_sy_exp(final_states)

        rho0 = self.get_thermal_rho(superposition=False)
        final_times, final_states = self.solve_concatenated_hamil(ham_list, times_list, rho0)
        exp_z = self.get_expectation_value(final_states, sigmaz())
        fid = 1 - ((exp_y[-1] + exp_z[-1]) + 2) / 4
        return self.get_real_time(final_times), exp_y, exp_z, fid

    def get_echo_fidelity(self, temp_list=None):
        return_list = []
        if temp_list == None:
            temp_list = np.linspace(3, 72, 10)
        old_temp = self.T
        old_omega = self.omega_echo
        plt.figure()
        for omega_echo in [old_omega, 2 * old_omega]:
            fid_list = []
            for temp in temp_list:
                #print(temp)
                self.T = temp
                self.omega_echo = omega_echo
                #                a,b,c,fid = self.get_echo_pulse()
                a, b, c, fid = self.get_echo_evolution(1e-7)
                fid_list += [fid]
            plt.plot(temp_list, fid_list)
            return_list += fid_list
        self.T = old_temp
        self.omega_echo = old_omega
        plt.xlabel('Temperature (K)')
        plt.ylabel('Process fidelity')
        plt.tight_layout()
        plt.savefig('echo_fidelity.pdf')
        plt.show()
        np.savetxt('retturn_list', return_list)
        return return_list

    def plot_time_evol(self):
        wait_time = 1e-6
        t_no, e_no = self.get_no_echo_evolution(wait_time)
        t_echo, e_echo, c, fid  = self.get_echo_evolution(wait_time)
        plt.figure()
        plt.plot(t_no * 1e9, e_no)
        plt.plot(t_echo * 1e9, e_echo)
        plt.xlabel('time (ns)')
        plt.ylabel('Coherence')
        plt.tight_layout()
        plt.savefig('time_evolution.pdf')
        plt.show()

    def get_timing_error(self, wait_time=1e-6):
        timing_error_list = np.logspace(-12, -10, 20)
        exp_list = []
        for idx, timing_error in enumerate(timing_error_list):
            #print(idx)
            times, exp_y, exp_z, fid = self.get_echo_evolution(wait_time, timing_error=timing_error)
            exp_list.append(fid)
        plt.figure()
        plt.plot(timing_error_list, np.abs(exp_list))
        plt.xscale('log')
        plt.xlabel('Timing error (s)')
        plt.ylabel('Coherence')
        plt.tight_layout()
        plt.savefig('timing_error.pdf')
        return timing_error_list, exp_list


if __name__ == '__main__':
    my_mol = QMolecule(T=300.0, N_rot=30) # B1=2 * pi * 4.55e9)
    #my_mol.plot_branches()
    #my_mol.plot_raman()
    my_mol.plot_time_evol()
    my_mol.get_echo_fidelity()
    #my_mol.get_timing_error()
    # echo_times, exp_echo = my_mol.get_echo_evolution(1e-6)
    # no_echo_times, exp_no_echo = my_mol.get_no_echo_evolution(1e-6)
    # plt.figure()
    # plt.plot(echo_times*1e9, exp_echo)
    # plt.plot(no_echo_times*1e9, exp_no_echo)
    # plt.show()
    # # echo_times, exp_echo = my_mol.get_echo_pulse()

    # How does the phase of the superposition evolve?
    # Calculate the phase difference between the electronic states
    # No phase difference for dJ=0 transition
    # Phase difference of superposition due to wave-packet generation of initial pulse?
    # If initial pulse would be dJ=0, no rotational dephasing would be observed? - What about the freq diff in P branch?

    # What wavepacket is generated by the first pulse? And how does this affect the coherence?

    # Include P,R branch into echo pulse. 

    # Add coriolis coupling, centrifugal distortion - how is this affecting the coherence? Does only the diff in distortion constants matter? D=4B**3/omega_vib**2

    # Process fidelity of echo pulse
    # Analyze echo with Q brnach. Each line!
    # Spin echo with PQR branches
