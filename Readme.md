## Qutip showcase for molecular quantum logic spectroscopy

These are som examples how qutip has been used to simulate results for time-domain quantum logic spectroscopy with molecular ions:

https://iopscience.iop.org/article/10.1088/1367-2630/ab3549

The repository consists of following files:

## detection_efficiency.py

Calculates the detection efficiency of absorbing a single photon on a molecule that is co-trapped with an ion. The simulation covers mainly the motional states of both particles. It uses modified destroy and create operatros to include the reduced sideband coupling coupling strength for large phonon numbers.

## molecule_rotation_qutip.py

Simulates rotational dephasing in Ramsey experiments. This simulation modifies the create and destroy operators to simulate the energy spectrum of a rotational degree of freedom.

## vibrational_excitation_qutip.py

Combines results from DFT simulations with Qutip. It simulates the response of vibrational transitions to ultrashort laser pulses.

## moltools.py

Utility functions specific to molecules.
