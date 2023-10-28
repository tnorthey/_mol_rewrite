"""
Test sa.py module

The following functions are tested:
$ grep "def " modules/sa.py 
def __init__(self):
def read_nm_displacements(self, fname, natoms):
def uniform_factors(self, nmodes, displacement_factors):
def displacements_from_wavenumbers(self, wavenumbers, step_size, exponential=False):
def simulate_trajectory(
def atomic_pre_molecular(self, atomic_numbers, qvector, aa, bb, cc, electron_mode=False):
def simulated_annealing_modes_ho(
    def get_angle_3d(a, b, c):
"""

import numpy as np
import os

# my own modules
import modules.mol as mol
import modules.x as xray
import modules.sa as sa

# create class objects
m = mol.Xyz()
x = xray.Xray()
sa = sa.Annealing()

#############################
### Initialise some stuff ###
#############################
# read test.xyz (perfectly linear H-O-H with exactly 1 Angstrom OH bonds)
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
natoms = len(atomlist)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]

# qvector
qlen = 241
qvector = np.linspace(1e-9, 24, qlen, endpoint=True)
#################################
### End Initialise some stuff ###
#################################


def test_simulated_annealing_modes_ho():
    """ test the simulated annealing function ... """
    (
        chi2_best,
        predicted_best,
        xyz_best,
        chi2_array,
        chi2_xray_best,
    ) = sa.simulated_annealing_modes_ho(
        atomlist,
        starting_xyz,
        reference_xyz,
        displacements,
        mode_indices,
        target_function,
        qvector,
        step_size_array,
        ho_indices1,
        ho_indices2,
        aho_indices1,
        aho_indices2,
        aho_indices3,
        starting_temp=0.2,
        nsteps=10000,
        inelastic=True,
        af=1,  # HO factor
        af2=1,  # angular HO factor
        pcd_mode=False,
        q_mode=False,
        electron_mode=False,
    )
