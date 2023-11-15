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
# qvector
qlen = 241
qvector = np.linspace(1e-9, 24, qlen, endpoint=True)
inelastic = True

def xyz2iam(xyz, atomlist):
    """convert xyz file to IAM signal"""
    electron_mode = False  # x-rays
    atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
    compton_array = x.compton_spline(atomic_numbers, qvector)
    iam, atomic, molecular, compton = x.iam_calc(
        atomic_numbers, xyz, qvector, electron_mode, inelastic, compton_array
    )
    return iam

# define target_data
start_xyz_file = "xyz/chd_opt.xyz"
reference_xyz_file = "xyz/chd_opt.xyz"
target_xyz_file = "xyz/target.xyz"
_, _, atomlist, starting_xyz = m.read_xyz(start_xyz_file)
_, _, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
_, _, atomlist, target_xyz = m.read_xyz(target_xyz_file)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
starting_iam = xyz2iam(starting_xyz, atomlist)
reference_iam = xyz2iam(reference_xyz, atomlist)
target_iam = xyz2iam(target_xyz, atomlist)

noise_bool = False
noise = 4
### ADDITION OF RANDOM NOISE
if noise_bool:
    mu = 0  # normal distribution with mean of mu
    sigma = noise
    noise_array = sigma * np.random.randn(qlen) + mu
    target_iam += noise_array
###

target_data = 100 * (target_iam / reference_iam - 1)

nmfile = "nm/chd_normalmodes.txt"
natoms = starting_xyz.shape[0]
displacements = sa.read_nm_displacements(nmfile, natoms)
nmodes = displacements.shape[0]

# mode_indices = np.arange(0, 28)  # CHD, this removes hydrogen modes
mode_indices = np.arange(0, nmodes)  # CHD, all modes
print("including modes:")
print(mode_indices)

step_size_array = 0.01 * np.ones(nmodes)

# CHD specific; notably not indices 0, 5 (the ring-opening)
ho_indices = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]  # chd specific!

starting_temp = 0.2
nsteps = 20
harmonic_factor = 0.1
pcd_mode = True
electron_mode = False  # x-rays
xyz_save = True


#################################
### End Initialise some stuff ###
#################################


def test_simulated_annealing_modes_ho():
    """test the simulated annealing function ..."""
    (
        f_best,
        f_xray_best,
        predicted_best,
        xyz_best,
        f_array,
        xyz_array,
    ) = sa.simulated_annealing_modes_ho(
        atomlist,
        starting_xyz,
        reference_xyz,
        displacements,
        mode_indices,
        target_data,
        qvector,
        step_size_array,
        ho_indices,
        starting_temp,
        nsteps,
        inelastic,
        harmonic_factor,
        pcd_mode,
        electron_mode,
        xyz_save,
    )
    # it outputs xyz correctly (correct shape)
    assert xyz_best.shape == starting_xyz.shape, "xyz_best.shape != starting_xyz.shape"
    assert (
        f_best >= f_xray_best
    ), "total target function should be greater (or equal) than x-ray component"
    assert (
        predicted_best.shape == target_data.shape
    ), "predicted_best.shape != target_data.shape"
    assert len(f_array) <= nsteps, "len(f_array) !<= nsteps"
    assert xyz_array.shape[2] == nsteps, "xyz_array.shape[2] != nsteps"
