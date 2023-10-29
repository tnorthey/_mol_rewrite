import numpy as np
import sys
import scipy.io
from timeit import default_timer
from numpy.random import random_sample as random
# my modules
import modules.mol as mol
import modules.x as xray
import modules.sa as sa

start = default_timer()

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
    iam, atomic, molecular, compton = x.iam_calc(atomic_numbers, xyz, qvector, electron_mode, inelastic, compton_array)
    return iam

# define target_function
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
    mu = 0		# normal distribution with mean of mu
    sigma = noise
    noise_array = sigma * np.random.randn(qlen) + mu
    target_iam += noise_array
###

target_function = 100 * (target_iam / reference_iam - 1)

q_mode = False
# multiply by q**m optionally
if q_mode:
    print("q_mode = true")
    q_exponent = 0.5
    target_function *= qvector ** q_exponent

# definitions
non_h_indices = [0, 1, 2, 3, 4, 5]

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
nsteps = 8000
af = 0.01
pcd_mode = True
electron_mode = False  # x-rays


#################################
### End Initialise some stuff ###
#################################
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
    ho_indices,
    starting_temp,
    nsteps,
    inelastic,
    af,
    pcd_mode,
    q_mode,
    electron_mode,
)

run_id_ = 0  # define a number to label the start of the output filenames
run_id = str(run_id_).zfill(2)  # pad with zeros

print("writing to xyz... (chi2: %10.8f)" % chi2_xray_best)
chi2_best_str = ("%10.8f" % chi2_xray_best).zfill(12)
m.write_xyz(
    "%s_%s.xyz" % (run_id, chi2_best_str),
    "run_id: %s" % run_id,
    atomlist,
    xyz_best,
)
np.savetxt(
    "%s_%s.dat" % (run_id, chi2_best_str),
    np.column_stack((qvector, predicted_best)),
)


print("%10.8f" % chi2_best)

print("Total time: %3.2f s" % float(default_timer() - start))
