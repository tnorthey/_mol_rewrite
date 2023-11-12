"""
Run the simulated annealing function for CHD
"""
import numpy as np
import sys
import scipy.io
from timeit import default_timer
from numpy.random import random_sample as random

# my modules
import modules.mol as mol
import modules.x as xray
import modules.sa as sa
import modules.gd as gd

start = default_timer()

# create class objects
m = mol.Xyz()
x = xray.Xray()
sa = sa.Annealing()
gd = gd.G()

###################################
# command line arguments
run_id_ = int(sys.argv[1])  # define a number to label the start of the output filenames
start_xyz_file = str(sys.argv[2])
target_xyz_file = str(sys.argv[3])
###################################

#############################
### arguments             ###
#############################
reference_xyz_file = "xyz/chd_reference.xyz"
nsteps = 8000
qmin = 1e-9
qmax = 8.0
qlen = 81
starting_temp = 0.2
step_size = 0.01
harmonic_factor = 0.1  # HO factor
n_trials = 4  # repeats n_trails times, only saves lowest chi2

electron_mode = False  # x-rays
inelastic = True
noise_bool = False
noise = 4
nmfile = "nm/chd_normalmodes.txt"
pcd_mode = True
q_mode = False

# gradient descent parameters
nsteps_gd = 100
step_size_gd = 0.0001

# ho_indices = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]  # chd (C-C bonds)
ho_indices = [
    [0, 1, 2, 3, 4, 6, 12, 5, 5, 0, 0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5, 7, 13, 12, 13, 6, 7, 8, 9, 10, 11],
]  # chd (C-C and C-H bonds)

run_id = str(run_id_).zfill(2)  # pad with zeros
#############################
### end arguments         ###
#############################

### Rarely edit after this...

#############################
### Initialise some stuff ###
#############################
# qvector
qvector = np.linspace(qmin, qmax, qlen, endpoint=True)


def xyz2iam(xyz, atomlist):
    """convert xyz file to IAM signal"""
    electron_mode = False  # x-rays
    atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
    compton_array = x.compton_spline(atomic_numbers, qvector)
    iam, atomic, molecular, compton = x.iam_calc(
        atomic_numbers, xyz, qvector, electron_mode, inelastic, compton_array
    )
    return iam


# define target_function
_, _, atomlist, starting_xyz = m.read_xyz(start_xyz_file)
_, _, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
_, _, atomlist, target_xyz = m.read_xyz(target_xyz_file)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
starting_iam = xyz2iam(starting_xyz, atomlist)
reference_iam = xyz2iam(reference_xyz, atomlist)
target_iam = xyz2iam(target_xyz, atomlist)

### ADDITION OF RANDOM NOISE
if noise_bool:
    mu = 0  # normal distribution with mean of mu
    sigma = noise
    noise_array = sigma * np.random.randn(qlen) + mu
    target_iam += noise_array
###

target_function = 100 * (target_iam / reference_iam - 1)

# multiply by q**m optionally
if q_mode:
    print("q_mode = true")
    q_exponent = 0.5
    target_function *= qvector ** q_exponent

natoms = starting_xyz.shape[0]
displacements = sa.read_nm_displacements(nmfile, natoms)
nmodes = displacements.shape[0]

mode_indices = np.arange(0, nmodes)  # CHD, all modes
print("including modes:")
print(mode_indices)

step_size_array = step_size * np.ones(nmodes)

#################################
### End Initialise some stuff ###
#################################

chi2_best_ = 1e9
for k in range(n_trials):

    if False:
        # this isn't needed as random initial conditions happen in the simulated_annealing function anyway..
        # random perturbation to starting xyz (to vary initial conditions)
        delta_start = 0.01
        a, b = -delta_start, delta_start
        rand_arr = (b - a) * random((natoms, 3)) + a
        xyz_perturbed = xyz + rand_arr

    # Run simulated annealing
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
        harmonic_factor,
        pcd_mode,
        q_mode,
        electron_mode,
    )

    print('chi2_best (SA): %9.8f' % chi2_best)
    # gradient descent...
    starting_xyz_gd = xyz_best
    pcd_mode = True
    # Target function has to be absolute I(q) for gradient descent...
    #target_function_gd = xyz2iam(starting_xyz_gd, atomlist)
    (chi2_best, predicted_best, xyz_best) = gd.gradient_descent_cartesian(
        target_function,
        atomic_numbers,
        starting_xyz_gd,
        qvector,
        nsteps_gd,
        step_size_gd,
        pcd_mode,
        reference_iam,
    )
    print('chi2_best (GD): %9.8f' % chi2_best)
    #print("%10.8f" % chi2_best)

    # store best values from the n_trials
    if chi2_best < chi2_best_:
        chi2_best_, chi2_xray_best_, predicted_best_, xyz_best_ = (
            chi2_best,
            chi2_xray_best,
            predicted_best,
            xyz_best,
        )

# calculate raw IAM data
iam_best = xyz2iam(xyz_best_, atomlist)

# save final IAM signal
np.savetxt("tmp_/%s_iam_best.dat" % run_id, np.column_stack((qvector, iam_best)))

print("writing to xyz... (chi2: %10.8f)" % chi2_xray_best_)
chi2_best_str = ("%10.8f" % chi2_xray_best_).zfill(12)
m.write_xyz(
    "tmp_/%s_%s.xyz" % (run_id, chi2_best_str),
    "run_id: %s" % run_id,
    atomlist,
    xyz_best_,
)
np.savetxt(
    "tmp_/%s_%s.dat" % (run_id, chi2_best_str),
    np.column_stack((qvector, predicted_best_)),
)

### Final save to files
# target function
np.savetxt(
    "tmp_/%s_target_function.dat" % run_id, np.column_stack((qvector, target_function))
)
# save raw target data:
np.savetxt("tmp_/%s_target_iam.dat" % run_id, np.column_stack((qvector, target_iam)))
# save starting IAM signal
np.savetxt(
    "tmp_/%s_starting_iam.dat" % run_id, np.column_stack((qvector, starting_iam))
)

print("Total time: %3.2f s" % float(default_timer() - start))
