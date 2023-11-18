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
simulated_annealing_bool = bool(int(sys.argv[4]))
gradient_descent_bool = bool(int(sys.argv[5]))
###################################

#############################
### arguments             ###
#############################
reference_xyz_file = "xyz/chd_reference.xyz"
nsteps = 20000
qmin = 1e-9
qmax = 8.0
qlen = 81
starting_temp = 0.2
step_size = 0.01
harmonic_factor = 0.01  # HO factor
n_trials = 20  # repeats n_trails times, only saves lowest f

electron_mode = False  # x-rays
inelastic = True
noise_bool = False
noise = 0
nmfile = "nm/chd_normalmodes.txt"
pcd_mode = True
xyz_save = False

# gradient descent parameters
nsteps_gd = 500
# step_size_gd = 0.0001  # works (?)
# step_size_gd = 0.000001  # 1e-6
step_size_gd = 1e-5

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

natoms = starting_xyz.shape[0]
displacements = sa.read_nm_displacements(nmfile, natoms)
nmodes = displacements.shape[0]

mode_indices = np.arange(0, nmodes)  # CHD, all modes
print("including modes:")
print(mode_indices)

step_size_array = step_size * np.ones(nmodes)

###############################################
### Initial condition generation parameters ###
###############################################
# alternative step-sizes for generating initial conditions
# > 10x step-sizes, hydrogen modes damped
hydrogen_modes = np.arange(28, nmodes)
h_mode_modification = np.ones(nmodes) 
for i in hydrogen_modes:
    h_mode_modification[i] *= 0.05
ic_step_size_array = 10 * step_size_array * h_mode_modification
ic_harmonic_factor = 0.1  # a stronger HO factor for IC generation
ninitials = 1000
ic_nsteps = 200
generate_initial_conditions = True
###############################################
###############################################

#################################
### End Initialise some stuff ###
#################################

if generate_initial_conditions:
    f_best_ = 1e9
    for j in range(ninitials):
        # Run simulated annealing
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
            target_function,
            qvector,
            ic_step_size_array,
            ho_indices,
            starting_temp,
            ic_nsteps,
            inelastic,
            ic_harmonic_factor,
            pcd_mode,
            electron_mode,
            False,
        )
        print("f_best (SA): %9.8f" % f_best)

        ### store results as xyz files ###
        print("writing to xyz... (f: %10.8f)" % f_best)
        f_best_str = ("%10.8f" % f_best).zfill(12)
        m.write_xyz(
            "tmp_/ic_%s_%s.xyz" % (run_id, f_best_str),
            "run_id: %s" % run_id,
            atomlist,
            xyz_best,
        )

        # store best value from the n_initials
        if f_best < f_best_:
            f_best_, f_xray_best_, predicted_best_, xyz_best_ = (
                f_best,
                f_xray_best,
                predicted_best,
                xyz_best,
            )
    # Finally,
    # this will be the starting point for the full run in the next step
    print('IC generation complete.')
    print('starting_xyz chosen: f_best = %9.8f f_xray_best = %9.8f' % (f_best_, f_xray_best_))
    starting_xyz = xyz_best_

f_best_ = 1e9
xyz_best = starting_xyz
for k in range(n_trials):
 
    if simulated_annealing_bool:
        starting_temp = 0
        # Run simulated annealing
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
            target_function,
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
        print("f_best (SA): %9.8f" % f_best)
        ### save xyz_array as an xyz trajectory
        if xyz_save:
            print("saving xyz array...")
            fname = "tmp_/save_array.xyz"
            m.write_xyz_traj(fname, atomlist, xyz_array)


    if gradient_descent_bool:
        # gradient descent...
        starting_xyz_gd = xyz_best
        # pcd_mode = True
        target_function = target_iam
        pcd_mode = False
        (f_best, predicted_best, xyz_best) = gd.gradient_descent_cartesian(
            target_function,
            atomic_numbers,
            starting_xyz_gd,
            qvector,
            nsteps_gd,
            step_size_gd,
            pcd_mode,
            reference_iam,
        )
        print("f_best (GD): %9.8f" % f_best)
        f_xray_best = f_best

    # store best values from the n_trials
    if f_best < f_best_:
        f_best_, f_xray_best_, predicted_best_, xyz_best_ = (
            f_best,
            f_xray_best,
            predicted_best,
            xyz_best,
        )

# calculate raw IAM data
iam_best = xyz2iam(xyz_best_, atomlist)

# save final IAM signal
np.savetxt("tmp_/%s_iam_best.dat" % run_id, np.column_stack((qvector, iam_best)))

print("writing to xyz... (f: %10.8f)" % f_xray_best_)
f_best_str = ("%10.8f" % f_xray_best_).zfill(12)
m.write_xyz(
    "tmp_/%s_%s.xyz" % (run_id, f_best_str),
    "run_id: %s" % run_id,
    atomlist,
    xyz_best_,
)
np.savetxt(
    "tmp_/%s_%s.dat" % (run_id, f_best_str),
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
