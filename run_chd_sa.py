import numpy as np
import sys
import scipy.io
from timeit import default_timer
from numpy.random import random_sample as random

# import chemcoord
# my module
import molecule

start = default_timer()

# initialise classes
m = molecule.Xyz()
x = molecule.Xray()
d = molecule.Descent()

# command line arguments
nsteps = int(sys.argv[1])
qmin = float(sys.argv[2])
qmax = float(sys.argv[3])
qlen = int(sys.argv[4])
starting_temp = float(sys.argv[5])
start_xyz_file = str(sys.argv[6])
reference_xyz_file = str(sys.argv[7])
target_xyz_file = str(sys.argv[8])
harmonic_factor = float(sys.argv[9])  # HO factor
n_restarts = int(sys.argv[10])  # n_restarts number of times it entirely restarts
n_trials = int(sys.argv[11])  # repeats n_trails times, only saves lowest chi2
print("harmonic factor: %f" % harmonic_factor)

# select q-range
# qmin_index, qmax_index = 0, len(qexp)
qvector = np.linspace(qmin, qmax, qlen, endpoint=True)
print("qmin, qmax = %9.8f, %9.8f" % (qvector[0], qvector[-1]))

inelastic = True
pcd_mode = True
q_mode = False
noise_bool = True
noise = 4

def xyz2iam(xyz, atomlist):
    """convert xyz file to IAM signal"""
    atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
    compton_array = x.compton_spline(atomic_numbers, qvector)
    iam = x.iam_calc_compton(atomic_numbers, xyz, qvector, inelastic, compton_array)
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
    mu = 0		# normal distribution with mean of mu
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

# definitions
non_h_indices = [0, 1, 2, 3, 4, 5]

nmfile = "nm/chd_normalmodes.txt"
natoms = starting_xyz.shape[0]
displacements = d.read_nm_displacements(nmfile, natoms)
nmodes = displacements.shape[0]

# mode_indices = np.arange(0, 28)  # CHD, this removes hydrogen modes
mode_indices = np.arange(0, 36)  # CHD, all modes
print("including modes:")
print(mode_indices)

step_size = 0.01 * np.ones(nmodes)

# CHD specific; notably not indices 0, 5 (the ring-opening)
ho_indices1 = [0, 1, 2, 3, 4]  # chd specific!
ho_indices2 = [1, 2, 3, 4, 5]  # chd specific!

# ho_indices1 = [0, 1, 2, 3, 4, 0]  # chd ring-closed specific! (Carbons)
# ho_indices2 = [1, 2, 3, 4, 5, 5]  # chd ring-closed specific!

# ho_indices1 = [0, 1, 2, 3, 4, 0, 6, 12, 5,  5,  0, 0, 1, 2, 3,  4 ]  # chd ring-closed specific!
# ho_indices2 = [1, 2, 3, 4, 5, 5, 7, 13, 12, 13, 6, 7, 8, 9, 10, 11]  # chd ring-closed specific!

# ho_indices1 = [0, 1, 2, 3, 4, 0, 6, 12, 5,  5,  0, 0, 1, 2, 3,  4,  0, 1, 2,  3,  4,  2, 3, 4,  5,  0,  0,  4,  4, 5, 5, 1, 1 ]  # chd ring-closed specific!
# ho_indices2 = [1, 2, 3, 4, 5, 5, 7, 13, 12, 13, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, 8, 9, 10, 11, 12, 13, 12, 13, 6, 7, 6, 7]  # chd ring-closed specific!

run_id_ = 0  # define a number to label the start of the output filenames
run_id = str(run_id_).zfill(2)  # pad with zeros

def chi2(predicted_function, target_function):
    qlen = len(predicted_function)
    xray_contrib = (
        #np.sum((predicted_function - target_function) ** 2 / target_function) / qlen
        np.sum((predicted_function - target_function) ** 2) / qlen
    )
    return xray_contrib


for k1 in range(n_restarts):
    chi2_best_ = 1e9
    for k2 in range(n_trials):
        # Run simulated annealing
        (
            chi2_best,
            predicted_best,
            xyz_best,
            chi2_array,
            chi2_best_xray,
        ) = d.simulated_annealing_modes_ho(
            atomlist,
            starting_xyz,
            reference_xyz,
            displacements,
            mode_indices,
            target_function,
            qvector,
            step_size,
            ho_indices1,
            ho_indices2,
            starting_temp,
            nsteps,
            inelastic,
            harmonic_factor,
            pcd_mode,
            q_mode,
        )

        print("%10.8f" % chi2_best)

        # store best values from the n_trials
        if chi2_best < chi2_best_:
            chi2_best_, chi2_best_xray_, predicted_best_, xyz_best_ = (
                chi2_best,
                chi2_best_xray,
                predicted_best,
                xyz_best,
            )

    # calculate raw IAM data
    iam_best = xyz2iam(xyz_best_, atomlist)

    # chi2 between raw data
    chi2_ = chi2(target_iam, iam_best)
    chi2_str = ("%10.8f" % chi2_).zfill(12)

    # save final IAM signal
    np.savetxt(
        "%s_iam_best_%s.dat" % (run_id, chi2_str), np.column_stack((qvector, iam_best))
    )

    # Kabsch rotation to target
    # rmsd, r = sa.rmsd_kabsch(xyz_best, starting_xyz, non_h_indices)
    # xyz_best = np.dot(xyz_best, r.as_matrix())

    print("writing to xyz... (chi2: %10.8f)" % chi2_best_xray_)
    chi2_best_str = ("%10.8f" % chi2_best_xray_).zfill(12)
    m.write_xyz(
        "%s_%s.xyz" % (run_id, chi2_best_str),
        "run_id: %s" % run_id,
        atomlist,
        xyz_best_,
    )
    np.savetxt(
        "%s_%s.dat" % (run_id, chi2_best_str),
        np.column_stack((qvector, predicted_best_)),
    )


### Final save to files
# target function
np.savetxt(
    "%s_target_function.dat" % run_id, np.column_stack((qvector, target_function))
)
# save raw target data:
np.savetxt("%s_target_iam.dat" % run_id, np.column_stack((qvector, target_iam)))
# save starting IAM signal
np.savetxt("%s_starting_iam.dat" % run_id, np.column_stack((qvector, starting_iam)))

print("Total time: %3.2f s" % float(default_timer() - start))
