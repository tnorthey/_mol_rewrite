"""
Run the simulated annealing function for CHD
Strategy 1: Generate a lot of initial conditions via short "hot" SA runs,
    start from the best structure from that -> N restarts of longer "cooler" SA runs
    - This should find a reasonably close starting point from the ICs,
    then optimise it further with the subsequent longer runs
"""
import numpy as np
import sys
from timeit import default_timer

# my modules
import modules.mol as mol
import modules.wrap as wrap

start = default_timer()

# create class objects
m = mol.Xyz()
w = wrap.Wrapper()

###################################
# command line arguments
run_id = int(sys.argv[1])  # define a number to label the start of the output filenames
start_xyz_file = str(sys.argv[2])
target_xyz_file = str(sys.argv[3])
reference_xyz_file = "xyz/chd_reference.xyz"
###################################

w.chd_strategy1(
    run_id,
    start_xyz_file,
    reference_xyz_file,
    target_xyz_file,
    qvector=np.linspace(1e-9, 8.0, 81, endpoint=True),
    ic_ninitials=200,
    ic_nsteps=200,
    ic_step_size = 0.1,
    ic_starting_temp = 0.2,
    ic_harmonic_factor = 0.1,  # a stronger HO factor for IC generation
    sa_nsteps = 8000,
    sa_step_size = 0.02,
    sa_starting_temp = 0.5,
    sa_harmonic_factor = 0.00,
    sa_n_trials = 1,  # repeats n_trails times, only saves lowest f
    sa_n_restarts = 20,  # entire thing repeats n_restarts times
    save_ic_xyzs = False, # save all ic_.xyz files in tmp_/
)

print("Total time: %3.2f s" % float(default_timer() - start))
