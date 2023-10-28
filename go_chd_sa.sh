#!/bin/bash

### nsteps = int(sys.argv[1])
### qmin = float(sys.argv[2])
### qmax = float(sys.argv[3])
### qlen = int(sys.argv[4])
### starting_temp = float(sys.argv[5])
### start_xyz_file = str(sys.argv[6])
### reference_xyz_file = str(sys.argv[7])
### target_xyz_file = str(sys.argv[8])
### harmonic_factor = float(sys.argv[9])  # HO factor
### n_restarts = int(sys.argv[10])  # n_restarts number of times it entirely restarts
### n_trials = int(sys.argv[11])  # repeats n_trails times, only saves lowest chi2

nsteps=8000
qmin=0.1
qmax=4
qlen=41
starting_temp=0.2
starting_xyz_file="xyz/start.xyz"
reference_xyz_file="xyz/chd_reference.xyz"
#target_xyz_file="xyz/R_traj001_frame8.xyz"
target_xyz_file="xyz/target.xyz"
harmonic_factor=1
n_restarts=20  # outer loop
n_trials=20    # inner loop, it saves the best of each trial

python3 run_chd_sa.py $nsteps $qmin $qmax $qlen $starting_temp $starting_xyz_file $reference_xyz_file $target_xyz_file $harmonic_factor $n_restarts $n_trials
