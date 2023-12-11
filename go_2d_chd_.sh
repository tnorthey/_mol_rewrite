#!/bin/bash

# "0th" step
if [ 1 -eq 1 ]
then
    run_id="20_2d"
    starting_xyz_file="xyz/start.xyz"  # ring-open
    target_xyz_file="xyz/target_traj099/target_20.xyz"  # ring-open
    python3 run_2D_chd_.py $run_id $starting_xyz_file $target_xyz_file
fi
