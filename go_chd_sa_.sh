#!/bin/bash

# "0th" step
if [ 1 -eq 1 ]
then
    run_id=0
    starting_xyz_file="xyz/start.xyz"
    #target_xyz_file="xyz/target_traj099/target_00.xyz"
    target_xyz_file="xyz/target_traj099/target_20.xyz"
    python3 run_chd_sa_.py $run_id $starting_xyz_file $target_xyz_file
    ls tmp_
fi

# steps 01 - 08
if [ 0 -eq 1 ]
then
    echo "loop..."
    # loop over target_00.xyz, target_01.xyz, ...
    for i in {01..75}
    do
        run_id=$((10#$i))  # specify it's base 10 otherwise it assumes octal and errors
        starting_xyz_file=$( ls -1 tmp_/"$(printf %02d $((run_id - 1)))"_*.xyz | head -n 1 )
        target_xyz_file="xyz/target_traj099/target_$i.xyz"
        echo "starting xyz: $starting_xyz_file"
        echo "target_xyz: $target_xyz_file"
        python3 run_chd_sa_.py $run_id $starting_xyz_file $target_xyz_file
    done
fi
