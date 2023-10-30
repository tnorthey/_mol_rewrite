#!/bin/bash

# "0th" step
run_id=0
starting_xyz_file="xyz/start.xyz"
target_xyz_file="xyz/target_00.xyz"
python3 run_chd_sa_.py $run_id $starting_xyz_file $target_xyz_file
ls tmp_

if [ 1 -eq 1 ]
then
    echo "loop..."
    # loop over target_00.xyz, target_01.xyz, ...
    for i in {01..02}
    do
        run_id=$i
        starting_xyz_file=$( ls -1 tmp_/"$(printf %02d $((run_id - 1)))"_*.xyz | head -n 1 )
        target_xyz_file="xyz/target_$i.xyz"
        echo "starting xyz: $starting_xyz_file"
        echo "target_xyz: $target_xyz_file"
        python3 run_chd_sa_.py $run_id $starting_xyz_file $target_xyz_file
    done
fi
