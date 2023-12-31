#!/bin/bash

# "0th" step
if [ 1 -eq 1 ]
then
    run_id="35"
    #starting_xyz_file="xyz/start.xyz"
    #starting_xyz_file="tmp_/target_20_best_00_000.00304575.xyz"
    starting_xyz_file="xyz/target_traj099/target_30.xyz"  # ring-open
    #target_xyz_file="xyz/target_traj099/target_00.xyz"
    #target_xyz_file="xyz/target_traj099/target_20.xyz"  # ring-open
    target_xyz_file="xyz/target_traj099/target_35.xyz"  # ring-open
    #target_xyz_file="xyz/target_traj099/target_25.xyz"  # ring-open
    python3 run_strategy1_chd_.py $run_id $starting_xyz_file $target_xyz_file
    info="1000_400"
    cd tmp_
    gnuplot plot.gp
    cp plot.png "$info"_plot.png
    cp "$run_id"_ic_r05_array.dat "$info"_ic_r05_array.dat
    cp "$run_id"_ic_dihedral_array.dat "$info"_ic_dihedral_array.dat
    cd -
fi

# steps 01 - 08
if [ 1 -eq 0 ]
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
        python3 strategy1_chd_.py $run_id $starting_xyz_file $target_xyz_file $simulated_annealing $gradient_descent
    done
fi


# Repeat steps with only gradient descent ...
if [ 0 -eq 1 ]
then
    simulated_annealing=0
    gradient_descent=1
    echo "loop..."
    # loop over target_00.xyz, target_01.xyz, ...
    for i in {0..0}
    do
        run_id=$(printf %02d $i)
        starting_xyz_file=$( ls -1 tmp_/"$run_id"_*.xyz | head -n 1 )
        target_xyz_file="xyz/target_traj099/target_"$run_id".xyz"
        echo "starting xyz: $starting_xyz_file"
        echo "target_xyz: $target_xyz_file"
        python3 strategy1_chd_.py $run_id $starting_xyz_file $target_xyz_file $simulated_annealing $gradient_descent
    done
fi
