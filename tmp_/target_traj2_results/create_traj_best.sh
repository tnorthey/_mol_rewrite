#!/bin/bash

rm traj_best.xyz

for i in {00..14}
do
    num=$(printf %02d $((10#$i)))
    xyz=$(ls -1 "$num"_*.xyz | head -n 1)
    echo $xyz
    cat $xyz >> traj_best.xyz
done
