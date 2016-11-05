#!/bin/bash
sudo apt-get install -y python-numpy python-mpi4py openmpi-bin openmpi-doc libopenmpi-dev &&
git clone https://github.com/MartyZhang/Parallel_computing_labs.git &&
cd Parallel_computing_labs/Lab_2

NUM_PROC=$(grep -c ^processor /proc/cpuinfo)
ts=$(date +%s%N) ; mpirun -np $NUM_PROC python ~/Parallel_computing_labs/Lab_2/grid_512_512.py 1; tt=$((($(date +%s%N) - $ts)/1000000)) ; echo "Time taken: $tt"
ts=$(date +%s%N) ; mpirun -np $NUM_PROC python ~/Parallel_computing_labs/Lab_2/grid_512_512.py 1; tt=$((($(date +%s%N) - $ts)/1000000)) ; echo "Time taken: $tt"
ts=$(date +%s%N) ; mpirun -np $NUM_PROC python ~/Parallel_computing_labs/Lab_2/grid_512_512.py 1; tt=$((($(date +%s%N) - $ts)/1000000)) ; echo "Time taken: $tt"