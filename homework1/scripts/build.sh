#!/bin/bash

# -- DO NOT EXECUTE MANUALLY -- #

rm -rf build

mkdir build
cd build
module add mpi/openmpi4-x86_64
cmake ..
make

cd ../bin
# ./start_sbatch.sh

rm -rf slurm-*