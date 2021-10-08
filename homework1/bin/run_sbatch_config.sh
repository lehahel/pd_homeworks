#!/bin/bash
#
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --partition=RT
#SBATCH --job-name=example
#SBATCH --comment="Run mpi from config"
#SBATCH --output=out.txt
#SBATCH --error=error.txt
mpiexec ./MpiHomework 1000000
