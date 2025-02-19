#!/bin/bash                                                                                                                         
#SBATCH -N 1
#SBATCH --ntasks-per-node 32
#SBATCH --output output.log
#SBATCH --error error.log
#SBATCH --job-name=regression-test

source <PATH TO>/trilinos-cuda-config.sh
#source <PATH TO>/vertex-env-cpu.sh # For GPU comment above line and use this line.

export OMP_PROC_BIND=true
export OMP_PLACES=threads

export IOSS_PROPERTIES="COMPOSE_RESULTS=on:MINIMIZE_OPEN_FILES=on:MAXIMUM_NAME_LENGTH=64:DUPLICATE_FIELD_NAME_BEHAVIOR=WARNING"
export EXODUS_NETCDF4=1

export MAP_STRING=slot:pe=${SLURM_CPUS_PER_TASK}

cd regression_test
pytest -k "test"
