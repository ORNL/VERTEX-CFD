#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --time=1:00:00
#SBATCH -o output.log
#SBATCH -e error.log

source PATH_TO_ENVIRONMENT_SCRIPT
                                                                                                         
export OMP_PROC_BIND=true
export OMP_PLACES=threads

mpirun <PATH TO INSTALLATION LOCATION>/bin/vertexcfd --i=<PATH TO INPUT FILE LOCATION>/incompressible_2d_channel.xml
