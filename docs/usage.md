---
parent: VERTEX-CFD v1.0 User Guide
title: Usage
nav_order: 3
usemathjax: true
---

# Usage

Once installed, VERTEX-CFD relies on two files. The first one is the 'vertexcfd' executable and the second one is the input file for the simulation. After the installation, the executable is located in `<PATH TO INSTALLATION LOCATION>/bin/vertexcfd`. The input file is case spesific and there are example case files in `vertex-cfd/examples/inputs`. In this document, the input file located in `vertex-cfd/examples/inputs/incompressible/incompressible_2d_channel.xml` will be used as an example case. 

## Running a simulation
In order to run a simulation in serial, vertexcfd can be called directly as:

```
<PATH TO INSTALLATION LOCATION>/bin/vertexcfd --i=PATH_TO_INPUT_FILE/incompressible_2d_channel.xml
```
To run in parallel, `mpirun` is required. An example script for the SLURM scheduler is below:
```
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
```
Once the simulation starts, the example output should look like:
```
============================================================================
Time Integration Begin
Thu Mar 31 21:46:09 2022

  Stepper = Backward Euler
  Simulation Time Range  [0, 0.2]
----------------------------------------------------------------------------

Time Step = 1; Order = 1
CFL = 1.000e+00; dt = 6.186e-03; Time = 0.00000e+00
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   6.65e-02
           1   2.03e-02          1   6.45e-16
           2   4.66e-04          1   4.25e-16
           3   5.40e-07          1   5.36e-16
           4   3.19e-13          1   6.58e-16
Time step time to completion (s): 5.97e+00

Time Step = 2; Order = 1
CFL = 1.000e+00; dt = 5.644e-03; Time = 6.18583e-03
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   5.73e-02
           1   5.72e-03          1   4.68e-16
           2   5.40e-05          1   4.01e-16
           3   3.47e-09          1   7.18e-16
Time step time to completion (s): 8.96e+00

Time Step = 3; Order = 1
CFL = 1.000e+00; dt = 5.385e-03; Time = 1.18299e-02
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   5.37e-02
           1   4.14e-03          1   4.17e-16
           2   3.35e-05          1   3.47e-16
           3   2.01e-09          1   5.54e-16
Time step time to completion (s): 7.03e-01
... ...
... ...
... ...
... ...
Time Step = 40; Order = 1
CFL = 1.000e+00; dt = 4.853e-03; Time = 1.93731e-01
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   3.57e-02
           1   1.15e-03          1   3.89e-16
           2   1.68e-06          1   3.90e-16
           3   1.66e-12          1   3.20e-16
Time step time to completion (s): 1.73e+00
    41 *  (dt = 4.852e-03, new = 1.417e-03)  Adjusting dt to hit final time.

Time Step = 41; Order = 1
CFL = 1.000e+00; dt = 1.417e-03; Time = 1.98583e-01
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   3.56e-02
           1   3.11e-04          1   2.95e-16
           2   3.67e-08          1   2.73e-16
           3   4.40e-16          1   2.86e-16
Time step time to completion (s): 6.91e+00

----------------------------------------------------------------------------
Total runtime = 1.45e+02 sec
              = 2.42e+00 min
              = 0.04 hr
Thu Mar 31 21:51:51 2022
Time integration complete.
============================================================================
```
Once the simulation is completed. The results should be ready for visualization. For the visualization, we suggest ParaView. However, any visualization software that supports Exodus format should work. The example solution file screenshot visualized in Paraview is shown below:

![paraview-ss](https://github.com/user-attachments/assets/75483e4a-cfec-4a51-891f-3b7c72262ba7)

