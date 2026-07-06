#!/bin/bash
#Metadata and output files
#SBATCH -J dem_test
#SBATCH -D ./ #set the working directory (optional), here the submitting directory
#SBATCH -o ./%x.%j.%N.out #write standard output and error to the specified file (parent directory must exist), directory here is relative to -D directory, for options see the slurm documentation https://slurm.schedmd.com/archive/slurm-20.11.9/sbatch.html

#The next 2 options ensure a consistent job environment -> IMPORTANT
#SBATCH --get-user-env #get the user environment
#SBATCH --export=NONE #Do not export the submitting environment

#Where the job should run
#SBATCH --clusters=serial #options are htnm, cm4, serial
#SBATCH --partition=serial_std #options depend on the cluster for htnm: htnm_sam, for cm4: cm4_tiny, cm4_std and for serial: serial_std, serial_long
##SBATCH --qos=cm4_tiny #required only for cm4, select cm4_tiny for the cm4_tiny partition and cm4_std for the cm4_std partition

#Required resources and time
#SBATCH --ntasks=1 # htnm has 192 cores per node, alternatively set --nodes=x --ntasks-per-node=y and check the cheat sheet!
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00

#Setup the runtime environment (probably only mpi), which is not equal to the build environment
module load slurm_setup #Required!
module load deal.II-run-time-dependencies/

#instead of srun you can also use: mpiexec -n $SLURM_NTASKS or mpirun -n $SLURM_NTASKS or mpiexec or mpirun
mpirun /dss/dsshome1/0B/ge37fof2/development/meltpooldg/build_release/applications/mp-dem/mp-dem ./particle_drop.json