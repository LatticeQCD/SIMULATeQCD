#!/bin/bash
#SBATCH --output=./testing.log
#SBATCH --error=./testing.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --partition=volta_devel
##SBATCH --partition=volta_compute
##SBATCH --qos=regular
##SBATCH --tasks-per-node=2
##SBATCH --cpus-per-task=1
##SBATCH --gpus-per-task=1
##SBATCH --gpus-per-node=2

## --ntasks 2 --gres=gpu:2 
srun $SIMULATEQCD_BUILD/applications/fourier_emt $SIMULATEQCD_BUILD/parameter/fourier_emt.param