#!/bin/bash
##SBATCH --output=logs/testing.log
##SBATCH --error=logs/testing.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
##SBATCH --ntasks=1
#SBATCH --partition=volta_devel
##SBATCH --partition=volta_compute
##SBATCH --qos=regular
##SBATCH --tasks-per-node=2
##SBATCH --cpus-per-task=1
##SBATCH --gpus-per-task=1
##SBATCH --gpus-per-node=2

$SIMULATEQCD_BUILD/scripts/runTests_3.bash