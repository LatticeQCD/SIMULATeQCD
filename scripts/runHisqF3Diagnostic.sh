#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=HisqF3Diagnostic_%j.err
#SBATCH --output=HisqF3Diagnostic_%j.out
#SBATCH --job-name=HisqF3Diagnostic
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=volta_devel
#SBATCH --mail-type=NONE

set -euo pipefail

module load compilers/gnu/12.2.1
module load cuda/12.8
module load mpi/openmpi/ompi-cuda-5.0.7

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this test with sbatch; do not run it on a login node." >&2
    exit 2
fi

submit_directory=${SLURM_SUBMIT_DIR:?Submit this script from the testrun directory}
project_directory=${HISQ_PROJECT_DIR:-$(dirname "$submit_directory")}
run_directory=${HISQ_RUN_DIR:-$project_directory/testrun}
binary=${HISQ_F3_BINARY:-$project_directory/buildSIMULATeQCD_hisq_f3/testing/hisqForce}

if [[ ! -d "$run_directory" ]]; then
    echo "Run directory is not available: $run_directory" >&2
    exit 2
fi

if [[ ! -x "$binary" ]]; then
    echo "F3 diagnostic binary is not executable: $binary" >&2
    echo "Build target hisqForce in buildSIMULATeQCD_hisq_f3 first." >&2
    exit 2
fi

cd "$run_directory"
"$binary"
