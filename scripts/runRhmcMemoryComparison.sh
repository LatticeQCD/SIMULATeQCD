#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=RhmcMemory_%j.err
#SBATCH --output=RhmcMemory_%j.out
#SBATCH --job-name=RhmcMemory
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=volta_devel
#SBATCH --mail-type=NONE

# Print SIMULATeQCD's live device allocations after one RHMC trajectory.
# This deliberately uses MemoryManagement::memorySummary(), not nvidia-smi.

set -euo pipefail
export LC_ALL=C

read -r -a benchmark_modules <<< "${RHMC_MODULES-compilers/gnu/12.2.1 cuda/12.8 mpi/openmpi/ompi-cuda-5.0.7}"
if (( ${#benchmark_modules[@]} )); then
    module load "${benchmark_modules[@]}"
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this memory report with sbatch." >&2
    exit 2
fi
if [[ ${SLURM_NTASKS:-1} != 1 ]]; then
    echo "The RHMC memory report requires one MPI rank." >&2
    exit 2
fi

run_directory=$(realpath -e "${3:-${SLURM_SUBMIT_DIR:-.}}")
project_directory=$(dirname "$run_directory")
legacy_binary=$(realpath -e "${1:-${RHMC_LEGACY_BINARY:-$project_directory/buildSIMULATeQCD_legacy_1732861/testing/rhmcBenchmark}}")
recursive_binary=$(realpath -e "${2:-${RHMC_RECURSIVE_BINARY:-$project_directory/buildSIMULATeQCD/testing/rhmcBenchmark}}")

if [[ ! -x $legacy_binary || ! -x $recursive_binary || $legacy_binary == "$recursive_binary" ]]; then
    echo "Supply distinct legacy and recursive rhmcBenchmark binaries." >&2
    exit 2
fi

results_directory="${RHMC_MEMORY_RESULTS_DIR:-$run_directory/rhmc_memory/$SLURM_JOB_ID}"
mkdir -p "$results_directory"
results_directory=$(realpath -e "$results_directory")

echo "Memory source: SIMULATeQCD MemoryManagement device container"
echo "Measurement point: live allocations immediately after one complete HMC.update()"
echo "Legacy binary: $legacy_binary"
echo "Recursive binary: $recursive_binary"
echo "Results directory: $results_directory"

report_total_bytes=0

run_report() {
    local implementation=$1
    local binary=$2
    local logfile="$results_directory/${implementation}.log"
    local reportfile="$results_directory/${implementation}_device_memory.txt"

    echo "Running $implementation RHMC trajectory" >&2
    if ! (cd "$run_directory" && "$binary" --memory-summary) > "$logfile" 2>&1; then
        cat "$logfile" >&2
        echo "$implementation RHMC trajectory failed." >&2
        exit 1
    fi

    awk '
        /RHMC MemoryManagement device report BEGIN/ {inside=1; next}
        /RHMC MemoryManagement device report END/   {inside=0}
        inside
    ' "$logfile" > "$reportfile"

    if [[ ! -s $reportfile ]]; then
        echo "$implementation did not emit an internal memory report." >&2
        echo "Rebuild both rhmcBenchmark binaries from the common updated driver." >&2
        exit 1
    fi

    report_total_bytes=$(sed -n 's/.*Total: \([0-9][0-9]*\) Bytes.*/\1/p' "$reportfile" | tail -n 1)
    if [[ -z $report_total_bytes ]]; then
        cat "$reportfile" >&2
        echo "Could not read the MemoryManagement device total for $implementation." >&2
        exit 1
    fi

    echo
    echo "===== $implementation: SIMULATeQCD-managed GPU memory ====="
    cat "$reportfile"
    awk -v bytes="$report_total_bytes" \
        'BEGIN {printf "Managed GPU total: %s bytes (%.3f GiB)\n", bytes, bytes/1073741824}'
}

run_report legacy "$legacy_binary"
legacy_bytes=$report_total_bytes

run_report recursive "$recursive_binary"
recursive_bytes=$report_total_bytes

legacy_gib=$(awk -v bytes="$legacy_bytes" 'BEGIN {printf "%.6f", bytes/1073741824}')
recursive_gib=$(awk -v bytes="$recursive_bytes" 'BEGIN {printf "%.6f", bytes/1073741824}')
difference_bytes=$((recursive_bytes - legacy_bytes))
difference_gib=$(awk -v bytes="$difference_bytes" 'BEGIN {printf "%+.6f", bytes/1073741824}')
difference_percent=$(awk -v old="$legacy_bytes" -v new="$recursive_bytes" \
    'BEGIN {printf "%+.2f", 100.0*(new-old)/old}')

summary_file="$results_directory/rhmc_managed_device_memory.csv"
printf '%s\n' 'implementation,managed_device_bytes,managed_device_gib' > "$summary_file"
printf 'legacy,%s,%s\n' "$legacy_bytes" "$legacy_gib" >> "$summary_file"
printf 'recursive,%s,%s\n' "$recursive_bytes" "$recursive_gib" >> "$summary_file"

echo
echo "===== INTERNAL GPU MEMORY SUMMARY ====="
printf 'Legacy managed GPU memory:    %s bytes (%s GiB)\n' "$legacy_bytes" "$legacy_gib"
printf 'Recursive managed GPU memory: %s bytes (%s GiB)\n' "$recursive_bytes" "$recursive_gib"
printf 'Recursive - legacy:           %s bytes (%s GiB, %s%%)\n' \
    "$difference_bytes" "$difference_gib" "$difference_percent"
echo "Allocation details: $results_directory/{legacy,recursive}_device_memory.txt"
echo "CSV summary: $summary_file"
