#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=hisqForceTiming_%j.err
#SBATCH --output=hisqForceTiming_%j.out
#SBATCH --job-name=HisqForceTiming
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=volta_devel
#SBATCH --mail-type=NONE

set -euo pipefail
export LC_ALL=C

module load compilers/gnu/12.2.1
module load cuda/12.8
module load mpi/openmpi/ompi-cuda-5.0.7

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: sbatch $0 LEGACY_BINARY RECURSIVE_BINARY [RUN_DIRECTORY]" >&2
    exit 2
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this timing comparison with sbatch; do not run it on a login node." >&2
    exit 2
fi

legacy_binary=$(realpath -e "$1")
recursive_binary=$(realpath -e "$2")
run_directory=$(realpath -e "${3:-.}")
project_directory=$(dirname "$run_directory")
plot_script=$(realpath -e "${HISQ_PLOT_SCRIPT:-$project_directory/SIMULATeQCD/scripts/plotHisqPerformance.py}")
results_directory="${HISQ_FORCE_RESULTS_DIR:-$run_directory/hisq_force_results/$SLURM_JOB_ID}"
mkdir -p "$results_directory"
results_directory=$(realpath -e "$results_directory")

if [[ $(basename "$legacy_binary") != "hisqForceBenchmark" ||
      $(basename "$recursive_binary") != "hisqForceBenchmark" ]]; then
    echo "Both inputs must be separately built hisqForceBenchmark binaries." >&2
    exit 2
fi

parameter_file="$run_directory/../parameter/tests/hisqForce_bench.param"
gauge_file="$run_directory/../test_conf/l528f21b6315m00282m0759_001.1610"

if [[ ! -r "$parameter_file" || ! -r "$gauge_file" ]]; then
    echo "The common run directory does not resolve the required benchmark inputs:" >&2
    echo "  $parameter_file" >&2
    echo "  $gauge_file" >&2
    exit 2
fi

if ! grep -Eq '^[[:space:]]*Lattice[[:space:]]*=[[:space:]]*52[[:space:]]+52[[:space:]]+52[[:space:]]+8([[:space:]]|$)' "$parameter_file"; then
    echo "Benchmark parameter file is not configured for 52^3 x 8." >&2
    exit 2
fi

if ! grep -Eq '^[[:space:]]*Nodes[[:space:]]*=[[:space:]]*1[[:space:]]+1[[:space:]]+1[[:space:]]+1([[:space:]]|$)' "$parameter_file"; then
    echo "Benchmark parameter file is not configured for one MPI rank." >&2
    exit 2
fi

if ! grep -Eq '^[[:space:]]*Gaugefile[[:space:]]*=[[:space:]]*\.\./test_conf/l528f21b6315m00282m0759_001\.1610([[:space:]]|$)' "$parameter_file"; then
    echo "Benchmark parameter file does not select the required gauge configuration." >&2
    exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1 &&
   ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'V100'; then
    echo "No V100 GPU was detected. Run this script inside a one-V100 allocation." >&2
    exit 2
fi

echo "HISQ force benchmark lattice: 52^3 x 8"
echo "PROVENANCE: 52^3 x 8 is the ONLY REAL THERMALIZED lattice in the force benchmarks"
echo "HISQ force input: REAL THERMALIZED gauge configuration"
echo "Gauge configuration: $gauge_file"

run_once() {
    local label=$1
    local binary=$2
    local repetition=$3
    local output
    local time_record
    local logfile="$results_directory/${label}_${repetition}.log"

    echo "Running $label repetition $repetition/3" >&2

    # jobscript1 launches the single-rank SIMULATeQCD executable directly.
    if ! output=$(cd "$run_directory" && "$binary" 2>&1); then
        printf '%s\n' "$output" >&2
        echo "$label repetition $repetition failed." >&2
        exit 1
    fi

    printf '%s\n' "$output" > "$logfile"

    time_record=$(printf '%s\n' "$output" | grep -Eo 'HISQ force time: [0-9]+([.][0-9]+)?s' | tail -n 1 || true)

    if [[ -z "$time_record" ]]; then
        printf '%s\n' "$output" >&2
        echo "$label repetition $repetition did not emit a canonical HISQ force time; rebuild both binaries." >&2
        exit 1
    fi

    printf '%s\n' "${time_record#HISQ force time: }" | sed 's/s$//'
}

median_of_three() {
    printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n '2p'
}

declare -a legacy_times
declare -a recursive_times

# Alternate order to reduce systematic first/second-run bias.
legacy_times[0]=$(run_once legacy "$legacy_binary" 1)
recursive_times[0]=$(run_once recursive "$recursive_binary" 1)

recursive_times[1]=$(run_once recursive "$recursive_binary" 2)
legacy_times[1]=$(run_once legacy "$legacy_binary" 2)

legacy_times[2]=$(run_once legacy "$legacy_binary" 3)
recursive_times[2]=$(run_once recursive "$recursive_binary" 3)

legacy_median=$(median_of_three "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}")
recursive_median=$(median_of_three "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}")

speedup=$(awk -v legacy="$legacy_median" -v recursive="$recursive_median" 'BEGIN { printf "%.6f", legacy / recursive }')
reduction=$(awk -v legacy="$legacy_median" -v recursive="$recursive_median" 'BEGIN { printf "%.2f", 100.0 * (legacy - recursive) / legacy }')

printf '\nLegacy times:    %ss %ss %ss\n' "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}"
printf 'Recursive times: %ss %ss %ss\n' "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}"
printf 'Median legacy time:    %ss\n' "$legacy_median"
printf 'Median recursive time: %ss\n' "$recursive_median"
printf 'Speedup: %sx\n' "$speedup"
printf 'Percent time reduction: %s%%\n' "$reduction"

summary_file="$results_directory/hisq_force_timing.csv"
printf '%s\n' 'spatial_l,nt,sites,gauge_kind,gauge_file,legacy_1_s,legacy_2_s,legacy_3_s,recursive_1_s,recursive_2_s,recursive_3_s,legacy_median_s,recursive_median_s,speedup,reduction_percent' > "$summary_file"
printf '52,8,1124864,thermalized,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$gauge_file" \
    "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}" \
    "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}" \
    "$legacy_median" "$recursive_median" "$speedup" "$reduction" >> "$summary_file"

python3 "$plot_script" --hisq-force "$summary_file" --output-dir "$results_directory"
printf 'HISQ-force timing data: %s\n' "$summary_file"
printf 'HISQ-force timing plot: %s\n' "$results_directory/hisq_force_timing_gain.pdf"
