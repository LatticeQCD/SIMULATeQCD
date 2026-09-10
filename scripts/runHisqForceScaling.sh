#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=hisqForceScaling_%j.err
#SBATCH --output=hisqForceScaling_%j.out
#SBATCH --job-name=HisqForceScaling
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=volta_devel
#SBATCH --mail-type=NONE

set -euo pipefail
export LC_ALL=C

read -r -a benchmark_modules <<< "${HISQ_MODULES-compilers/gnu/12.2.1 cuda/12.8 mpi/openmpi/ompi-cuda-5.0.7}"
if (( ${#benchmark_modules[@]} )); then
    module load "${benchmark_modules[@]}"
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this benchmark with sbatch; do not run it on a login node." >&2
    exit 2
fi

if [[ $# -gt 3 ]]; then
    echo "Usage: sbatch $0 [LEGACY_BINARY] [RECURSIVE_BINARY] [RUN_DIRECTORY]" >&2
    exit 2
fi

run_directory=$(realpath -e "${3:-${SLURM_SUBMIT_DIR:-.}}")
project_directory=$(dirname "$run_directory")
legacy_binary=$(realpath -e "${1:-${HISQ_LEGACY_BINARY:-$project_directory/buildSIMULATeQCD_legacy_1732861/testing/hisqForceBenchmark}}")
recursive_binary=$(realpath -e "${2:-${HISQ_RECURSIVE_BINARY:-$project_directory/buildSIMULATeQCD/testing/hisqForceBenchmark}}")
plot_script=$(realpath -e "${HISQ_PLOT_SCRIPT:-$project_directory/SIMULATeQCD/scripts/plotHisqPerformance.py}")

if [[ ! -x $legacy_binary || ! -x $recursive_binary || $legacy_binary == "$recursive_binary" ]]; then
    echo "Supply two distinct executable hisqForceBenchmark binaries." >&2
    exit 2
fi

parameter_file="$run_directory/../parameter/tests/hisqForce_bench.param"
gauge_file="$run_directory/../test_conf/l528f21b6315m00282m0759_001.1610"
if [[ ! -r $parameter_file || ! -r $gauge_file ]]; then
    echo "Required benchmark input is missing:" >&2
    echo "  $parameter_file" >&2
    echo "  $gauge_file" >&2
    exit 2
fi

if ! grep -Eq '^[[:space:]]*Gaugefile[[:space:]]*=[[:space:]]*\.\./test_conf/l528f21b6315m00282m0759_001\.1610([[:space:]]|$)' "$parameter_file"; then
    echo "Benchmark parameter file does not select the required 52^3 x 8 thermalized gauge configuration." >&2
    exit 2
fi

if [[ ${SLURM_NTASKS:-1} != 1 ]]; then
    echo "This benchmark requires one MPI rank." >&2
    exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1 &&
   ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'V100'; then
    echo "No V100 GPU was detected." >&2
    exit 2
fi

read -r -a spatial_sizes <<< "${HISQ_SCALING_SIZES:-16 24 32 40 48 52}"
for spatial_l in "${spatial_sizes[@]}"; do
    if [[ ! $spatial_l =~ ^[0-9]+$ ]] || (( spatial_l < 2 || spatial_l % 2 != 0 )); then
        echo "HISQ_SCALING_SIZES must contain positive even integers." >&2
        exit 2
    fi
done

results_directory="${HISQ_FORCE_SCALING_RESULTS_DIR:-$run_directory/hisq_force_scaling/$SLURM_JOB_ID}"
mkdir -p "$results_directory"
results_directory=$(realpath -e "$results_directory")
summary_file="$results_directory/hisq_force_scaling.csv"

echo "HISQ force volume-scaling provenance:"
echo "  52^3 x 8 is the ONLY REAL THERMALIZED lattice: $gauge_file"
echo "  Every other lattice size: SEEDED RANDOM gauge configuration (synthetic scaling only)"
echo "Legacy binary: $legacy_binary"
echo "Recursive binary: $recursive_binary"
echo "Results directory: $results_directory"

printf '%s\n' 'spatial_l,nt,sites,gauge_kind,gauge_file,legacy_1_s,legacy_2_s,legacy_3_s,recursive_1_s,recursive_2_s,recursive_3_s,legacy_median_s,recursive_median_s,speedup,reduction_percent' > "$summary_file"

run_once() {
    local label=$1
    local binary=$2
    local spatial_l=$3
    local repetition=$4
    local gauge_kind=$5
    local output
    local time_record
    local logfile="$results_directory/L${spatial_l}_${gauge_kind}_${label}_${repetition}.log"
    local -a gauge_option=()

    if [[ $gauge_kind == random ]]; then
        gauge_option=(--random-gauge)
    fi

    echo "Running L=${spatial_l}, input=${gauge_kind}, $label repetition $repetition/3" >&2
    if ! output=$(cd "$run_directory" && "$binary" "${gauge_option[@]}" \
        "$parameter_file" "Lattice=$spatial_l $spatial_l $spatial_l 8" "Nodes=1 1 1 1" 2>&1); then
        printf '%s\n' "$output" >&2
        echo "L=${spatial_l} $label repetition $repetition failed." >&2
        exit 1
    fi
    printf '%s\n' "$output" > "$logfile"

    time_record=$(printf '%s\n' "$output" | grep -Eo 'HISQ force time: [0-9]+([.][0-9]+)?s' | tail -n 1 || true)
    if [[ -z $time_record ]]; then
        printf '%s\n' "$output" >&2
        echo "Missing HISQ force time; rebuild both binaries with the common benchmark source." >&2
        exit 1
    fi
    printf '%s\n' "${time_record#HISQ force time: }" | sed 's/s$//'
}

median_of_three() {
    printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n '2p'
}

for spatial_l in "${spatial_sizes[@]}"; do
    gauge_kind=random
    gauge_record=random_seeded
    if (( spatial_l == 52 )); then
        gauge_kind=thermalized
        gauge_record=$gauge_file
    fi

    echo
    if [[ $gauge_kind == thermalized ]]; then
        echo "Lattice ${spatial_l}^3 x 8 uses the REAL THERMALIZED gauge configuration: $gauge_file"
    else
        echo "Lattice ${spatial_l}^3 x 8 uses a SEEDED RANDOM gauge configuration (synthetic scaling point)"
    fi

    declare -a legacy_times recursive_times
    legacy_times[0]=$(run_once legacy "$legacy_binary" "$spatial_l" 1 "$gauge_kind")
    recursive_times[0]=$(run_once recursive "$recursive_binary" "$spatial_l" 1 "$gauge_kind")
    recursive_times[1]=$(run_once recursive "$recursive_binary" "$spatial_l" 2 "$gauge_kind")
    legacy_times[1]=$(run_once legacy "$legacy_binary" "$spatial_l" 2 "$gauge_kind")
    legacy_times[2]=$(run_once legacy "$legacy_binary" "$spatial_l" 3 "$gauge_kind")
    recursive_times[2]=$(run_once recursive "$recursive_binary" "$spatial_l" 3 "$gauge_kind")

    legacy_median=$(median_of_three "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}")
    recursive_median=$(median_of_three "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}")
    speedup=$(awk -v old="$legacy_median" -v new="$recursive_median" 'BEGIN { printf "%.6f", old / new }')
    reduction=$(awk -v old="$legacy_median" -v new="$recursive_median" 'BEGIN { printf "%.2f", 100.0 * (old - new) / old }')
    sites=$((spatial_l * spatial_l * spatial_l * 8))

    printf '  Legacy median: %ss; recursive median: %ss; gain: %s%%\n' \
        "$legacy_median" "$recursive_median" "$reduction"
    printf '%s,8,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$spatial_l" "$sites" "$gauge_kind" "$gauge_record" \
        "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}" \
        "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}" \
        "$legacy_median" "$recursive_median" "$speedup" "$reduction" >> "$summary_file"
    unset legacy_times recursive_times
done

python3 "$plot_script" --hisq-force "$summary_file" --output-dir "$results_directory"
echo "HISQ-force scaling data: $summary_file"
echo "HISQ-force scaling plot: $results_directory/hisq_force_scaling.pdf"
echo "HISQ-force gain plot: $results_directory/hisq_force_gain.pdf"
