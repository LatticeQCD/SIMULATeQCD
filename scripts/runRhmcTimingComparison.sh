#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=RhmcBenchmark_%j.err
#SBATCH --output=RhmcBenchmark_%j.out
#SBATCH --job-name=RhmcBenchmark
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=volta_devel
#SBATCH --mail-type=NONE

set -euo pipefail
export LC_ALL=C

# Override RHMC_MODULES for another site; use an empty value for a preloaded environment.
read -r -a rhmc_modules <<< "${RHMC_MODULES-compilers/gnu/12.2.1 cuda/12.8 mpi/openmpi/ompi-cuda-5.0.7}"
if (( ${#rhmc_modules[@]} )); then
    module load "${rhmc_modules[@]}"
fi

if [[ ${SLURM_JOB_ID:-} == "" ]]; then
    echo "Submit this benchmark with sbatch; do not run it on a login node." >&2
    exit 2
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 {collaborator|validate|short|reverse|scaling|timing [REPETITION]}" >&2
    exit 2
fi

mode=$1
shift
if [[ $mode == collaborator ]]; then
    trap 'status=$?; if (( status != 0 )); then echo "Collaborator check: FAIL (see logs)" >&2; fi' EXIT
fi

case $mode in
    collaborator|validate|short|reverse|scaling)
        repetition=
        ;;
    timing)
        if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
            repetition=$SLURM_ARRAY_TASK_ID
        else
            repetition=${1:-}
            if [[ $# -gt 0 ]]; then shift; fi
        fi
        if [[ $repetition != 1 && $repetition != 2 && $repetition != 3 ]]; then
            echo "Use --array=1-3%1 with timing, or specify timing 1, 2, or 3." >&2
            exit 2
        fi
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        exit 2
        ;;
esac

if [[ $# -gt 3 ]]; then
    echo "Too many arguments." >&2
    exit 2
fi

# Resolve from the submission/run directory, never Slurm's spool copy of this script.
run_directory=$(realpath -e "${3:-${SLURM_SUBMIT_DIR:-.}}")
project_directory=$(dirname "$run_directory")
legacy_binary=$(realpath -e "${1:-${RHMC_LEGACY_BINARY:-$project_directory/buildSIMULATeQCD_legacy_1732861/testing/rhmcBenchmark}}")
recursive_binary=$(realpath -e "${2:-${RHMC_RECURSIVE_BINARY:-$project_directory/buildSIMULATeQCD/testing/rhmcBenchmark}}")
if [[ ! -x $legacy_binary || ! -x $recursive_binary || $legacy_binary == "$recursive_binary" ]]; then
    echo "Supply two distinct executable legacy and recursive benchmarks." >&2
    exit 2
fi
if [[ ${SLURM_NTASKS:-1} != 1 ]]; then
    echo "This benchmark requires one MPI rank." >&2
    exit 2
fi

compare_binary=
if [[ $mode == short || $mode == reverse || $mode == collaborator ]]; then
    compare_binary=$(realpath -e "${RHMC_COMPARE_BINARY:-$(dirname "$recursive_binary")/rhmcTrajectoryCompare}")
fi

# Each array/job gets its own directory so results from different builds cannot
# silently mix. Set RHMC_RESULTS_DIR when submitting individual repetitions.
results_directory="${RHMC_RESULTS_DIR:-$run_directory/rhmc_results/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
mkdir -p "$results_directory"
results_directory=$(realpath -e "$results_directory")
echo "RHMC results directory: $results_directory"
echo "Legacy binary: $legacy_binary"
echo "Recursive binary: $recursive_binary"

parameter_file="$run_directory/../parameter/tests/rhmcBenchmark.param"
gauge_file="$run_directory/../test_conf/l528f21b6315m00282m0759_001.1610"

if [[ ! -r $parameter_file || ! -r $gauge_file ]]; then
    echo "The run directory does not resolve the RHMC benchmark inputs:" >&2
    echo "  $parameter_file" >&2
    echo "  $gauge_file" >&2
    exit 2
fi

if ! grep -Eq '^[[:space:]]*Lattice[[:space:]]*=[[:space:]]*52[[:space:]]+52[[:space:]]+52[[:space:]]+8([[:space:]]|$)' "$parameter_file"; then
    echo "RHMC benchmark parameter file is not configured for 52^3 x 8." >&2
    exit 2
fi

if ! grep -Eq '^[[:space:]]*Nodes[[:space:]]*=[[:space:]]*1[[:space:]]+1[[:space:]]+1[[:space:]]+1([[:space:]]|$)' "$parameter_file"; then
    echo "RHMC benchmark parameter file is not configured for one MPI rank." >&2
    exit 2
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'V100'; then
    echo "No V100 GPU was detected." >&2
    exit 2
fi

execute_trajectory() {
    local binary=$1
    shift
    local label=recursive
    if [[ $binary == "$legacy_binary" ]]; then label=legacy; fi
    local logfile="$results_directory/${mode}_${repetition:-validation}_${label}_${SLURM_JOB_ID}.log"
    echo "Full trajectory log: $logfile" >&2
    (cd "$run_directory" && "$binary" "$@" 2>&1) | tee -a "$logfile"
}

extract_time() {
    # Only the canonical seconds record is suitable for timing comparisons.
    # Older StopWatch autoFormat output rounds minutes; do not reconstruct it.
    awk '
        /RHMC trajectory time:/ {
            count++
            sub(/^.*RHMC trajectory time: */, "")
            if ($0 ~ /^[0-9]+([.][0-9]+)?s[[:space:]]*$/) {
                seconds = $0 + 0
                valid = seconds > 0
            }
        }
        END {
            if (count != 1 || !valid) {
                print "Missing/invalid seconds record: rebuild BOTH rhmcBenchmark binaries with the common driver." > "/dev/stderr"
                exit 1
            }
            printf "%.3f\n", seconds
        }'
}

extract_status() {
    grep -Eo 'RHMC trajectory status: (ACCEPTED|REJECTED)' | tail -n 1 | sed 's/^RHMC trajectory status: //'
}

extract_delta_h() {
    sed -n 's/.*Delta H = *//p' | tail -n 1 | awk '{print $1}'
}

if [[ $mode == "validate" ]]; then
    echo "Initial legacy RHMC trajectory"
    if ! legacy_output=$(execute_trajectory "$legacy_binary"); then
        printf '%s\n' "$legacy_output" >&2
        exit 1
    fi
    printf '%s\n' "$legacy_output" | grep -E 'Hamilton|H_(initial|final)|Delta H|Initial (plaquette|rectangle|Polyakov)|Final (plaquette|rectangle|Polyakov)|RHMC trajectory (time|status)' || true

    echo
    echo "Initial recursive RHMC trajectory"
    if ! recursive_output=$(execute_trajectory "$recursive_binary"); then
        printf '%s\n' "$recursive_output" >&2
        exit 1
    fi
    printf '%s\n' "$recursive_output" | grep -E 'Hamilton|H_(initial|final)|Delta H|Initial (plaquette|rectangle|Polyakov)|Final (plaquette|rectangle|Polyakov)|RHMC trajectory (time|status)' || true

    legacy_time=$(printf '%s\n' "$legacy_output" | extract_time || true)
    recursive_time=$(printf '%s\n' "$recursive_output" | extract_time || true)
    legacy_status=$(printf '%s\n' "$legacy_output" | extract_status || true)
    recursive_status=$(printf '%s\n' "$recursive_output" | extract_status || true)
    legacy_delta_h=$(printf '%s\n' "$legacy_output" | extract_delta_h || true)
    recursive_delta_h=$(printf '%s\n' "$recursive_output" | extract_delta_h || true)

    if [[ -z $legacy_time || -z $recursive_time || -z $legacy_status || -z $recursive_status ]]; then
        echo "Validation run did not emit the required internal timing/status records." >&2
        exit 1
    fi

    printf '\nLegacy validation:    %ss, %s, Delta H=%s\n' "$legacy_time" "$legacy_status" "${legacy_delta_h:-unavailable}"
    printf 'Recursive validation: %ss, %s, Delta H=%s\n' "$recursive_time" "$recursive_status" "${recursive_delta_h:-unavailable}"

    if [[ $legacy_status != "$recursive_status" ]]; then
        echo "WARNING: legacy and recursive acceptance decisions differ." >&2
    fi

    if [[ -n $legacy_delta_h && -n $recursive_delta_h ]] &&
       awk -v legacy="$legacy_delta_h" -v recursive="$recursive_delta_h" 'BEGIN {
           difference = legacy - recursive;
           if (difference < 0) difference = -difference;
           legacy_abs = legacy < 0 ? -legacy : legacy;
           recursive_abs = recursive < 0 ? -recursive : recursive;
           scale = legacy_abs > recursive_abs ? legacy_abs : recursive_abs;
           threshold = 1e-4 + 1e-3 * scale;
           exit !(difference > threshold);
       }'; then
        echo "WARNING: substantial legacy/recursive Delta H discrepancy detected." >&2
    fi

    exit 0
fi

print_diagnostics() {
    grep -E 'Hamilton|H_(initial|final)|Delta H|Initial (plaquette|rectangle|Polyakov)|Final (plaquette|rectangle|Polyakov)|RHMC trajectory (time|status)|Reversibility max|Difference in saved and evolved Gaugefields' || true
}

run_short_check() {
    echo "One-step deterministic legacy trajectory"
    legacy_output=$(execute_trajectory "$legacy_binary" "$parameter_file" no_md=1 always_acc=1 --gauge-output "$results_directory/rhmc_short_legacy")
    printf '%s\n' "$legacy_output" | print_diagnostics

    echo
    echo "One-step deterministic recursive trajectory"
    recursive_output=$(execute_trajectory "$recursive_binary" "$parameter_file" no_md=1 always_acc=1 --gauge-output "$results_directory/rhmc_short_recursive")
    printf '%s\n' "$recursive_output" | print_diagnostics

    echo
    echo "Entire final gauge-field comparison"
    (cd "$run_directory" && "$compare_binary" "$results_directory/rhmc_short_legacy" "$results_directory/rhmc_short_recursive")
    echo "RHMC short-trajectory comparison: PASS"
}

if [[ $mode == "short" ]]; then
    run_short_check
    exit 0
fi

if [[ $mode == "reverse" ]]; then
    echo "One-step legacy reversibility trajectory"
    legacy_output=$(execute_trajectory "$legacy_binary" "$parameter_file" no_md=1 always_acc=1 --reverse --gauge-output "$results_directory/rhmc_reverse_legacy")
    printf '%s\n' "$legacy_output" | print_diagnostics

    echo
    echo "Legacy forward/reverse gauge-field comparison"
    (cd "$run_directory" && "$compare_binary" "$gauge_file" "$results_directory/rhmc_reverse_legacy")

    echo
    echo "One-step recursive reversibility trajectory"
    recursive_output=$(execute_trajectory "$recursive_binary" "$parameter_file" no_md=1 always_acc=1 --reverse --gauge-output "$results_directory/rhmc_reverse_recursive")
    printf '%s\n' "$recursive_output" | print_diagnostics

    echo
    echo "Recursive forward/reverse gauge-field comparison"
    (cd "$run_directory" && "$compare_binary" "$gauge_file" "$results_directory/rhmc_reverse_recursive")
    exit 0
fi

if [[ $mode == "scaling" ]]; then
    printf 'Implementation  step_size  no_md  Delta_H\n'
    for implementation in legacy recursive; do
        if [[ $implementation == legacy ]]; then
            binary=$legacy_binary
        else
            binary=$recursive_binary
        fi

        for setup in '0.02 1' '0.01 2' '0.005 4'; do
            read -r step_size no_md <<< "$setup"
            output=$(execute_trajectory "$binary" "$parameter_file" "step_size=$step_size" "no_md=$no_md" always_acc=1)
            delta_h=$(printf '%s\n' "$output" | extract_delta_h || true)
            if [[ -z $delta_h ]]; then
                printf '%s\n' "$output" >&2
                echo "No Delta H found for $implementation step_size=$step_size." >&2
                exit 1
            fi
            printf '%-15s %-10s %-5s %s\n' "$implementation" "$step_size" "$no_md" "$delta_h"
        done
    done
    exit 0
fi

run_timed() {
    local label=$1
    local binary=$2
    local repetition=$3
    local output
    local trajectory_time

    if [[ $mode == collaborator ]]; then
        echo "Running $label RHMC trajectory (single timing pair)" >&2
    else
        echo "Running $label RHMC repetition $repetition/3" >&2
    fi
    if ! output=$(execute_trajectory "$binary"); then
        printf '%s\n' "$output" >&2
        echo "$label RHMC repetition $repetition failed." >&2
        exit 1
    fi

    trajectory_time=$(printf '%s\n' "$output" | extract_time || true)
    if [[ -z $trajectory_time ]]; then
        printf '%s\n' "$output" >&2
        echo "$label RHMC repetition $repetition emitted no internal trajectory time." >&2
        exit 1
    fi

    printf '%s\n' "$trajectory_time"
}


# Test exit codes determine correctness; benchmark speed never determines PASS.
run_check() {
    local label=$1
    shift
    local logfile="$results_directory/${label// /_}.log"
    if ! (cd "$run_directory" && "$@" 2>&1) | tee "$logfile"; then
        echo "$label: FAIL" >&2
        return 1
    fi
    echo "$label: PASS"
}

if [[ $mode == collaborator ]]; then
    test_directory=$(dirname "$recursive_binary")
    legacy_force_binary=$(dirname "$legacy_binary")/hisqForceBenchmark
    for executable in "$test_directory/hisqSmearingTest" "$test_directory/hisqForce" \
                      "$test_directory/hisqSmearingRecursiveTest" \
                      "$test_directory/hisqForceBenchmark" "$test_directory/hisqForceLargeCompare" \
                      "$legacy_force_binary" "$compare_binary"; do
        if [[ ! -x $executable ]]; then
            echo "Missing test executable: $executable (see README_RHMC_BENCHMARK.md)" >&2
            exit 2
        fi
    done
    for input in "$project_directory/test_conf/gauge12750" \
                 "$project_directory/test_conf/smearing_reference_conf" \
                 "$project_directory/test_conf/force_reference" \
                 "$project_directory/parameter/tests/hisqSmearingTest.param" \
                 "$project_directory/parameter/tests/hisqForce.param" \
                 "$project_directory/parameter/tests/hisqSmearingRecursiveTest.param" \
                 "$project_directory/parameter/tests/hisqForce_bench.param"; do
        if [[ ! -r $input ]]; then
            echo "Missing test input: $input" >&2
            exit 2
        fi
    done

    run_check "Historical HISQ smearing regression" "$test_directory/hisqSmearingTest" \
        "$project_directory/parameter/tests/hisqSmearingTest.param" "Lattice=8 8 8 4" "Nodes=1 1 1 1"
    run_check "Historical HISQ force regression" "$test_directory/hisqForce" \
        "$project_directory/parameter/tests/hisqForce.param" "Lattice=8 8 8 4" "Nodes=1 1 1 1"
    run_check "Recursive HISQ smearing regression" "$test_directory/hisqSmearingRecursiveTest"

    # These force dumps are correctness inputs; their timings are not reported.
    run_check "Legacy force dump" "$legacy_force_binary" \
        --force-output "$results_directory/force_large_legacy"
    run_check "Recursive force dump" "$test_directory/hisqForceBenchmark" \
        --force-output "$results_directory/force_large_recursive"
    run_check "Large-lattice HISQ force comparison" "$test_directory/hisqForceLargeCompare" \
        --legacy-force "$results_directory/force_large_legacy" \
        --recursive-force "$results_directory/force_large_recursive"

    run_short_check
    echo "Collaborator correctness checks: PASS"

    repetition=1
    legacy_time=$(run_timed legacy "$legacy_binary" "$repetition")
    recursive_time=$(run_timed recursive "$recursive_binary" "$repetition")
    printf 'legacy %s\nrecursive %s\n' "$legacy_time" "$recursive_time" > "$results_directory/rhmc_single_pair.dat"
    printf '\nLegacy RHMC time:    %ss\nRecursive RHMC time: %ss\n' "$legacy_time" "$recursive_time"
    awk -v legacy="$legacy_time" -v recursive="$recursive_time" 'BEGIN {
        printf "RHMC speedup: %.4fx\n", legacy / recursive
        printf "RHMC trajectory time reduction: %.2f%% (single timing pair)\n", 100 * (legacy - recursive) / legacy
    }'
    echo "Collaborator check: PASS"
    exit 0
fi

if [[ $repetition == 2 ]]; then
    recursive_time=$(run_timed recursive "$recursive_binary" "$repetition")
    legacy_time=$(run_timed legacy "$legacy_binary" "$repetition")
else
    legacy_time=$(run_timed legacy "$legacy_binary" "$repetition")
    recursive_time=$(run_timed recursive "$recursive_binary" "$repetition")
fi

result_file="$results_directory/rhmc_timing_${repetition}.dat"
temporary_file="$result_file.${SLURM_JOB_ID}.tmp"

printf 'legacy %s\nrecursive %s\n' "$legacy_time" "$recursive_time" > "$temporary_file"
mv "$temporary_file" "$result_file"

printf '\nRHMC timing repetition %s\n' "$repetition"
printf 'Legacy:    %ss\n' "$legacy_time"
printf 'Recursive: %ss\n' "$recursive_time"
printf 'Saved: %s\n' "$result_file"
