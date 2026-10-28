#!/usr/bin/env bash
# Multi-rank consistency check of the CPU RHMC, inside a Slurm allocation (uses srun).
# One trajectory on a 24^3x4 lattice (gauge12750 tiled 3x3x3x1) is saved on one rank,
# then repeated with x/y/z splits (t is never split) and compared with it. The RNG is
# seeded over the global volume, so every layout must agree with the one-rank run to
# rounding. A split direction needs local extent > 2*HaloDepth = 8, which is why the
# 8^3x4 reference suite (runCpuRhmcTests.sh) can only run on one rank.
# Usage: bash scripts/runCpuMultiRankTests.sh BUILD_DIR OUT_DIR
# CPU_REF_THREADS: OpenMP threads of the one-rank reference run (default 32).
set -euo pipefail
export LC_ALL=C
if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/runCpuMultiRankTests.sh BUILD_DIR OUT_DIR" >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
build=$(cd -- "$1" && pwd -P)
mkdir -p "$2"
out=$(cd -- "$2" && pwd -P)
[[ -x "$build/testing/rhmcTest" ]] || { echo "Missing $build/testing/rhmcTest" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "Run inside a Slurm allocation" >&2; exit 2; }
refThreads=${CPU_REF_THREADS:-32}
# label  Nodes  ranks threads
layouts=(
    "x2111    2 1 1 1  2 8"
    "y1211    1 2 1 1  2 8"
    "z1121    1 1 2 1  2 8"
    "xy2211   2 2 1 1  4 4"
    "xyz2221  2 2 2 1  8 2"
)
# After 10 MD steps rounding differences between layouts grow to ~1e-11 in the field
# (job 3806795: <= 7.3e-11) and to ~1e-7 in CHI_UD, whose CG solves stop at residue_meas.
# Same field/observable policy as the 8^3x4 suite; CHI_UD 1e-6 (the suite uses 1e-5).
fieldTolerance=1e-9
observableTolerance=1e-11
chiTolerance=1e-6

mkdir -p "$out/run" "$out/parameter/tests" "$out/test_conf" "$out/references"
cp "$repo/parameter/tests/rhmcTrajectoryReference.param" "$out/parameter/tests/"
cp "$repo/parameter/sample_force_test.rat" "$out/parameter/"
conf=$out/test_conf/gauge12750_24x24x24x4
[[ -s $conf ]] || python3 -I "$repo/scripts/tileNerscConf.py" "$repo/test_conf/gauge12750" "$conf" 3 3 3 1
rm -f -- "$out/RESULT.txt" "$out/references/"* "$out/"*.log
lattice=('Lattice=24 24 24 4' "gauge_file=../test_conf/$(basename "$conf")" 'no_md=10')
param=../parameter/tests/rhmcTrajectoryReference.param
cd "$out/run"

verdict() {  # LOG PATTERN
    sed -E 's/\x1B\[[0-9;]*[mK]//g' "$1" | grep -Fq "$2"
}
status=0
for mu in 0 0.4; do
    prefix=../references/double_mu${mu}_md10
    log=$out/save_mu${mu}.log
    echo "--- mu0=$mu: one-rank reference ($refThreads threads) ---"
    OMP_NUM_THREADS=$refThreads srun --exact -n 1 -c "$refThreads" --cpu-bind=cores \
        "$build/testing/rhmcTest" --save-trajectory-reference "$prefix" double "$param" \
        "${lattice[@]}" "mu0=$mu" 'Nodes=1 1 1 1' > "$log" 2>&1 \
        && verdict "$log" "RHMC trajectory reference: SAVED" \
        || { echo "FAIL: reference mu0=$mu (see $log)"; status=1; continue; }
    pids=()
    for layout in "${layouts[@]}"; do
        read -r label n0 n1 n2 n3 ranks threads <<< "$layout"
        log=$out/${label}_mu${mu}.log
        OMP_NUM_THREADS=$threads srun --exact -n "$ranks" -c "$threads" --cpu-bind=cores \
            "$build/testing/rhmcTest" --check-trajectory-reference "$prefix" double \
            "$fieldTolerance" "$observableTolerance" "$param" \
            "${lattice[@]}" "mu0=$mu" "Nodes=$n0 $n1 $n2 $n3" "--chi-tolerance=$chiTolerance" > "$log" 2>&1 &
        pids+=("$!:$label:$log")
    done
    for entry in "${pids[@]}"; do
        IFS=: read -r pid label log <<< "$entry"
        if wait "$pid" && verdict "$log" "RHMC trajectory reference double mu0=$mu no_md=10: PASS"; then
            echo "$label mu0=$mu: PASS"
        else
            echo "$label mu0=$mu: FAIL (see $log)"; status=1
        fi
    done
done
grep -E "Delta H|full-field max|observable\[0\]|observable\[3\]" "$out/"*.log \
    | sed -E 's/\x1B\[[0-9;]*[mK]//g' > "$out/NUMBERS.txt" || true
if [[ $status -eq 0 ]]; then
    echo "Multi-rank RHMC consistency: PASS" | tee "$out/RESULT.txt"
else
    echo "Multi-rank RHMC consistency: FAIL" | tee "$out/RESULT.txt"
fi
exit $status
