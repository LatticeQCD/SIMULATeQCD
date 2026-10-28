#!/usr/bin/env bash
# CPU workstation or allocated CPU job; never submits jobs or compiles.
# CPU_LAUNCH / CPU_LAUNCH_SERIAL prefix the MPI tests / single-rank tests (e.g. 'srun -n 1').
# The 8^3x4 fixtures cannot be split (local extent must exceed 2*HaloDepth = 8), so keep
# CPU_NODES at '1 1 1 1'; runCpuMultiRankTests.sh checks multi-rank runs on 24^3x4.
# Threads per rank follow OMP_NUM_THREADS; CPU_RESULTS overrides the output directory.
set -euo pipefail
export LC_ALL=C
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/runCpuRhmcTests.sh BUILD_DIR [references|historical]" >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
build=$(cd -- "$1" && pwd -P)
mode=${2:-references}
nodes=${CPU_NODES:-1 1 1 1}
read -r -a launch <<< "${CPU_LAUNCH:-}"
read -r -a launchSerial <<< "${CPU_LAUNCH_SERIAL:-}"
case "$mode" in references|historical) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac
grep -qx 'BACKEND:STRING=cpu' "$build/CMakeCache.txt" || { echo "Not a CPU build" >&2; exit 2; }
for target in cpuBackendTest dslashTest inverterTest rhmcTest; do
    [[ -x "$build/testing/$target" ]] || { echo "Missing $target; build cpuRhmcTests first" >&2; exit 2; }
done
out=${CPU_RESULTS:-"$build/cpu-rhmc-results"}
mkdir -p "$out/run" "$out/parameter/tests" "$out/test_conf"
out=$(cd -- "$out" && pwd -P)
# Clear prior results so a failed rerun cannot leave an old PASS or summary.
rm -f -- "$out/RESULT.txt" "$out/NUMBERS.txt" \
    "$out/"{runtime,dslash,inverter,rhmc}.{log,txt}
cp "$repo/parameter/tests/"{dslashTest,inverterTest,rhmcTrajectoryReference,rhmcTest}.param "$out/parameter/tests/"
cp "$repo/parameter/"{sample_force_test,sample_eo}.rat "$out/parameter/"
fixtures=(gauge12750)
if [[ $mode == historical ]]; then fixtures+=(rhmc_reference_conf); fi
for fixture in "${fixtures[@]}"; do
    [[ -s "$repo/test_conf/$fixture" ]] || { echo "Missing input: $fixture" >&2; exit 2; }
    if head -c 80 "$repo/test_conf/$fixture" | grep -q 'git-lfs.github.com/spec'; then
        echo "Missing LFS contents for $fixture" >&2; exit 2
    fi
    ln -sfn "$repo/test_conf/$fixture" "$out/test_conf/$fixture"
done
cp "$build/CMakeCache.txt" "$out/CMakeCache.txt"
echo "Nodes=$nodes launch=${CPU_LAUNCH:-none} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}" > "$out/layout.txt"
git -C "$repo" rev-parse HEAD > "$out/commit.txt"
git -C "$repo" status --short > "$out/worktree-status.txt"
sha256sum "$build/testing/"{cpuBackendTest,dslashTest,inverterTest,rhmcTest} \
    "$out/parameter/"*.rat "$out/parameter/tests/"*.param \
    "$out/test_conf/"* > "$out/sha256.txt"
echo "Logs and numerical results: $out"
echo "Live output below is also saved in each .log file."
cd "$out/run"
run_test() {
    local label=$1 verdict=$2
    shift 2
    echo "--- $label ---"
    if ! "$@" 2>&1 | tee "$out/$label.log"; then
        echo "FAIL: $label (see $out/$label.log)" >&2; exit 1
    fi
    # Strip colors before evaluating test verdicts.
    sed -E 's/\x1B\[[0-9;]*[mK]//g' "$out/$label.log" > "$out/$label.txt"
    if ! grep -Fq "$verdict" "$out/$label.txt"; then
        echo "FAIL: missing verdict: $verdict" >&2; exit 1
    fi
    if grep -Eiq '(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)|: FAIL|maximum.*iteration|max.*iterations?.*reached|did not converge' "$out/$label.txt"; then
        echo "FAIL: nonfinite value, failed check, or solver limit in $label" >&2; exit 1
    fi
}
run_test runtime 'CPU runtime regression: PASS' "${launchSerial[@]}" "$build/testing/cpuBackendTest"
# dslashTest uses HaloDepth 0, so it is a single-rank test (as in runTestsSingleGPU.bash).
run_test dslash 'Dslash regression: PASS' "${launchSerial[@]}" "$build/testing/dslashTest" \
    ../parameter/tests/dslashTest.param 'Nodes=1 1 1 1'
run_test inverter 'Inverter regression: PASS' "${launch[@]}" "$build/testing/inverterTest" \
    ../parameter/tests/inverterTest.param "Nodes=$nodes" 'Lattice=8 8 8 4'
if [[ $mode == references ]]; then
    refs="$repo/test_conf/rhmc_trajectory_references"
    for mu in 0 0.4; do
        for steps in 1 10; do
            for ext in txt nersc; do
                file="$refs/double_mu${mu}_md${steps}.$ext"
                [[ -s "$file" ]] || { echo "Missing reference: $file" >&2; exit 2; }
                sha256sum "$file" >> "$out/sha256.txt"
            done
        done
    done
    echo 'Double precision: one trajectory per case, 1/10 MD steps, mu0=0/0.4.'
    echo 'Each case restarts from gauge12750; these are NOT ten consecutive trajectories.'
    run_test rhmc 'RHMC legacy-reference regression suite: PASS' "${launch[@]}" "$build/testing/rhmcTest" \
        --trajectory-regression-suite ../parameter/tests/rhmcTrajectoryReference.param "$refs" --double-only \
        "Nodes=$nodes"
else
    echo 'Historical 10^4 test: reversibility checks followed by ten complete updates.'
    run_test rhmc 'ALL TESTS PASSED' "$build/testing/rhmcTest" ../parameter/tests/rhmcTest.param
fi
grep -E 'Delta H|Reference plaquette|Reference Polyakov|CHI_UD|RHMC observable|full-field max|tolerance|relative error|PASS|acceptance|Ploop' \
    "$out/inverter.txt" "$out/rhmc.txt" > "$out/NUMBERS.txt" || true
echo "CPU RHMC $mode suite: PASS" | tee "$out/RESULT.txt"
echo "Numerical summary: $out/NUMBERS.txt"
