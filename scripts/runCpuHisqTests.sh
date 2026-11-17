#!/usr/bin/env bash
# Run on a CPU machine or inside an allocated CPU batch job. No job submission.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/runCpuHisqTests.sh /absolute/path/to/build-cpu" >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
build=$(cd -- "$1" && pwd -P)
if ! grep -qx 'BACKEND:STRING=cpu' "$build/CMakeCache.txt"; then
    echo "FAIL: this is not a BACKEND=cpu build" >&2
    exit 2
fi
for target in cpuBackendTest hisqSmearingTest hisqSmearingRecursiveTest hisqForce; do
    [[ -x "$build/testing/$target" ]] || { echo "Missing executable: $target" >&2; exit 2; }
done
for fixture in gauge12750 smearing_reference_conf force_reference; do
    [[ -s "$repo/test_conf/$fixture" ]] || { echo "Missing fixture: $fixture" >&2; exit 2; }
    if head -c 80 "$repo/test_conf/$fixture" | grep -q 'git-lfs.github.com/spec'; then
        echo "FAIL: $fixture is an LFS pointer; obtain the actual reference file first" >&2
        exit 2
    fi
done
out="$build/cpu-hisq-results"
mkdir -p "$out/run" "$out/test_conf" "$out/parameter/tests"
# Reuse the readable output path, without leaving verdicts/logs from an older run.
rm -f -- "$out/RESULT.txt" "$out/cpuBackendTest.log" \
    "$out/hisqSmearingTest.log" "$out/hisqSmearingRecursiveTest.log" "$out/hisqForce.log"
cp "$repo/parameter/tests/hisqForce.param" "$out/parameter/tests/"
cp "$repo/parameter/tests/hisqSmearingTest.param" "$out/parameter/tests/"
cp "$repo/parameter/sample_force_test.rat" "$out/parameter/"
for fixture in gauge12750 smearing_reference_conf force_reference; do
    ln -sfn "$repo/test_conf/$fixture" "$out/test_conf/$fixture"
done
# The recursive test's second stage normally uses 52^3 x 8. Keep this CPU
# correctness run at 8^3 x 4, with the same saved input as its historical stage.
cat > "$out/parameter/tests/hisqSmearingRecursiveTest.param" <<'PARAM'
Lattice = 8 8 8 4
Nodes = 1 1 1 1
Gaugefile = ../test_conf/gauge12750
Topology = 0 0 0 0
endianness = auto
prec_out = 0
measurements_dir = ./
use_unit_conf = 0
PARAM
cp "$build/CMakeCache.txt" "$out/CMakeCache.txt"
git -C "$repo" rev-parse HEAD > "$out/commit.txt"
git -C "$repo" diff --binary > "$out/tracked-changes.patch"
git -C "$repo" status --short > "$out/worktree-status.txt"
sha256sum "$build/testing/"{cpuBackendTest,hisqSmearingTest,hisqSmearingRecursiveTest,hisqForce} \
    "$repo/test_conf/"{gauge12750,smearing_reference_conf,force_reference} \
    "$out/parameter/sample_force_test.rat" > "$out/sha256.txt"
echo "Logs and generated fields: $out"
cd "$out/run"
run_test() {
    local target=$1 verdict=$2
    if ! "$build/testing/$target" > "$out/$target.log" 2>&1; then
        echo "FAIL: $target (see $out/$target.log)" >&2
        tail -n 30 "$out/$target.log" >&2
        exit 1
    fi
    if ! grep -Fq "$verdict" "$out/$target.log"; then
        echo "FAIL: $target did not print its required PASS verdict" >&2
        exit 1
    fi
    if grep -Eiq '(^|[^[:alnum:]_])(nan|[-+]?inf(inity)?)([^[:alnum:]_]|$)|: FAIL' "$out/$target.log"; then
        echo "FAIL: $target logged a nonfinite value or failed check" >&2
        exit 1
    fi
    echo "$target: PASS"
}
run_test cpuBackendTest 'CPU runtime regression: PASS'
run_test hisqSmearingTest 'Historical HISQ smearing regression: PASS'
run_test hisqSmearingRecursiveTest 'Recursive HISQ smearing regression: PASS'
run_test hisqForce 'Historical HISQ force regression: PASS'
echo "CPU HISQ regression suite: PASS" | tee "$out/RESULT.txt"
