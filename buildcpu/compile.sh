#!/usr/bin/env bash
# Usage: bash buildcpu/compile.sh [portable|native|avx512] [parallel_jobs]
# Run from any directory. Uses the caller's C++ compiler and MPI environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir=$(cd -- "$script_dir/.." && pwd -P)
architecture=${1:-portable}
jobs=${2:-4}
if [[ $# -gt 2 || ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash buildcpu/compile.sh [portable|native|avx512] [parallel_jobs]" >&2
    exit 2
fi
case "$architecture" in
    portable|native|avx512) ;;
    *) echo "Unknown CPU architecture: $architecture" >&2; exit 2 ;;
esac
build_dir="$script_dir/$architecture"
cmake -S "$source_dir" -B "$build_dir" \
    -DBACKEND=cpu -DCPU_ARCHITECTURE="$architecture" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_AWARE_MPI=OFF -DUSE_GPU_P2P=OFF \
    -DUSE_NCCL=OFF -DUSE_MARKER=OFF
cmake --build "$build_dir" --target cpuRhmcTests --parallel "$jobs"
echo "Build complete: $build_dir"
echo "Run inside a compatible CPU allocation:"
printf 'bash %q %q\n' "$source_dir/scripts/runCpuHisqTests.sh" "$build_dir"
printf 'bash %q %q\n' "$source_dir/scripts/runCpuRhmcTests.sh" "$build_dir"
