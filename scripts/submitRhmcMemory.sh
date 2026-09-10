#!/usr/bin/env bash

# Submit one legacy-vs-recursive internal MemoryManagement report.

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_directory=$(dirname "$script_directory")
project_directory=$(dirname "$source_directory")
run_directory="$project_directory/testrun"

legacy_binary="$project_directory/buildSIMULATeQCD_legacy_1732861/testing/rhmcBenchmark"
recursive_binary="$project_directory/buildSIMULATeQCD/testing/rhmcBenchmark"

sbatch --chdir="$run_directory" \
    "$script_directory/runRhmcMemoryComparison.sh" \
    "$legacy_binary" "$recursive_binary" "$run_directory"
