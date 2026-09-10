#!/usr/bin/env bash

# Submit the complete legacy-vs-recursive HISQ-force timing comparison.

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_directory=$(dirname "$script_directory")
project_directory=$(dirname "$source_directory")
run_directory="$project_directory/testrun"

legacy_binary="$project_directory/buildSIMULATeQCD_legacy_1732861/testing/hisqForceBenchmark"
recursive_binary="$project_directory/buildSIMULATeQCD/testing/hisqForceBenchmark"

sbatch --chdir="$run_directory" \
    "$script_directory/runHisqForceTimingComparison.sh" \
    "$legacy_binary" "$recursive_binary" "$run_directory"
