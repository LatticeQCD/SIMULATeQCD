#!/usr/bin/env bash

# Submit three serialized legacy-vs-recursive RHMC timing comparisons and
# automatically generate the timing summary/PDF after all three finish.

set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_directory=$(dirname "$script_directory")
project_directory=$(dirname "$source_directory")
run_directory="$project_directory/testrun"

legacy_binary="$project_directory/buildSIMULATeQCD_legacy_1732861/testing/rhmcBenchmark"
recursive_binary="$project_directory/buildSIMULATeQCD/testing/rhmcBenchmark"

submission=$(sbatch --parsable --array=1-3%1 --chdir="$run_directory" \
    "$script_directory/runRhmcTimingComparison.sh" timing \
    "$legacy_binary" "$recursive_binary" "$run_directory")
timing_job=${submission%%;*}
results_directory="$run_directory/rhmc_results/$timing_job"

printf -v summary_command 'bash %q %q' \
    "$script_directory/summarizeRhmcTiming.sh" "$results_directory"

summary_submission=$(sbatch --parsable \
    --dependency="afterany:$timing_job" \
    --ntasks=1 --cpus-per-task=1 --time=0-00:05:00 \
    --partition=volta_devel --job-name=RhmcSummary \
    --chdir="$run_directory" \
    --output="$run_directory/RhmcSummary_%j.out" \
    --error="$run_directory/RhmcSummary_%j.err" \
    --wrap="$summary_command")
summary_job=${summary_submission%%;*}

echo "RHMC timing array submitted: $timing_job"
echo "Automatic summary job submitted: $summary_job"
echo "Final PDF: $results_directory/rhmc_timing_gain.pdf"
