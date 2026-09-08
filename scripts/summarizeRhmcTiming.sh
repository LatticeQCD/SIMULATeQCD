#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

run_directory=$(realpath -e "${1:-.}")

read_time() {
    local repetition=$1
    local implementation=$2
    local result_file="$run_directory/rhmc_timing_${repetition}.dat"

    if [[ ! -r $result_file ]]; then
        echo "Missing timing result: $result_file" >&2
        exit 1
    fi

    awk -v implementation="$implementation" '
        $1 == implementation {
            count++
            if (NF != 2 || $2 !~ /^[0-9]+([.][0-9]+)?$/ || $2 <= 0) invalid = 1
            value = $2
        }
        END {
            if (count != 1 || invalid) {
                print "Invalid timing record in " FILENAME > "/dev/stderr"
                exit 1
            }
            print value
        }' "$result_file"
}

median_of_three() {
    printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n '2p'
}

declare -a legacy_times
declare -a recursive_times

for repetition in 1 2 3; do
    legacy_times[$((repetition - 1))]=$(read_time "$repetition" legacy)
    recursive_times[$((repetition - 1))]=$(read_time "$repetition" recursive)
done

legacy_median=$(median_of_three "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}")
recursive_median=$(median_of_three "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}")

speedup=$(awk -v legacy="$legacy_median" -v recursive="$recursive_median" 'BEGIN { printf "%.6f", legacy / recursive }')
reduction=$(awk -v legacy="$legacy_median" -v recursive="$recursive_median" 'BEGIN { printf "%.2f", 100.0 * (legacy - recursive) / legacy }')

printf 'Legacy RHMC times:    %ss %ss %ss\n' "${legacy_times[0]}" "${legacy_times[1]}" "${legacy_times[2]}"
printf 'Recursive RHMC times: %ss %ss %ss\n\n' "${recursive_times[0]}" "${recursive_times[1]}" "${recursive_times[2]}"
printf 'Median legacy RHMC time:    %ss\n' "$legacy_median"
printf 'Median recursive RHMC time: %ss\n\n' "$recursive_median"
printf 'RHMC speedup: %sx\n' "$speedup"
printf 'RHMC trajectory time reduction: %s%%\n' "$reduction"

# Show spread as well as the median; three samples are not an error estimate.
legacy_range=$(printf '%s\n' "${legacy_times[@]}" | sort -n | awk 'NR == 1 { low = $1 } { high = $1 } END { printf "%ss .. %ss", low, high }')
recursive_range=$(printf '%s\n' "${recursive_times[@]}" | sort -n | awk 'NR == 1 { low = $1 } { high = $1 } END { printf "%ss .. %ss", low, high }')
printf '\nLegacy timing range:    %s\nRecursive timing range: %s\n' "$legacy_range" "$recursive_range"
printf 'Paired time reductions:'
for index in 0 1 2; do
    awk -v legacy="${legacy_times[$index]}" -v recursive="${recursive_times[$index]}" \
        'BEGIN { printf " %.2f%%", 100 * (legacy - recursive) / legacy }'
done
printf '\n'
