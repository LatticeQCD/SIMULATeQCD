#!/bin/bash

#
# runTests.bash
#
# D. Clarke
#
# The runTests scripts break all the test routines down into < 30 min chunks (at least on Pascal GPUs),
# so that these can easily be run interactively or in a a cluster's debug queue, where the time limit is
# commonly only 30 min.
#

source "../scripts/testingTools.bash"

# Associative arrays are like Python dictionaries. Associate to each test script the number of GPUs you want to use,
# and give any special directions needed for the splitting.
declare -A testRoutines
testRoutines[_compressionTest]="4k"
testRoutines[_confReadWriteTest]="4k"
testRoutines[_correlatorTest]="1"             # All tests involving correlators work only for one GPU.
testRoutines[_cudaIpcTest]="2"                # Only works for two GPUs.
testRoutines[_dslashTest]="1"
testRoutines[_generalFunctorTest]="4k"
testRoutines[_gfixplcTest]="1"
testRoutines[_inverterTest]="4k"
testRoutines[_linkPathTest]="4k"
testRoutines[_pureGaugeHmcTest]="4ks"
testRoutines[_spinorHaloTest]="4k"
testRoutines[_stackedSpinorTest]="4k"

numberOfTestRoutines="${#testRoutines[@]}"
numberOfTestRoutines="$((${numberOfTestRoutines}+${#testRoutinesNoParam[@]}))"

echo
date

# Run the test routines that can vary their layout.
jtest=0
for key in "${!testRoutines[@]}"; do
    ((jtest++))
    echo
    echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
    runByLayout "${key}" "${testRoutines[$key]}"
done

echo
date

echo
echo "${cyan}Now try runTests_2.bash.${endc}"
