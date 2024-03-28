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

# Clean up any previous runs.
for killFile in OUT_*; do
    if [ -f ${killFile} ]; then rm ${killFile}; fi
done
for killFile in runERR_*; do
    if [ -f ${killFile} ]; then rm ${killFile}; fi
done

# The correlator tests require correlatorNorm files.
cp ../parameter/ua_s8t4.norm .
cp ../parameter/us_s8t4.norm .

# Associative arrays are like Python dictionaries. Associate to each test script the number of GPUs you want to use,
# and give any special directions needed for the splitting.
declare -A testRoutines
testRoutines[_bulkIndexerTest]="4k"
testRoutines[_colorElectricCorrTest]="4s"
testRoutines[_dslashMultiTest]="4k"
testRoutines[_gradientFlowTest]="4k"           # Apparently multi node doesn't work?
testRoutines[_haloTest]="1"
testRoutines[_mixedPrecInverterTest]="4k"

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
echo "${cyan}Now try runTests_1.bash.${endc}"
