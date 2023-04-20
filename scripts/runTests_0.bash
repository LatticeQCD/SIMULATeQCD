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
cp ../parameter/UA_s8t4.norm .
cp ../parameter/US_s8t4.norm .

# Associative arrays are like Python dictionaries. Associate to each test script the number of GPUs you want to use,
# and give any special directions needed for the splitting.
declare -A testRoutines
testRoutines[_BulkIndexerTest]="4k"
testRoutines[_ColorElectricCorrTest]="4s"
testRoutines[_DslashMultiTest]="4k"
testRoutines[_gradientFlowTest]="4k"           # Apparently multi node doesn't work?
testRoutines[_HaloTest]="1"
testRoutines[_MixedPrecInverterTest]="4k"

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
