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

# In principle instead of having separate mains to test multiple GPU, some of these could be consolidated into one main
# that is run repeatedly with different parameters. For now I'm leaving this as is. This section is also used for tests
# that don't require a parameter file, like the memManTest.
declare -A testRoutinesNoParam
testRoutinesNoParam[_condensateTest]="1"
testRoutinesNoParam[_dotProductTest]="1"
testRoutinesNoParam[_DslashImagmuTest]="1"
testRoutinesNoParam[_gfixTestSingle]="1"
testRoutinesNoParam[_halfPrecMathTest]="1"
testRoutinesNoParam[_HBOR_single_test]="1"
testRoutinesNoParam[_hisqForce]="1"
testRoutinesNoParam[_hisqSmearingTest]="1"
testRoutinesNoParam[_hisqSmearingImagmuTest]="1"
testRoutinesNoParam[_memManTest]="1"
testRoutinesNoParam[_RndSingleTest]="1"
testRoutinesNoParam[_SimpleFunctorTest]="1"
testRoutinesNoParam[_UtimesUdaggerTest]="1"
testRoutinesNoParam[_TaylorMeasurementTest]="4"
testRoutinesNoParam[_weinbergTopTest]="1"

# Counting the number of test sets lets us give the user some indication of progress.
numberOfTestRoutines="$((${#testRoutinesNoParam[@]}))"
numberOfMultRoutines=5
numberOfTestRoutines="$((${numberOfTestRoutines}+${numberOfMultRoutines}))"

echo
date

# Run some test routines that have a fixed layout.
for key in "${!testRoutinesNoParam[@]}"; do
    ((jtest++))
    echo
    echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
    runTestRoutineNoParam "${key}" "${testRoutinesNoParam[$key]}"
done

# These tests need to be rerun with a different parameter.
forceExec="_hisqForce"
forceOut="OUT"${forceExec}
forceErr="runERR"${forceExec}
((jtest++))
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
echo '  '"${forceExec}"' using 1 GPU, mu_I == 0.0'
$run_command 1 ./${forceExec} ../parameter/tests/hisqForce.param Nodes="1 1 1 1" > ${forceOut} 2> ${forceErr}
echo '  '"${forceExec}"' using 1 GPU, mu_I != 0.0'
$run_command 1 ./${forceExec} ../parameter/tests/hisqForceImagMu.param Nodes="1 1 1 1" >> ${forceOut} 2>> ${forceErr}
if [ ! -s ${forceErr} ]; then rm ${forceErr}; fi

rhmcExec="_rhmcTest"
rhmcOut="OUT"${rhmcExec}
rhmcErr="runERR"${rhmcExec}
((jtest++))
echo
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
echo '  '"${rhmcExec}"' using 1 GPU, no_pf == 1'
$run_command 1 ./${rhmcExec} ../parameter/tests/rhmcTest.param Nodes="1 1 1 1" > ${rhmcOut} 2> ${rhmcErr}
echo '  '"${rhmcExec}"' using 1 GPU, no_pf == 4'
$run_command 1 ./${rhmcExec} ../parameter/tests/rhmcTest_4pf.param Nodes="1 1 1 1" >> ${rhmcOut}_4pf 2>> ${rhmcErr}_4pf
if [ ! -s ${rhmcErr} ]; then rm ${rhmcErr}; fi
if [ ! -s ${rhmcErr}_4pf ]; then rm ${rhmcErr}_4pf; fi


# These tests have to be run after their single counterparts, so we have to run them by hand here because the
# associative array is not ordered.
((jtest++))
echo
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
runTestRoutineNoParam "_RndMultipleTest" "2"
((jtest++))
echo
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
runByLayout "_HBOR_multiple_test" "4ks"
((jtest++))
echo
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
runByLayout "_gfixTestMulti" "4s"

# This one is a bit of a special case testing 2 and 4 GPU runs.
createExec="_hisqSmearingMultiCreate"
createOut="OUT"${createExec}
createErr="runERR"${createExec}
$run_command 1 ./${createExec} ../parameter/tests/run.param Nodes="1 1 1 1" > ${createOut} 2> ${createErr}
$run_command 2 ./${createExec} ../parameter/tests/run.param Nodes="1 2 1 1" >> ${createOut} 2>> ${createErr}
$run_command 4 ./${createExec} ../parameter/tests/run.param Nodes="1 2 2 1" >> ${createOut} 2>> ${createErr}
if [ ! -s ${createErr} ]; then rm ${createErr}; fi
if [ ! -s ${createOut} ]; then rm ${createOut}; fi
((jtest++))
echo
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
runTestRoutineNoParam "_hisqSmearingMulti" "1"

echo
date

echo
echo "${cyan}All tests done!${endc}"
