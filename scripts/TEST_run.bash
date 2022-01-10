#!/bin/bash

# 
# TEST_run.bash                                                               
# 
# D. Clarke, 16 Jan 2020 
# 
# A script to run all the test routines. Must be run in the testing subfolder. 
# 

cred="\e[91m"
cyan="\e[36m"
endc="\e[0m"


# ----------------------------------------------------------------------------------------------------- DEFINE FUNCTIONS


# use srun inside a slurm script, and mpiexec everywhere else
run_command="mpiexec --oversubscribe -np";

function runTestRoutine {
    routine="$1"
    nGPUs="$2"
    nodeLayout="$3"
    paramFile="${routine:1}".param
    outFile='OUT'"${routine}"
    errFile='runERR'"${routine}"
    echo '  '${routine}' using '${nGPUs}' GPUs and layout '${nodeLayout}
    $run_command ${nGPUs} ./${routine} "../parameter/tests/${paramFile}" "${nodeLayout}" >> ${outFile} 2>> ${errFile}
    echo " " >> ${outFile}
    # This is just to remove these files if they are empty.
    if [ ! -s ${errFile} ]; then rm ${errFile}; fi
    if [ ! -s ${outFile} ]; then rm ${outFile}; fi
}

function runTestRoutineNoParam {
    routine="$1"
    nGPUs="$2"
    outFile='OUT'"${routine}"
    errFile='runERR'"${routine}"
    echo '  '${routine}' using '${nGPUs}' GPUs.'
    $run_command ${nGPUs} ./${routine} >> ${outFile} 2>> ${errFile}
    echo " " >> ${outFile}
    # This is just to remove these files if they are empty.
    if [ ! -s ${errFile} ]; then rm ${errFile}; fi
    if [ ! -s ${outFile} ]; then rm ${outFile}; fi
}

# The GPUkey tells you the number of processors and sometimes extra info about the node layout:
#    s: Split only in spatial directions. Useful for observables like the Polyakov loop, where one prefers not to
#       split the lattice in the Euclidean time direction.
#    k: No long directions. Useful whenever 4 does not divide a lattice extension.
function runByLayout {
    routine="$1"
    GPUkey="$2"
    if [ $GPUkey == "4" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 2 1 1 2" \
                            "Nodes = 1 2 2 1" "Nodes = 1 2 1 2" "Nodes = 1 1 2 2" \
                            "Nodes = 4 1 1 1" "Nodes = 1 4 1 1" "Nodes = 1 1 4 1" "Nodes = 1 1 1 4")
        nGPUs=4
    elif [ $GPUkey == "4s" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 1 2 2 1" \
                            "Nodes = 4 1 1 1" "Nodes = 1 4 1 1" "Nodes = 1 1 4 1")
        nGPUs=4
    elif [ $GPUkey == "4k" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 2 1 1 2" \
                            "Nodes = 1 2 2 1" "Nodes = 1 2 1 2" "Nodes = 1 1 2 2")
        nGPUs=4
    elif [ $GPUkey == "4ks" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 1 2 2 1")
        nGPUs=4
    elif [ $GPUkey == "2" ]; then
        declare -a layouts=("Nodes = 2 1 1 1" "Nodes = 1 2 1 1" "Nodes = 1 1 2 1" "Nodes = 1 1 1 2")
        nGPUs=2
    elif [ $GPUkey == "2s" ]; then
        declare -a layouts=("Nodes = 2 1 1 1" "Nodes = 1 2 1 1" "Nodes = 1 1 2 1")
        nGPUs=2
    elif [ $GPUkey == "1" ]; then
        declare -a layouts=("Nodes = 1 1 1 1")
        nGPUs=1
    else
        echo -e "  ${cred}ERROR--Invalid GPUkey encountered by runByLayout!${endc}"
    fi
    for layout in "${layouts[@]}"; do
        runTestRoutine "$routine" "$nGPUs" "$layout"
    done
}

# ----------------------------------------------------------------------------------------------------------------- MAIN

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
testRoutines[_CompressionTest]="4k"
testRoutines[_correlatorTest]="1"             # All tests involving correlators work only for one GPU.
testRoutines[_cudaIpcTest]="2"                # Only works for two GPUs.
testRoutines[_DslashTest]="1"
testRoutines[_DslashMultiTest]="4k"
testRoutines[_GeneralFunctorTest]="4k"
testRoutines[_gfixplcTest]="1"
testRoutines[_gradientFlowTest]="1"           # Apparently multi node doesn't work?
testRoutines[_HaloTest]="1"
testRoutines[_InverterTest]="4k"
testRoutines[_LinkPathTest]="4k"
testRoutines[_MixedPrecInverterTest]="4k"
testRoutines[_PureGaugeHmcTest]="4ks"
testRoutines[_SaveTest]="4k"
testRoutines[_SpinorHaloTest]="4k"
testRoutines[_StackedSpinorTest]="4k"

# In principle instead of having separate mains to test multiple GPU, some of these could be consolidated into one main
# that is run repeatedly with different parameters. For now I'm leaving this as is. This section is also used for tests
# that don't require a parameter file, like the memManTest.
declare -A testRoutinesNoParam
testRoutinesNoParam[_dotProductTest]="1"
testRoutinesNoParam[_DslashImagmuTest]="1"
testRoutinesNoParam[_gfixTestSingle]="1"
testRoutinesNoParam[_halfPrecMathTest]="1"
testRoutinesNoParam[_HBOR_single_test]="1"
testRoutinesNoParam[_hisqForce]="1"              # The read in construction doesn't seem to work.
testRoutinesNoParam[_hisqForceImagmu]="1"
testRoutinesNoParam[_hisqSmearingTest]="1"
testRoutinesNoParam[_hisqSmearingImagmuTest]="1"
testRoutinesNoParam[_memManTest]="1"
testRoutinesNoParam[_rhmcTest]="1"               # The read in construction doesn't seem to work. Test is a bit too long (30 min)
testRoutinesNoParam[_rhmcTest_4pf]="1"
testRoutinesNoParam[_RndSingleTest]="1"
testRoutinesNoParam[_SimpleFunctorTest]="1"
testRoutinesNoParam[_UtimesUdaggerTest]="1"

# Counting the number of test sets lets us give the user some indication of progress.
numberOfTestRoutines="${#testRoutines[@]}"
numberOfTestRoutines="$((${numberOfTestRoutines}+${#testRoutinesNoParam[@]}))"
numberOfMultRoutines=4
numberOfTestRoutines="$((${numberOfTestRoutines}+${numberOfMultRoutines}))"

# Run the test routines that can vary their layout.
jtest=0
for key in "${!testRoutines[@]}"; do
    ((jtest++))
    echo
    echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
    runByLayout "${key}" "${testRoutines[$key]}"
done

# Run some test routines that have a fixed layout.
for key in "${!testRoutinesNoParam[@]}"; do
    ((jtest++))
    echo
    echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
    runTestRoutineNoParam "${key}" "${testRoutinesNoParam[$key]}"
done

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
$run_command 1 ./${createExec} ../parameter/run.param Nodes="1 1 1 1" > ${createOut} 2> ${createErr}
$run_command 2 ./${createExec} ../parameter/run.param Nodes="1 2 1 1" >> ${createOut} 2>> ${createErr}
$run_command 4 ./${createExec} ../parameter/run.param Nodes="1 2 2 1" >> ${createOut} 2>> ${createErr}
if [ ! -s ${createErr} ]; then rm ${createErr}; fi
if [ ! -s ${createOut} ]; then rm ${createOut}; fi
((jtest++))
echo
echo "${cyan}Test set "${jtest}" of "${numberOfTestRoutines}":${endc}"
runTestRoutineNoParam "_hisqSmearingMulti" "1"

echo
echo "${cyan}All tests done!${endc}"
