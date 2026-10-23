#!/bin/bash

#
# runTestsMDWF.bash
#
# Runs the Mobius domain-wall fermion (MDWF) tests and then every MDWF program with its example parameter file
# (src/experimental/mdwf/USERGUIDE.md). Compile first, in your build folder,
#     make -j mdwftests mdwf
# then
#     cd testing
#     ../scripts/runTestsMDWF.bash             # tests and programs
#     ../scripts/runTestsMDWF.bash tests       # tests only
#     ../scripts/runTestsMDWF.bash programs    # programs only
# Everything runs on one GPU, launched with run_command from testingTools.bash (mpiexec or srun).
# As for the other runTests scripts, the output of each executable goes to OUT_<name>. A runERR_<name> file is left
# when an executable wrote to stderr or ended with a nonzero exit status. No runERR files: everything passed.
# mdwfKernelSpectrum and mdwfDiracModes read test_conf/nersc.l8t4b3360_bieHB (git lfs).
#

source "../scripts/testingTools.bash"

part="${1:-all}"

# Clean up any previous runs.
for killFile in OUT_mdwf* OUTmdwf* runERR_mdwf* runERRmdwf*; do
    if [ -f ${killFile} ]; then rm ${killFile}; fi
done


# Run one executable (name in this directory, or path) on one GPU; the remaining arguments are passed on
# (parameter file).
function runMDWFRoutine {
    exe="$1"
    shift
    [[ ${exe} == */* ]] || exe=./${exe}
    routine=$(basename ${exe})
    outFile='OUT'"${routine}"
    errFile='runERR'"${routine}"
    echo '  '${exe}' '"$@"
    if [ ! -x ${exe} ]; then
        echo "not built (make -j mdwftests mdwf)" >> ${errFile}
        echo -e "  ${cred}not built${endc}"
        return
    fi
    $run_command 1 ${exe} "$@" >> ${outFile} 2>> ${errFile}
    status=$?
    if [ ${status} -ne 0 ]; then
        echo "exit status ${status}" >> ${errFile}
        echo -e "  ${cred}FAILED, exit status ${status}${endc}"
    fi
    echo " " >> ${outFile}
    # This is just to remove these files if they are empty.
    if [ ! -s ${errFile} ]; then rm ${errFile}; fi
    if [ ! -s ${outFile} ]; then rm ${outFile}; fi
}


# Tests: physics parameters are built in, the lattice comes from parameter/tests/mdwfFifthDimTest.param
# (mdwfDSDRTest: parameter/tests/mdwfDSDRTest.param). mdwfRemezTest is a host program without the "_" variant.
testRoutines=(
    _mdwfWilsonKernelNormalizationTest
    _mdwfCloverAbelianFluxTest
    _mdwfMobiusCloverMappingTest
    _mdwfMobiusAdjointMappingTest
    _mdwfMobiusEvenOddTest
    _mdwfKernelSmearingTest
    _mdwfEffectiveOperatorTest
    _mdwfMobiusNormalSolveTest
    _mdwfSimultaneousMultiShiftCGTest
    mdwfRemezTest
    _mdwfMobiusRationalActionTest
    _mdwfMobiusFiniteDifferenceTest
    _mdwfMobiusForceContractionTest
    _mdwfMobiusAllLinkForceStorageTest
    _mdwfDeviceForceStorageTest
    _mdwfPauliVillarsActionTest
    _mdwfOneFlavorRhmcActionTest
    _mdwfEvenOddActionTest
    _mdwfMobiusHmcConventionTest
    _mdwfSymanzikGaugeForceTest
    _mdwfHmcTrajectoryTest
    _mdwfPauliVillarsHmcTrajectoryTest
    _mdwfPauliVillarsMultiscaleTest
    _mdwfTwoPlusOneHmcTrajectoryTest
    _mdwfEvenOddTwoPlusOneTrajectoryTest
    _mdwfAntiperiodicHmcTest
    _mdwfHasenbuschTest
    _mdwfDSDRTest
)

# Programs, each with its example parameter file parameter/tests/<program>.param; the HMC is an application
# (<build>/applications, parameter/applications) and runs first.
programRoutines=(
    _mdwfResidualMass
    _mdwfMeasure
    _mdwfSpectralBounds
    _mdwfKernelSpectrum
    _mdwfDiracModes
    _mdwfOperatorBench
    _mdwfProfileBench
)

echo
date

if [ "${part}" == "all" ] || [ "${part}" == "tests" ]; then
    echo
    echo -e "${cyan}MDWF tests${endc}"
    for routine in "${testRoutines[@]}"; do
        runMDWFRoutine "${routine}"
    done
fi

if [ "${part}" == "all" ] || [ "${part}" == "programs" ]; then
    echo
    echo -e "${cyan}MDWF programs with the example parameter files${endc}"
    runMDWFRoutine ../applications/_mdwfTwoPlusOneHmcRun ../parameter/applications/mdwfTwoPlusOneHmcRun.param
    for routine in "${programRoutines[@]}"; do
        runMDWFRoutine "${routine}" "../parameter/tests/${routine:1}.param"
    done
fi

echo
date
nErr=$(ls runERR_mdwf* runERRmdwf* 2>/dev/null | wc -l)
if [ ${nErr} -eq 0 ]; then
    echo -e "${cyan}No runERR files: all MDWF runs passed.${endc}"
else
    echo -e "${cred}${nErr} runERR files:${endc}"
    ls runERR_mdwf* runERRmdwf* 2>/dev/null
fi
