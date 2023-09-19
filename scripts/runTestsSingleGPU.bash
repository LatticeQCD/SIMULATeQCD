#!/bin/bash

## prefix could be for example optirun on some systems. Leave empty if not necessary.
if [ -z "$1" ]; then
    prefix=""
else
    prefix=$1
    echo "Run tests with prefix ${prefix}"
fi

## Default Nodes = "1 1 1 1"
if [ -z "$2" ]; then
    nodes="1 1 1 1"
else
    nodes="$2"
fi


echo Run with node settings: "${nodes}"

echo ==========================================================================================================
echo Running Test: ./_bulkIndexerTest ../parameter/tests/bulkIndexerTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_bulkIndexerTest ../parameter/tests/bulkIndexerTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_colorElectricCorrTest ../parameter/tests/colorElectricCorrTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_colorElectricCorrTest ../parameter/tests/colorElectricCorrTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_compressionTest ../parameter/tests/compressionTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_compressionTest ../parameter/tests/compressionTest.param Nodes="${nodes}"


cp ../parameter/ua_s8t4.norm ../parameter/us_s8t4.norm .
echo ""
echo ==========================================================================================================
echo Running Test: ./_correlatorTest ../parameter/tests/correlatorTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_correlatorTest ../parameter/tests/correlatorTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_dotProductTest ../parameter/tests/mixedPrecInverterTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_dotProductTest ../parameter/tests/mixedPrecInverterTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_dslashMultiTest ../parameter/tests/dslashMultiTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_dslashMultiTest ../parameter/tests/dslashMultiTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_dslashTest ../parameter/tests/dslashTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_dslashTest ../parameter/tests/dslashTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_generalFunctorTest ../parameter/tests/generalFunctorTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_generalFunctorTest ../parameter/tests/generalFunctorTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_gfixTestSingle ../parameter/tests/gfixplcTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_gfixTestSingle ../parameter/tests/gfixplcTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_gfixTestMulti ../parameter/tests/gfixTestMulti.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_gfixTestMulti ../parameter/tests/gfixTestMulti.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_gfixplcTest ../parameter/tests/gfixplcTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_gfixplcTest ../parameter/tests/gfixplcTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_gradientFlowTest ../parameter/tests/gradientFlowTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_gradientFlowTest ../parameter/tests/gradientFlowTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_haloTest ../parameter/tests/haloTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_haloTest ../parameter/tests/haloTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hbor_single_test ../parameter/tests/dslashTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hbor_single_test ../parameter/tests/dslashTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hbor_multiple_test ../parameter/tests/hbor_multiple_test.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hbor_multiple_test ../parameter/tests/hbor_multiple_test.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hisqForce ../parameter/tests/hisqForce.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hisqForce ../parameter/tests/hisqForce.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hisqSmearingImagmuTest ../parameter/tests/hisqSmearingTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hisqSmearingImagmuTest ../parameter/tests/hisqSmearingTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hisqSmearingMultiCreate ../parameter/tests/run.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hisqSmearingMultiCreate ../parameter/tests/run.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hisqSmearingTest ../parameter/tests/hisqSmearingTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hisqSmearingTest ../parameter/tests/hisqSmearingTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_inverterTest ../parameter/tests/inverterTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_inverterTest ../parameter/tests/inverterTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_linkPathTest ../parameter/tests/linkPathTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_linkPathTest ../parameter/tests/linkPathTest.param Nodes="${nodes}"

echo ""
echo Running Test: ./_mixedPrecInverterTest ../parameter/tests/mixedPrecInverterTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_mixedPrecInverterTest ../parameter/tests/mixedPrecInverterTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_pureGaugeHmcTest ../parameter/tests/pureGaugeHmcTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_pureGaugeHmcTest ../parameter/tests/pureGaugeHmcTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_rhmcTest_4pf ../parameter/tests/rhmcTest_4pf.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_rhmcTest_4pf ../parameter/tests/rhmcTest_4pf.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_rhmcTest ../parameter/tests/rhmcTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_rhmcTest ../parameter/tests/rhmcTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_saveTest ../parameter/tests/saveTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_saveTest ../parameter/tests/saveTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_spinorHaloTest ../parameter/tests/spinorHaloTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_spinorHaloTest ../parameter/tests/spinorHaloTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_stackedSpinorTest ../parameter/tests/stackedSpinorTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_stackedSpinorTest ../parameter/tests/stackedSpinorTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_dslashImagmuTest ../parameter/tests/dslashTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_dslashImagmuTest ../parameter/tests/dslashTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_memManTest
echo ==========================================================================================================
${prefix} ./_memManTest

echo ""
echo ==========================================================================================================
echo Running Test: ./_halfPrecMathTest
echo ==========================================================================================================
${prefix} ./_halfPrecMathTest

echo ""
echo ==========================================================================================================
echo Running Test: ./_rndSingleTest
echo ==========================================================================================================
${prefix} ./_rndSingleTest

echo ""
echo ==========================================================================================================
echo Running Test: ./_utimesUdaggerTest
echo ==========================================================================================================
${prefix} ./_utimesUdaggerTest

echo ""
echo ==========================================================================================================
echo Running Test: ./_simpleFunctorTest
echo ==========================================================================================================
${prefix} ./_simpleFunctorTest

if [[ $nodes != "1 1 1 1" ]]
then
    echo ""
    echo ==========================================================================================================
    echo Running Test: ./_cudaIpcTest ../parameter/tests/cudaIpcTest.param Nodes="${nodes}"
    echo ==========================================================================================================
    ${prefix} ./_cudaIpcTest ../parameter/tests/cudaIpcTest.param Nodes="${nodes}"
fi
