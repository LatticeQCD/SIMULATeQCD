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
echo Running Test: ./_BulkIndexerTest ../parameter/tests/BulkIndexerTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_BulkIndexerTest ../parameter/tests/BulkIndexerTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_ColorElectricCorrTest ../parameter/tests/ColorElectricCorrTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_ColorElectricCorrTest ../parameter/tests/ColorElectricCorrTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_CompressionTest ../parameter/tests/CompressionTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_CompressionTest ../parameter/tests/CompressionTest.param Nodes="${nodes}"


cp ../parameter/UA_s8t4.norm ../parameter/US_s8t4.norm .
echo ""
echo ==========================================================================================================
echo Running Test: ./_correlatorTest ../parameter/tests/correlatorTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_correlatorTest ../parameter/tests/correlatorTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_dotProductTest ../parameter/tests/MixedPrecInverterTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_dotProductTest ../parameter/tests/MixedPrecInverterTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_DslashMultiTest ../parameter/tests/DslashMultiTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_DslashMultiTest ../parameter/tests/DslashMultiTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_DslashTest ../parameter/tests/DslashTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_DslashTest ../parameter/tests/DslashTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_GeneralFunctorTest ../parameter/tests/GeneralFunctorTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_GeneralFunctorTest ../parameter/tests/GeneralFunctorTest.param Nodes="${nodes}"

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
echo Running Test: ./_HaloTest ../parameter/tests/HaloTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_HaloTest ../parameter/tests/HaloTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_HBOR_single_test ../parameter/tests/DslashTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_HBOR_single_test ../parameter/tests/DslashTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_HBOR_multiple_test ../parameter/tests/HBOR_multiple_test.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_HBOR_multiple_test ../parameter/tests/HBOR_multiple_test.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_hisqForceImagmu ../parameter/tests/hisqForceImagMu.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_hisqForceImagmu ../parameter/tests/hisqForceImagMu.param Nodes="${nodes}"

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
echo Running Test: ./_InverterTest ../parameter/tests/InverterTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_InverterTest ../parameter/tests/InverterTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_LinkPathTest ../parameter/tests/LinkPathTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_LinkPathTest ../parameter/tests/LinkPathTest.param Nodes="${nodes}"

echo ""
echo Running Test: ./_MixedPrecInverterTest ../parameter/tests/MixedPrecInverterTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_MixedPrecInverterTest ../parameter/tests/MixedPrecInverterTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_PureGaugeHmcTest ../parameter/tests/PureGaugeHmcTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_PureGaugeHmcTest ../parameter/tests/PureGaugeHmcTest.param Nodes="${nodes}"

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
echo Running Test: ./_SaveTest ../parameter/tests/SaveTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_SaveTest ../parameter/tests/SaveTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_SpinorHaloTest ../parameter/tests/SpinorHaloTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_SpinorHaloTest ../parameter/tests/SpinorHaloTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_StackedSpinorTest ../parameter/tests/StackedSpinorTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_StackedSpinorTest ../parameter/tests/StackedSpinorTest.param Nodes="${nodes}"

echo ""
echo ==========================================================================================================
echo Running Test: ./_DslashImagmuTest ../parameter/tests/DslashTest.param Nodes="${nodes}"
echo ==========================================================================================================
${prefix} ./_DslashImagmuTest ../parameter/tests/DslashTest.param Nodes="${nodes}"

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
echo Running Test: ./_RndSingleTest
echo ==========================================================================================================
${prefix} ./_RndSingleTest

echo ""
echo ==========================================================================================================
echo Running Test: ./_UtimesUdaggerTest
echo ==========================================================================================================
${prefix} ./_UtimesUdaggerTest

echo ""
echo ==========================================================================================================
echo Running Test: ./_SimpleFunctorTest
echo ==========================================================================================================
${prefix} ./_SimpleFunctorTest

if [[ $nodes != "1 1 1 1" ]]
then
    echo ""
    echo ==========================================================================================================
    echo Running Test: ./_cudaIpcTest ../parameter/tests/cudaIpcTest.param Nodes="${nodes}"
    echo ==========================================================================================================
    ${prefix} ./_cudaIpcTest ../parameter/tests/cudaIpcTest.param Nodes="${nodes}"
fi
