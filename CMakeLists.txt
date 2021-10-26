#
# CMakeLists.txt
#
# CMake file for programs using the multiGPU code. Compilation target definitions are organized by type, and then
# alphabetically. Please try to follow these guidelines when adding your target.
#
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(ParallelGPUCode LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find and include+link MPI and CUDA
find_package(MPI 3.1 REQUIRED)
find_package(CUDA 11.0 REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
link_libraries(${MPI_LIBRARIES} ${CUDA_LIBRARIES})

# Read architecture, CUDA-aware MPI, CUDA P2P settings
set(ARCHITECTURE "" CACHE STRING "Set the architecture. For example 70 will compile for Volta support")
if(NOT ARCHITECTURE)
	message(FATAL_ERROR "No GPU architecture set!")
endif()
message(STATUS "CUDA Architecture: ${ARCHITECTURE}")
set(CMAKE_CUDA_ARCHITECTURES "${ARCHITECTURE}")

set(USE_CUDA_AWARE_MPI OFF CACHE BOOL "Set to ON to build cuda-aware MPI code (default = OFF)")
if (USE_CUDA_AWARE_MPI)
	add_definitions(-DUSE_CUDA_AWARE_MPI)
endif()
set(USE_CUDA_P2P ON CACHE BOOL "Set to ON to build with CUDA Direct P2P (default = ON)")
if (USE_CUDA_P2P)
	add_definitions(-DUSE_CUDA_P2P)
endif()

option(USE_MPFR "Set to ON to include MPFR - needed for rat_approx (default - OFF)" OFF)

# Additional compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-comment -fPIC")
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -O3 -Wno-deprecated-gpu-targets --std=c++17 -arch=sm_${ARCHITECTURE} -Xcudafe --display_error_number -prec-div=true -prec-sqrt=true")



# --------------------------------------------------------------------------------------------------VARIABLE DEFINITIONS


set(SOURCE_FILES_BASE
		src/base/gutils.cpp
		src/base/memoryManagement.cpp
		src/base/indexer/initGPUIndexer.cu
		src/base/indexer/initCPUIndexer.cpp
		src/base/communication/communicationBase_mpi.cpp
        src/base/IO/parameterManagement.cpp
        src/base/IO/fileWriter.cpp
        src/base/math/grnd.cu
		src/gauge/gaugefield.cu
		src/gauge/gaugefield.cpp
		src/gauge/GaugeAction.cu
        src/base/LatticeContainer.cu)
set(SOURCE_FILES_COLORELECTRICCORRTEST
		src/modules/observables/ColorElectricCorr.cu
		src/modules/observables/PolyakovLoop.cu)
set(SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST
        src/modules/observables/ColorElectricCorr.cu
        src/modules/observables/ColorMagneticCorr.cu
        src/modules/observables/PolyakovLoop.cu)
set(SOURCE_FILES_GENERATEQUENCHED
        src/modules/gauge_updates/PureGaugeUpdates.cu
		src/modules/observables/PolyakovLoop.cu)
set(SOURCE_FILES_GFIX
		src/modules/gaugeFixing/gfix.cu
		src/modules/observables/PolyakovLoop.cu
        src/modules/gaugeFixing/PolyakovLoopCorrelator.cu
		src/modules/observables/WilsonLineCorrelator.cu)
set(SOURCE_FILES_GFLOW
		src/modules/observables/Topology.cu
		src/modules/observables/Blocking.cu
		src/modules/observables/ColorElectricCorr.cu
		src/modules/observables/ColorMagneticCorr.cu)
set(SOURCE_FILES_GETMAGICTRACEANOMALY
		src/modules/gauge_updates/SubLatMeas.cu)
set(SOURCE_FILES_MEASUREHADRONS
		src/modules/measureHadrons/measureHadrons.cu
		src/modules/HISQ/hisqSmearing.cu
		src/modules/dslash/dslash.cu
		src/modules/inverter/inverter.cu
		src/spinor/spinorfield.cu)
set(SOURCE_FILES_SUBLATTICEUPDATES
		src/modules/gauge_updates/luscherweisz.cu
		src/modules/gauge_updates/SubLatMeas.cpp
		src/modules/gauge_updates/SubLatMeas.cu)
set(SOURCE_FILES_HMC
		src/modules/rhmc/pure_gauge_hmc.cu
		src/modules/rhmc/integrator.cu
        src/modules/dslash/dslash.cu
		src/modules/inverter/inverter.cu
        src/modules/HISQ/hisqSmearing.cu
        src/modules/HISQ/hisqForce.cu
		src/spinor/spinorfield.cu)
set(SOURCE_FILES_RHMC
		src/modules/rhmc/rhmc.cu
		src/modules/rhmc/integrator.cu
        src/modules/dslash/dslash.cu
		src/modules/inverter/inverter.cu
        src/modules/HISQ/hisqSmearing.cu
        src/modules/HISQ/hisqForce.cu
		src/modules/observables/PolyakovLoop.cu
		src/spinor/spinorfield.cu)
set(ALL_NSTACK
		NSTACKS_1=1
		NSTACKS_2=1
		NSTACKS_4=1
		NSTACKS_8=1
		NSTACKS_12=1
		NSTACKS_14=1)
set(REALLY_ALL_NSTACK
		${ALL_NSTACK}
		NSTACKS_3=1
		NSTACKS_5=1
		NSTACKS_6=1
		NSTACKS_7=1
		NSTACKS_9=1
		NSTACKS_10=1
		NSTACKS_11=1)
set(ALL_LAYOUTS
		LAYOUT_ALL=1
		LAYOUT_ODD=1
		LAYOUT_EVEN=1)
set(ALL_HALODEPTHS
        HALODEPTH_0=1
        HALODEPTH_1=1
        HALODEPTH_2=1
        HALODEPTH_3=1
        HALODEPTH_4=1)


#-------------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: CODEBASE


add_library(CodeBase OBJECT ${SOURCE_FILES_BASE})
target_compile_definitions(CodeBase PRIVATE
		ARCHITECTURE=${ARCHITECTURE}
		SINGLEPREC=1 DOUBLEPREC=1
		COMP_R18=1 COMP_U3R14=1 COMP_R14=1 COMP_R12=1 COMP_STAGGR12=1
	    ${ALL_HALODEPTHS}
	    HALODEPTHSPIN_0=1 HALODEPTHSPIN_4=1
		${REALLY_ALL_NSTACK} ${ALL_LAYOUTS}
		)
set_target_properties(CodeBase
		PROPERTIES
		COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON)


#---------------------------------------------------------------------------------------TARGET FUNCTIONS WITH PROPERTIES


FUNCTION(add_ParallelGPU_executable TARGET)
	add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
	add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
	target_link_libraries("_${TARGET}" CodeBase)
	set_target_properties(${TARGET} _${TARGET}
			PROPERTIES
			COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON)
ENDFUNCTION()
FUNCTION(set_ParallelGPU_property TARGET)
	set_target_properties(${TARGET} _${TARGET} ${ARGN})
ENDFUNCTION()
FUNCTION(ParallelGPU_target_compile_definitions TARGET)
	target_compile_definitions(${TARGET} ${ARGN} ARCHITECTURE=${ARCHITECTURE})
	target_compile_definitions(_${TARGET} ${ARGN} ARCHITECTURE=${ARCHITECTURE})
ENDFUNCTION()

FUNCTION(add_to_compound_ParallelGPU_target TARGET1 TARGET2)
	add_dependencies(${TARGET1} _${TARGET2})
ENDFUNCTION()


#-------------------------------------------------------------------------------------------------------COMPOUND TARGETS


# These targets re-use the CodeBase library when compiling (this speeds up the compilation if you want to compile many
# things at once!) Executables will have a leading "_" in their name if compiled as part of these targets
add_custom_target(applications)
add_custom_target(examples)
add_custom_target(profilers)
add_custom_target(tests)
add_custom_target(tools)
add_custom_target(everything)
add_dependencies(everything
		applications
		examples
		profilers
		tests
		tools
		)


#----------------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: TESTS


add_ParallelGPU_executable(BulkIndexerTest src/testing/main_BulkIndexerTest.cu)
set_ParallelGPU_property(BulkIndexerTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(BulkIndexerTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests BulkIndexerTest)

add_ParallelGPU_executable(ColorElectricCorrTest src/testing/main_ColorElectricCorrTest.cu ${SOURCE_FILES_COLORELECTRICCORRTEST})
set_ParallelGPU_property(ColorElectricCorrTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(ColorElectricCorrTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests ColorElectricCorrTest)

add_ParallelGPU_executable(ColorElectricMagneticCloverTest src/testing/main_ColorElectricMagneticCloverTest.cu ${SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST})
set_ParallelGPU_property(ColorElectricMagneticCloverTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(ColorElectricMagneticCloverTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests ColorElectricMagneticCloverTest)

add_ParallelGPU_executable(FieldStrengthTensorTest src/testing/main_FieldStrengthTensorTest.cu)
set_ParallelGPU_property(FieldStrengthTensorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(FieldStrengthTensorTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests FieldStrengthTensorTest)

add_ParallelGPU_executable(half_prec_math_test src/testing/main_half_prec_math_test.cu src/spinor/spinorfield.cu )
set_ParallelGPU_property(half_prec_math_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(half_prec_math_test PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 SINGLEPREC=1 HALFPREC=1 COMP_R18=1 COMP_U3R14=1 ${ALL_NSTACK} ${ALL_LAYOUTS})

add_ParallelGPU_executable(CompressionTest src/testing/main_CompressionTest.cu)
set_ParallelGPU_property(CompressionTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(CompressionTest PRIVATE HALODEPTH_2=1  DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 COMP_R12=1 COMP_STAGGR12=1)
add_to_compound_ParallelGPU_target(tests CompressionTest)

add_ParallelGPU_executable(correlatorTest src/testing/main_correlatorTest.cu ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(correlatorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(correlatorTest PRIVATE HALODEPTH_0=1  DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests correlatorTest)

add_ParallelGPU_executable(cudaIpcTest src/testing/main_cudaIpcTest.cu)
set_ParallelGPU_property(cudaIpcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(cudaIpcTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests cudaIpcTest)

add_ParallelGPU_executable(DslashTest src/testing/main_DslashTest.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(DslashTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(DslashTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests DslashTest)

add_ParallelGPU_executable(DslashMultiTest src/testing/main_DslashMultiTest.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(DslashMultiTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(DslashMultiTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests DslashMultiTest)

add_ParallelGPU_executable(GeneralFunctorTest src/testing/main_GeneralFunctorTest.cu src/modules/dslash/dslash.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(GeneralFunctorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(GeneralFunctorTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests GeneralFunctorTest)

add_ParallelGPU_executable(gfixplcTest src/testing/main_gfixplcTest.cu ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(gfixplcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(gfixplcTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests gfixplcTest)

add_ParallelGPU_executable(gfixTestMulti src/testing/main_gfixTestMulti.cu ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(gfixTestMulti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(gfixTestMulti PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests gfixTestMulti)

add_ParallelGPU_executable(gfixTestSingle src/testing/main_gfixTestSingle.cu ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(gfixTestSingle PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(gfixTestSingle PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests gfixTestSingle)

add_ParallelGPU_executable(gradientFlowTest src/testing/main_gradientFlowTest.cu src/modules/observables/Topology.cu)
set_ParallelGPU_property(gradientFlowTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(gradientFlowTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_ParallelGPU_target(tests gradientFlowTest)

add_ParallelGPU_executable(HaloTest src/testing/main_HaloTest.cu)
set_ParallelGPU_property(HaloTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(HaloTest PRIVATE HALODEPTH_0=1 HALODEPTH_1=1 HALODEPTH_2=1  HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests HaloTest)

add_ParallelGPU_executable(HBOR_multiple_test src/testing/main_HBOR_multiple_test.cu src/modules/gauge_updates/PureGaugeUpdates.cu)
set_ParallelGPU_property(HBOR_multiple_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(HBOR_multiple_test PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests HBOR_multiple_test)

add_ParallelGPU_executable(HBOR_single_test src/testing/main_HBOR_single_test.cu src/modules/gauge_updates/PureGaugeUpdates.cu)
set_ParallelGPU_property(HBOR_single_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(HBOR_single_test PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests HBOR_single_test)

add_ParallelGPU_executable(hisqForce src/testing/main_hisqForce.cu src/modules/HISQ/hisqSmearing.cu src/modules/dslash/dslash.cu src/modules/inverter/inverter.cu src/modules/HISQ/hisqForce.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(hisqForce PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(hisqForce PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_0=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests hisqForce)

add_ParallelGPU_executable(hisqSmearingMulti src/testing/main_HisqSmearingTestMulti.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(hisqSmearingMulti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(hisqSmearingMulti PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests hisqSmearingMulti)

add_ParallelGPU_executable(hisqSmearingMultiCreate src/testing/main_HisqSmearing_Create_Multi.cu src/modules/HISQ/hisqSmearing.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(hisqSmearingMultiCreate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(hisqSmearingMultiCreate PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests hisqSmearingMultiCreate)

add_ParallelGPU_executable(hisqSmearingTest src/testing/main_HisqSmearingTest.cu src/modules/HISQ/hisqSmearing.cu)
set_ParallelGPU_property(hisqSmearingTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(hisqSmearingTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests hisqSmearingTest)

add_ParallelGPU_executable(InverterTest src/testing/main_InverterTest.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/modules/inverter/inverter.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(InverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(InverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests InverterTest)

add_ParallelGPU_executable(MixedPrecInverterTest src/testing/main_MixedPrecInverterTest.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/modules/inverter/inverter.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(MixedPrecInverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(MixedPrecInverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 HALFPREC=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})

add_ParallelGPU_executable(MixedPrecMultiShiftInverterTest src/testing/main_MixedPrecShiftedInverterTest.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/modules/inverter/inverter.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(MixedPrecMultiShiftInverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(MixedPrecMultiShiftInverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 HALFPREC=1 SINGLEPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})

add_ParallelGPU_executable(dotProductTest src/testing/main_dotProduct.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(dotProductTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(dotProductTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests dotProductTest)

add_ParallelGPU_executable(LinkPathTest src/testing/main_LinkPathTest.cu)
set_ParallelGPU_property(LinkPathTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(LinkPathTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests LinkPathTest)

add_ParallelGPU_executable(measureHadronsTest src/testing/main_measureHadronsTest.cu ${SOURCE_FILES_MEASUREHADRONS})
set_ParallelGPU_property(measureHadronsTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(measureHadronsTest PRIVATE COMP_R18=1 HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R14=1 NSTACKS_1=1 NSTACKS_3=1 ${ALL_LAYOUTS})

add_ParallelGPU_executable(memManTest src/testing/main_memManTest.cu)
set_ParallelGPU_property(memManTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(memManTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests memManTest)

add_ParallelGPU_executable(PureGaugeHmcTest src/testing/main_pure_gauge_hmc_test.cu src/modules/observables/PolyakovLoop.cu ${SOURCE_FILES_HMC})
set_ParallelGPU_property(PureGaugeHmcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(PureGaugeHmcTest PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests PureGaugeHmcTest)

add_ParallelGPU_executable(rhmcTest src/testing/main_rhmc_test.cu ${SOURCE_FILES_RHMC})
set_ParallelGPU_property(rhmcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(rhmcTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests rhmcTest)

add_ParallelGPU_executable(RndMultipleTest src/testing/main_rnd_multiple_test.cu)
set_ParallelGPU_property(RndMultipleTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(RndMultipleTest PRIVATE HALODEPTH_1=1 HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests RndMultipleTest)

add_ParallelGPU_executable(RndSingleTest src/testing/main_rnd_single_test.cu)
set_ParallelGPU_property(RndSingleTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(RndSingleTest PRIVATE HALODEPTH_1=1 HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests RndSingleTest)

add_ParallelGPU_executable(SaveTest src/testing/main_SaveTest.cu)
set_ParallelGPU_property(SaveTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(SaveTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests SaveTest)

add_ParallelGPU_executable(SpinorHaloTest src/testing/main_SpinorHaloTest.cu)
set_ParallelGPU_property(SpinorHaloTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(SpinorHaloTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tests SpinorHaloTest)

add_ParallelGPU_executable(StackedSpinorTest src/testing/main_StackedSpinorTest.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(StackedSpinorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(StackedSpinorTest PRIVATE HALODEPTH_4=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests StackedSpinorTest)

add_ParallelGPU_executable(cudaAwareMPITest src/testing/main_cudaAwareMPITest.cu)
set_ParallelGPU_property(cudaAwareMPITest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(cudaAwareMPITest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)

# This one is special because it does not need the CodeBase
add_executable(SimpleFunctorTest src/testing/main_SimpleFunctorTest.cu src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
add_executable(_SimpleFunctorTest src/testing/main_SimpleFunctorTest.cu src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
set_target_properties(SimpleFunctorTest _SimpleFunctorTest PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON RUNTIME_OUTPUT_DIRECTORY "testing")
target_compile_definitions(SimpleFunctorTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1)
target_compile_definitions(_SimpleFunctorTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1)
add_to_compound_ParallelGPU_target(tests SimpleFunctorTest)

add_ParallelGPU_executable(UtimesUdaggerTest src/testing/main_UtimesUdaggerTest.cu)
set_ParallelGPU_property(UtimesUdaggerTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
ParallelGPU_target_compile_definitions(UtimesUdaggerTest PRIVATE HALODEPTH_0=1 HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tests UtimesUdaggerTest)


#-------------------------------------------------------------------------------------COMPILATION DEFINITIONS: PROFILING


add_ParallelGPU_executable(ForceProfiling src/profiling/main_ForceProfiling.cu src/modules/HISQ/hisqSmearing.cu src/modules/dslash/dslash.cu src/modules/inverter/inverter.cu src/modules/HISQ/hisqForce.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(ForceProfiling PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
ParallelGPU_target_compile_definitions(ForceProfiling PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_2=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(profilers ForceProfiling)

add_ParallelGPU_executable(HBOR_benchmark src/profiling/main_HBOR_benchmark.cu src/modules/gauge_updates/PureGaugeUpdates.cu)
set_ParallelGPU_property(HBOR_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
ParallelGPU_target_compile_definitions(HBOR_benchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(profilers HBOR_benchmark)

add_ParallelGPU_executable(InverterProf src/profiling/main_InverterProfiling.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/modules/inverter/inverter.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(InverterProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
ParallelGPU_target_compile_definitions(InverterProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(profilers InverterProf)

add_ParallelGPU_executable(MultiRHSProf src/profiling/main_MrhsDSlashProf.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(MultiRHSProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
ParallelGPU_target_compile_definitions(MultiRHSProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 HALFPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(profilers MultiRHSProf)

add_ParallelGPU_executable(MultiRHSProf_half src/testing/main_half_prec_dslash_test.cu src/modules/dslash/dslash.cu src/modules/HISQ/hisqSmearing.cu src/spinor/spinorfield.cu)
set_ParallelGPU_property(MultiRHSProf_half PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
ParallelGPU_target_compile_definitions(MultiRHSProf_half PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 HALFPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(profilers MultiRHSProf_half)


#--------------------------------------------------------------------------------------COMPILATION DEFINITIONS: EXAMPLES


add_ParallelGPU_executable(gradientFlow_example src/examples/main_gradientFlow_example.cu)
set_ParallelGPU_property(gradientFlow_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
ParallelGPU_target_compile_definitions(gradientFlow_example PRIVATE HALODEPTH_3=1 DOUBLEPREC=1 COMP_R18=1 ADAPTIVE_STEPSIZE=1 ZEUTHEN_FLOW=1)
add_to_compound_ParallelGPU_target(examples gradientFlow_example)

add_ParallelGPU_executable(plaquette src/examples/main_plaquette.cu)
set_ParallelGPU_property(plaquette PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
ParallelGPU_target_compile_definitions(plaquette PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(examples plaquette)

add_ParallelGPU_executable(PolyakovLoop src/examples/main_polyakovloop.cu)
set_ParallelGPU_property(PolyakovLoop PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
ParallelGPU_target_compile_definitions(PolyakovLoop PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(examples PolyakovLoop)

add_ParallelGPU_executable(WilsonLinesFields src/examples/main_wilson_lines_fields.cu src/modules/gaugeFixing/gfix.cu)
set_ParallelGPU_property(WilsonLinesFields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
ParallelGPU_target_compile_definitions(WilsonLinesFields PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_ParallelGPU_target(examples WilsonLinesFields)

add_ParallelGPU_executable(WilsonLinesFieldsShared src/examples/main_wilson_lines_fields_stacked_shared.cu src/modules/gaugeFixing/gfix.cu)
set_ParallelGPU_property(WilsonLinesFieldsShared PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
ParallelGPU_target_compile_definitions(WilsonLinesFieldsShared PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_ParallelGPU_target(examples WilsonLinesFieldsShared)


#---------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: APPLICATIONS


add_ParallelGPU_executable(fluxtube src/applications/main_flt.cu ${SOURCE_FILES_BASE} ${SOURCE_FILES_GFIX})
set_ParallelGPU_property( fluxtube PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(fluxtube PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
#add_to_compound_ParallelGPU_target(applications fluxtube)

add_ParallelGPU_executable(GenerateQuenched src/applications/main_GenerateQuenched.cu ${SOURCE_FILES_GENERATEQUENCHED})
set_ParallelGPU_property(GenerateQuenched PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(GenerateQuenched PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(applications GenerateQuenched)

add_ParallelGPU_executable(gfix src/applications/main_gaugeFixing.cu ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(gfix PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(gfix PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(applications gfix)

add_ParallelGPU_executable(gradientFlow src/applications/main_gradientFlow.cu ${SOURCE_FILES_GFLOW} ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(gradientFlow PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications") 
ParallelGPU_target_compile_definitions(gradientFlow PRIVATE COMP_R18=1 HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 DOUBLEPREC=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_ParallelGPU_target(applications gradientFlow)

add_ParallelGPU_executable(gradientFlow_zeuthen src/applications/main_gradientFlow.cu ${SOURCE_FILES_GFLOW} ${SOURCE_FILES_GFIX})
set_ParallelGPU_property(gradientFlow_zeuthen PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(gradientFlow_zeuthen PRIVATE COMP_R18=1 DOUBLEPREC=1 HALODEPTH_3=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1 ZEUTHEN_FLOW=1)
add_to_compound_ParallelGPU_target(applications gradientFlow_zeuthen)

add_ParallelGPU_executable(polSuscRenorm src/applications/main_polSuscRenorm.cu src/modules/observables/PolyakovLoop.cu)
set_ParallelGPU_property(polSuscRenorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(polSuscRenorm PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(applications polSuscRenorm)

add_ParallelGPU_executable(RHMC src/applications/main_rhmc.cu ${SOURCE_FILES_RHMC})
set_ParallelGPU_property(RHMC PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(RHMC PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(applications RHMC)

add_ParallelGPU_executable(sublatticeUpdates src/applications/main_sublatticeUpdates.cu ${SOURCE_FILES_SUBLATTICEUPDATES})
set_ParallelGPU_property(sublatticeUpdates PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(sublatticeUpdates PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(applications sublatticeUpdates)

add_ParallelGPU_executable(measureHadrons src/applications/main_measureHadrons.cu ${SOURCE_FILES_MEASUREHADRONS})
set_ParallelGPU_property(measureHadrons PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(measureHadrons PRIVATE COMP_R18=1 HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R14=1 NSTACKS_1=1 NSTACKS_3=1 ${ALL_LAYOUTS})

add_ParallelGPU_executable(sampleTopology src/applications/main_sampleTopology.cu ${SOURCE_FILES_GFLOW} ${SOURCE_FILES_GFIX} ${SOURCE_FILES_GENERATEQUENCHED})
set_ParallelGPU_property(sampleTopology PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(sampleTopology PRIVATE COMP_R18=1 HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 DOUBLEPREC=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_ParallelGPU_target(applications sampleTopology)

add_ParallelGPU_executable(WilsonLinesCorrelatorMiltiGPUStacked src/applications/main_wilson_lines_correlator_stacked.cu src/modules/gaugeFixing/gfix.cu src/modules/observables/WilsonLineCorrelatorMultiGPU.cu)
set_ParallelGPU_property(WilsonLinesCorrelatorMiltiGPUStacked PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(WilsonLinesCorrelatorMiltiGPUStacked PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_ParallelGPU_target(applications WilsonLinesCorrelatorMiltiGPUStacked)

add_ParallelGPU_executable(MilcToNersc src/applications/main_milc_to_nersc.cu)
set_ParallelGPU_property(MilcToNersc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
ParallelGPU_target_compile_definitions(MilcToNersc PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_ParallelGPU_target(applications MilcToNersc)


#-----------------------------------------------------------------------------------------COMPILATION DEFINITIONS: TOOLS


add_ParallelGPU_executable(condensateTest src/tools/main_condensate_test.cu ${SOURCE_FILES_RHMC})
set_ParallelGPU_property(condensateTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "tools")
ParallelGPU_target_compile_definitions(condensateTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_ParallelGPU_target(tools condensateTest)

add_ParallelGPU_executable(correlatorNorm src/tools/main_createCorrelatorNorm.cu)
set_ParallelGPU_property(correlatorNorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "tools")
ParallelGPU_target_compile_definitions(correlatorNorm PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_ParallelGPU_target(tools correlatorNorm)


#------------------------------------------------------------------------ADDITIONAL FIND FUNCTIONS AND TARGETS FOR TOOLS


if(USE_MPFR)
find_path(GMP_INCLUDE_DIR gmp.h gmpxx.h 
    PATHS ${GMP_PREFIX}/include /usr/include /usr/local/include )

find_library(GMP_LIBRARY NAMES gmp libgmp 
    PATHS ${GMP_PREFIX}/lib /usr/lib /usr/local/lib)


if(GMP_INCLUDE_DIR AND GMP_LIBRARY)
    get_filename_component(GMP_LIBRARY_DIR ${GMP_LIBRARY} PATH)
    set(GMP_FOUND TRUE)
endif()

if(GMP_FOUND)
      MESSAGE(STATUS "Found GMP: ${GMP_LIBRARY}")
elseif(GMP_FOUND)
      message(WARNING "Could not find GMP")
endif()

# Find MPFR Library

find_path(MPFR_INCLUDE_DIR mpf2mpfr.h mpfr.h 
    PATHS ${MPFR_PREFIX}/include /usr/include /usr/local/include )

find_library(MPFR_LIBRARY NAMES mpfr libmpfr 
    PATHS ${MPFR_PREFIX}/lib /usr/lib /usr/local/lib)


if(MPFR_INCLUDE_DIR AND MPFR_LIBRARY)
    get_filename_component(MPFR_LIBRARY_DIR ${MPFR_LIBRARY} PATH)
    set(MPFR_FOUND TRUE)
endif()

if(MPFR_FOUND)
      MESSAGE(STATUS "Found MPFR: ${MPFR_LIBRARY}")
elseif(MPFR_FOUND)
      message(WARNING "Could not find MPFR")
endif()

add_executable(rat_approx src/tools/rational_approx/poly4.C src/tools/rational_approx/alg_remez.C)

set_target_properties(rat_approx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "tools")
target_include_directories(rat_approx PUBLIC ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR})
target_link_libraries(rat_approx PUBLIC ${MPFR_LIBRARY} ${GMP_LIBRARY})
endif()


#-------------------------------------------------------------------------------------------------------------COPY FILES


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_conf      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameter      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
