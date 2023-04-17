#
# CMakeLists.txt
#
# CMake file for programs using the multiGPU code. Compilation target definitions are organized by type, and then
# alphabetically. Please try to follow these guidelines when adding your target.
#
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Get the latest abbreviated commit hash of the working branch
execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

message(STATUS "Using git version ${GIT_HASH}")

set(BACKEND "cuda" CACHE STRING "Specify what API should be used in the backend. Possible choices are cuda, hip_nvidia (experimental!) and hip_amd (experimental!)")

# Determine the backend for GPU acceleration
if (BACKEND STREQUAL "cuda")
    set(USE_CUDA ON)
    add_definitions(-DUSE_CUDA)
    message(STATUS "Using CUDA backend")
    
elseif (BACKEND STREQUAL "hip_nvidia")
    set(USE_HIP_NVIDIA ON)
    set(USE_HIP ON)
    add_definitions(-DUSE_HIP_NVIDIA)
    add_definitions(-DUSE_HIP)
    message(STATUS "Using HIP backend for NVIDIA GPUs (Experimental!)")
    
elseif (BACKEND STREQUAL "hip_amd")
    set(USE_HIP_AMD ON)
    set(USE_HIP ON)
    add_definitions(-DUSE_HIP_AMD)
    add_definitions(-DUSE_HIP)
    message(STATUS "Using HIP backend for AMD GPUs. (Experimental!)")
    
else()
    # If the backend is not recognized, print an error message
    message(STATUS "Backend ${BACKEND} is not known!")
endif()



if (USE_CUDA)
    project(SIMULATeQCD LANGUAGES CXX CUDA)
elseif (USE_HIP_AMD)
    # This will hopefully work in the future
    # project(SIMULATeQCD LANGUAGES CXX HIP)
    project(SIMULATeQCD LANGUAGES CXX)
elseif (USE_HIP_NVIDIA)
    # This will hopefully work in the future
    # project(SIMULATeQCD LANGUAGES CXX HIP)
    project(SIMULATeQCD LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(MPI 3.1 REQUIRED)


set(SIMULATEQCD_INCLUDE_DIRS "${MPI_INCLUDE_PATH}")
set(SIMULATEQCD_LIBRARIES "${MPI_LIBRARIES}")

if (USE_CUDA OR USE_HIP_NVIDIA)
    find_package(CUDA 11.0 REQUIRED)
    set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
    set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
    set(SIMULATEQCD_INCLUDE_DIRS "${SIMULATEQCD_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS}")
    set(SIMULATEQCD_LIBRARIES "${SIMULATEQCD_LIBRARIES};${CUDA_LIBRARIES}")
endif()

if (USE_HIP)
    set(HIP_PATH "" CACHE STRING "Specify path to the HIP installation.")

    if(NOT HIP_PATH)
        set(HIP_PATH $ENV{HIP_PATH})
    endif()

    if(NOT HIP_PATH)
        message(FATAL_ERROR "No HIP_PATH set!")
    endif()

    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)
    find_package(hipcub REQUIRED CONFIG PATHS "${HIP_PATH}/hipcub/")

    if(USE_HIP_AMD)
        find_package(rocprim REQUIRED CONFIG PATHS "${HIP_PATH}/rocprim")
    endif()
    
    set(SIMULATEQCD_INCLUDE_DIRS "${SIMULATEQCD_INCLUDE_DIRS};${HIP_PATH}/include;${HIP_INCLUDE_DIRS}")
    set(SIMULATEQCD_LIBRARIES "${SIMULATEQCD_LIBRARIES};${HIP_LIBRARIES}")
endif()

include_directories(SYSTEM "${SIMULATEQCD_INCLUDE_DIRS}")
link_libraries("${SIMULATEQCD_LIBRARIES}")


if (USE_CUDA)
    set(ARCHITECTURE "" CACHE STRING "Set the architecture. For example 70 will compile with Volta support")
    message(STATUS "CUDA Architecture: ${ARCHITECTURE}")
    set(CMAKE_CUDA_ARCHITECTURES "${ARCHITECTURE}")
elseif (USE_HIP_NVIDIA)
    set(ARCHITECTURE "" CACHE STRING "Set the (NVIDIA) architecture. For example 70 will compile with Volta support")
    message(STATUS "HIP (NVIDIA) Architecture: ${ARCHITECTURE}")
    set(CMAKE_CUDA_ARCHITECTURES "${ARCHITECTURE}")
    set(CMAKE_HIP_ARCHITECTURES "${ARCHITECTURE}")
elseif (USE_HIP_AMD)
    set(ARCHITECTURE "" CACHE STRING "Set the (AMD) architecture. For example gfx906,gfx908.")
    message(STATUS "HIP (AMD) Architecture: ${ARCHITECTURE}")
    set(CMAKE_HIP_ARCHITECTURES "${ARCHITECTURE}")
endif()

if(NOT ARCHITECTURE)
    message(FATAL_ERROR "No GPU architecture set!")
endif()

set(USE_GPU_AWARE_MPI OFF CACHE BOOL "Set to ON to build gpu-aware MPI code (default = OFF)")
if (USE_GPU_AWARE_MPI)
    add_definitions(-DUSE_GPU_AWARE_MPI)
endif()
set(USE_GPU_P2P ON CACHE BOOL "Set to ON to build with GPU Direct P2P (default = ON)")
if (USE_GPU_P2P)
    if(USE_HIP)
        message(FATAL_ERROR "GPU_P2P is not supported by HIP!")
    endif()
    add_definitions(-DUSE_GPU_P2P)
endif()


# Additional compiler flags
if (USE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-comment -fPIC")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -O3 -Wno-deprecated-gpu-targets --std=c++17 -arch=sm_${ARCHITECTURE} -Xcudafe --display_error_number -prec-div=true -prec-sqrt=true")

elseif (USE_HIP_AMD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-comment -fPIC -fgpu-rdc")

    add_definitions(-D__HIP_PLATFORM_AMD__)
    set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} "-O3 -std=c++17 -D__HIP_PLATFORM_AMD__ --amdgpu-target=${ARCHITECTURE} -fgpu-rdc")
    set(CMAKE_EXE_LINKER_FLAGS "-O3 -fgpu-rdc --hip-link")

elseif (USE_HIP_NVIDIA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-comment -fPIC")

    add_definitions(-D__HIP_PLATFORM_NVIDIA__ -D__HIP_PLATFORM_NVCC__)
    set(HIP_NVCC_FLAGS "${HIP_NVCC_FLAGS} -O3 -std=c++17 -D__HIP_PLATFORM_NVCC__ -D__HIP_PLATFORM_NVIDIA__ --generate-code arch=compute_${ARCHITECTURE},code=sm_${ARCHITECTURE} --generate-code arch=compute_${ARCHITECTURE},code=compute_${ARCHITECTURE} -rdc=true")
    set(NVCC_LINK_FLAGS "${NVCC_LINK_FLAGS} -rdc=true")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3")
endif()

option(USE_MPFR "Set to ON to include MPFR - needed for rat_approx (default - OFF)" OFF)

# --------------------------------------------------------------------------------------------------VARIABLE DEFINITIONS


set(SOURCE_FILES_BASE
    src/base/gutils.cpp
    src/base/memoryManagement.cpp
    src/base/indexer/initGPUIndexer.cpp
    src/base/indexer/initCPUIndexer.cpp
    src/base/communication/communicationBase_mpi.cpp
    src/base/IO/parameterManagement.cpp
    src/base/IO/fileWriter.cpp
    src/base/math/grnd.cpp
    src/gauge/gaugefield_device.cpp
    src/gauge/gaugefield.cpp
    src/gauge/GaugeAction.cpp
    src/base/LatticeContainer.cpp)
set(SOURCE_FILES_COLORELECTRICCORRTEST
    src/modules/observables/ColorElectricCorr.cpp
    src/modules/observables/PolyakovLoop.cpp)
set(SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST
    src/modules/observables/ColorElectricCorr.cpp
    src/modules/observables/ColorMagneticCorr.cpp
    src/modules/observables/PolyakovLoop.cpp)
set(SOURCE_FILES_DSLASH
        src/modules/dslash/dslash.cpp 
		src/modules/HISQ/hisqSmearing.cpp 
		src/spinor/spinorfield.cpp)
set(SOURCE_FILES_GENERATEQUENCHED
    src/modules/gauge_updates/PureGaugeUpdates.cpp
    src/modules/observables/PolyakovLoop.cpp)
set(SOURCE_FILES_GFIX
    src/modules/gaugeFixing/gfix.cpp
    src/modules/observables/PolyakovLoop.cpp
    src/modules/gaugeFixing/PolyakovLoopCorrelator.cpp
    src/modules/observables/WilsonLineCorrelator.cpp)
set(SOURCE_FILES_GFLOW
    src/modules/observables/Topology.cpp
    src/modules/observables/Blocking.cpp
    src/modules/observables/ColorElectricCorr.cpp
    src/modules/observables/ColorMagneticCorr.cpp)
set(SOURCE_FILES_GETMAGICTRACEANOMALY
    src/modules/gauge_updates/SubLatMeas_device.cpp)
set(SOURCE_FILES_MEASUREHADRONS
    src/modules/measureHadrons/measureHadrons.cpp
    src/modules/HISQ/hisqSmearing.cpp
    src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp
    src/spinor/spinorfield.cpp)
set(SOURCE_FILES_SUBLATTICEUPDATES
    src/modules/gauge_updates/luscherweisz.cpp
    src/modules/gauge_updates/SubLatMeas.cpp
    src/modules/gauge_updates/SubLatMeas_device.cpp)
set(SOURCE_FILES_HMC
    src/modules/rhmc/pure_gauge_hmc.cpp
    src/modules/rhmc/integrator.cpp
    src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp
    src/modules/HISQ/hisqSmearing.cpp
    src/modules/HISQ/hisqForce.cpp
    src/spinor/spinorfield.cpp)
set(SOURCE_FILES_RHMC
    src/modules/rhmc/rhmc.cpp
    src/modules/rhmc/integrator.cpp
    src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp
    src/modules/HISQ/hisqSmearing.cpp
    src/modules/HISQ/hisqForce.cpp
    src/modules/observables/PolyakovLoop.cpp
    src/spinor/spinorfield.cpp)
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

FUNCTION(set_SIMULATeQCD_gpu_backend TARGET)
    if (USE_CUDA)
        set_source_files_properties(${TARGET} ${ARGN} PROPERTIES LANGUAGE CUDA)
    elseif (USE_HIP)
        set_source_files_properties(${TARGET} ${ARGN} PROPERTIES LANGUAGE CXX HIP_SOURCE_PROPERTY_FORMAT 1)
    endif()
ENDFUNCTION()


#-------------------------------------------------------------------------------------------------------SET GPU BACKEND

set_SIMULATeQCD_gpu_backend(
    src/base/LatticeContainer.cpp
    src/base/indexer/initGPUIndexer.cpp
    src/base/math/grnd.cpp
    src/gauge/GaugeAction.cpp
    src/gauge/gaugefield_device.cpp
    src/gauge/gauge_kernels.cpp
    src/modules/HISQ/hisqForce.cpp
    src/modules/HISQ/hisqSmearing.cpp
    src/modules/dslash/dslash.cpp
    src/modules/gaugeFixing/PolyakovLoopCorrelator.cpp
    src/modules/gaugeFixing/gfix.cpp
    src/modules/gauge_updates/PureGaugeUpdates.cpp
    src/modules/gauge_updates/SubLatMeas_device.cpp
    src/modules/gauge_updates/luscherweisz.cpp
    src/modules/inverter/inverter.cpp
    src/modules/measureHadrons/measureHadrons.cpp
    src/modules/observables/Blocking.cpp
    src/modules/observables/ColorElectricCorr.cpp
    src/modules/observables/ColorMagneticCorr.cpp
    src/modules/observables/PolyakovLoop.cpp
    src/modules/observables/Topology.cpp
    src/modules/observables/WilsonLineCorrelator.cpp
    src/modules/rhmc/integrator.cpp
    src/modules/rhmc/pure_gauge_hmc.cpp
    src/modules/rhmc/rhmc.cpp
    src/spinor/spinorfield.cpp
    )

if (USE_HIP_AMD)
set_SIMULATeQCD_gpu_backend(
    src/gauge/gaugefield.cpp
    )
endif()

if (USE_CUDA)
    add_library(CodeBase OBJECT ${SOURCE_FILES_BASE})
    target_compile_definitions(CodeBase PRIVATE
        ARCHITECTURE=${ARCHITECTURE}
        SINGLEPREC=1 DOUBLEPREC=1
        COMP_R18=1 COMP_U3R14=1 COMP_R14=1 COMP_R12=1 COMP_STAGGR12=1
        ${ALL_HALODEPTHS}
        HALODEPTHSPIN_0=1 HALODEPTHSPIN_4=1
        ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS} GIT_HASH="${GIT_HASH}"
        )
# This might work in the future
#elseif (USE_HIP)
#    hip_add_library(CodeBase OBJECT ${SOURCE_FILES_BASE})
endif()

if (USE_CUDA)
    set_target_properties(CodeBase
        PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON)

# This might work in the future
#elseif (USE_HIP_AMD)
#    set_target_properties(CodeBase
#        PROPERTIES
#        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}"  HIP_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX HIP_RESOLVE_DEVICE_SYMBOLS ON)
#
#elseif (USE_HIP_NVIDIA)
#    set_target_properties(CodeBase
#        PROPERTIES
#        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}"  HIP_SEPARABLE_COMPILATION ON CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#
endif()

#---------------------------------------------------------------------------------------TARGET FUNCTIONS WITH PROPERTIES


FUNCTION(add_SIMULATeQCD_executable TARGET)
    if (USE_CUDA)
        add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
        add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
        
        target_link_libraries("_${TARGET}" CodeBase)
        set_target_properties(${TARGET} _${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON)
    elseif (USE_HIP_AMD)
        hip_add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
        hip_add_executable(_${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target

# This might work in the future
#        hip_add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
#        target_link_libraries("_${TARGET}" CodeBase hip::hipcub)
        set_target_properties(${TARGET} _${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX HIP_RESOLVE_DEVICE_SYMBOLS ON)

    elseif (USE_HIP_NVIDIA)
        hip_add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
        hip_add_executable(_${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target

# This might work in the future
#        hip_add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
#        target_link_libraries("_${TARGET}" CodeBase hip::hipcub)
        set_target_properties(${TARGET} _${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
ENDFUNCTION()

FUNCTION(set_SIMULATeQCD_property TARGET)
    set_target_properties(${TARGET} _${TARGET} ${ARGN})
ENDFUNCTION()

FUNCTION(SIMULATeQCD_target_compile_definitions TARGET)
    target_compile_definitions(${TARGET} ${ARGN} ARCHITECTURE=${ARCHITECTURE} GIT_HASH="${GIT_HASH}")
    target_compile_definitions(_${TARGET} ${ARGN} ARCHITECTURE=${ARCHITECTURE} GIT_HASH="${GIT_HASH}")
ENDFUNCTION()

FUNCTION(add_to_compound_SIMULATeQCD_target TARGET1 TARGET2)
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


set_SIMULATeQCD_gpu_backend(src/testing/main_BulkIndexerTest.cpp)
add_SIMULATeQCD_executable(BulkIndexerTest src/testing/main_BulkIndexerTest.cpp)
set_SIMULATeQCD_property(BulkIndexerTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(BulkIndexerTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests BulkIndexerTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_ColorElectricCorrTest.cpp)
add_SIMULATeQCD_executable(ColorElectricCorrTest src/testing/main_ColorElectricCorrTest.cpp ${SOURCE_FILES_COLORELECTRICCORRTEST})
set_SIMULATeQCD_property(ColorElectricCorrTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(ColorElectricCorrTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests ColorElectricCorrTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_CompressionTest.cpp)
add_SIMULATeQCD_executable(CompressionTest src/testing/main_CompressionTest.cpp)
set_SIMULATeQCD_property(CompressionTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(CompressionTest PRIVATE HALODEPTH_2=1  DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 COMP_R12=1 COMP_STAGGR12=1)
add_to_compound_SIMULATeQCD_target(tests CompressionTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_condensate_test.cpp)
add_SIMULATeQCD_executable(condensateTest src/testing/main_condensate_test.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(condensateTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(condensateTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests condensateTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_confReadWriteTest.cpp src/gauge/gaugefield.cpp)
add_SIMULATeQCD_executable(confReadWriteTest src/testing/main_confReadWriteTest.cpp)
set_SIMULATeQCD_property(confReadWriteTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(confReadWriteTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests confReadWriteTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_correlatorTest.cpp)
add_SIMULATeQCD_executable(correlatorTest src/testing/main_correlatorTest.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(correlatorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(correlatorTest PRIVATE HALODEPTH_0=1  DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests correlatorTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_cudaIpcTest.cpp)
add_SIMULATeQCD_executable(cudaIpcTest src/testing/main_cudaIpcTest.cpp)
set_SIMULATeQCD_property(cudaIpcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(cudaIpcTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests cudaIpcTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_DslashTest.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(DslashTest src/testing/main_DslashTest.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(DslashTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(DslashTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests DslashTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_DslashImagmuTest.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(DslashImagmuTest src/testing/main_DslashImagmuTest.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(DslashImagmuTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(DslashImagmuTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests DslashImagmuTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_DslashMultiTest.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(DslashMultiTest src/testing/main_DslashMultiTest.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(DslashMultiTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(DslashMultiTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests DslashMultiTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_GeneralFunctorTest.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(GeneralFunctorTest src/testing/main_GeneralFunctorTest.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(GeneralFunctorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(GeneralFunctorTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests GeneralFunctorTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_gfixplcTest.cpp)
add_SIMULATeQCD_executable(gfixplcTest src/testing/main_gfixplcTest.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(gfixplcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(gfixplcTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests gfixplcTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_gfixTestMulti.cpp)
add_SIMULATeQCD_executable(gfixTestMulti src/testing/main_gfixTestMulti.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(gfixTestMulti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(gfixTestMulti PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests gfixTestMulti)

set_SIMULATeQCD_gpu_backend(src/testing/main_gfixTestSingle.cpp)
add_SIMULATeQCD_executable(gfixTestSingle src/testing/main_gfixTestSingle.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(gfixTestSingle PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(gfixTestSingle PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests gfixTestSingle)

set_SIMULATeQCD_gpu_backend(src/testing/main_gradientFlowTest.cpp src/modules/observables/Topology.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(gradientFlowTest src/testing/main_gradientFlowTest.cpp src/modules/observables/Topology.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
set_SIMULATeQCD_property(gradientFlowTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(gradientFlowTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(tests gradientFlowTest)

if (NOT USE_HIP)
set_SIMULATeQCD_gpu_backend(src/testing/main_half_prec_math_test.cpp src/spinor/spinorfield.cpp )
add_SIMULATeQCD_executable(halfPrecMathTest src/testing/main_half_prec_math_test.cpp src/spinor/spinorfield.cpp )
set_SIMULATeQCD_property(halfPrecMathTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(halfPrecMathTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 SINGLEPREC=1 HALFPREC=1 COMP_R18=1 COMP_U3R14=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests halfPrecMathTest)
endif()

set_SIMULATeQCD_gpu_backend(src/testing/main_HaloTest.cpp)
add_SIMULATeQCD_executable(HaloTest src/testing/main_HaloTest.cpp)
set_SIMULATeQCD_property(HaloTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(HaloTest PRIVATE HALODEPTH_0=1 HALODEPTH_1=1 HALODEPTH_2=1  HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests HaloTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_HBOR_multiple_test.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(HBOR_multiple_test src/testing/main_HBOR_multiple_test.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
set_SIMULATeQCD_property(HBOR_multiple_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(HBOR_multiple_test PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests HBOR_multiple_test)

set_SIMULATeQCD_gpu_backend(src/testing/main_HBOR_single_test.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(HBOR_single_test src/testing/main_HBOR_single_test.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
set_SIMULATeQCD_property(HBOR_single_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(HBOR_single_test PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests HBOR_single_test)

set_SIMULATeQCD_gpu_backend(src/testing/main_hisqForce.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/HISQ/hisqForce.cpp) 
add_SIMULATeQCD_executable(hisqForce src/testing/main_hisqForce.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/HISQ/hisqForce.cpp) 
set_SIMULATeQCD_property(hisqForce PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqForce PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_0=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests hisqForce)

set_SIMULATeQCD_gpu_backend(src/testing/main_HisqSmearingTestMulti.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(hisqSmearingMulti src/testing/main_HisqSmearingTestMulti.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(hisqSmearingMulti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingMulti PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingMulti)

set_SIMULATeQCD_gpu_backend(src/testing/main_HisqSmearing_Create_Multi.cpp src/modules/HISQ/hisqSmearing.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(hisqSmearingMultiCreate src/testing/main_HisqSmearing_Create_Multi.cpp src/modules/HISQ/hisqSmearing.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(hisqSmearingMultiCreate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingMultiCreate PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingMultiCreate)

set_SIMULATeQCD_gpu_backend(src/testing/main_HisqSmearingTest.cpp src/modules/HISQ/hisqSmearing.cpp)
add_SIMULATeQCD_executable(hisqSmearingTest src/testing/main_HisqSmearingTest.cpp src/modules/HISQ/hisqSmearing.cpp)
set_SIMULATeQCD_property(hisqSmearingTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingTest)


set_SIMULATeQCD_gpu_backend(src/testing/main_HypSmearingTest.cpp src/modules/hyp/hypSmearing.cpp)
add_SIMULATeQCD_executable(hypSmearingTest src/testing/main_HypSmearingTest.cpp src/modules/hyp/hypSmearing.cpp)
set_SIMULATeQCD_property(hypSmearingTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hypSmearingTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hypSmearingTest)


set_SIMULATeQCD_gpu_backend(src/testing/main_HisqSmearingImagmuTest.cpp src/modules/HISQ/hisqSmearing.cpp)
add_SIMULATeQCD_executable(hisqSmearingImagmuTest src/testing/main_HisqSmearingImagmuTest.cpp src/modules/HISQ/hisqSmearing.cpp)
set_SIMULATeQCD_property(hisqSmearingImagmuTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingImagmuTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingImagmuTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_InverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(InverterTest src/testing/main_InverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(InverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(InverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests InverterTest)

if (NOT USE_HIP)
set_SIMULATeQCD_gpu_backend(src/testing/main_MixedPrecInverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(MixedPrecInverterTest src/testing/main_MixedPrecInverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(MixedPrecInverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(MixedPrecInverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 HALFPREC=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests MixedPrecInverterTest)
endif()

set_SIMULATeQCD_gpu_backend(src/testing/main_dotProduct.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(dotProductTest src/testing/main_dotProduct.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(dotProductTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(dotProductTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests dotProductTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_LinkPathTest.cpp)
add_SIMULATeQCD_executable(LinkPathTest src/testing/main_LinkPathTest.cpp)
set_SIMULATeQCD_property(LinkPathTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(LinkPathTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests LinkPathTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_measureHadronsTest.cpp ${SOURCE_FILES_MEASUREHADRONS})
add_SIMULATeQCD_executable(measureHadronsTest src/testing/main_measureHadronsTest.cpp ${SOURCE_FILES_MEASUREHADRONS})
set_SIMULATeQCD_property(measureHadronsTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(measureHadronsTest PRIVATE COMP_R18=1 HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R14=1 NSTACKS_1=1 NSTACKS_3=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests measureHadronsTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_memManTest.cpp)
add_SIMULATeQCD_executable(memManTest src/testing/main_memManTest.cpp)
set_SIMULATeQCD_property(memManTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(memManTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests memManTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_pure_gauge_hmc_test.cpp src/modules/observables/PolyakovLoop.cpp)
add_SIMULATeQCD_executable(PureGaugeHmcTest src/testing/main_pure_gauge_hmc_test.cpp src/modules/observables/PolyakovLoop.cpp ${SOURCE_FILES_HMC})
set_SIMULATeQCD_property(PureGaugeHmcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(PureGaugeHmcTest PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests PureGaugeHmcTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_rhmc_test.cpp)
add_SIMULATeQCD_executable(rhmcTest src/testing/main_rhmc_test.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(rhmcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(rhmcTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests rhmcTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_rnd_multiple_test.cpp)
add_SIMULATeQCD_executable(RndMultipleTest src/testing/main_rnd_multiple_test.cpp)
set_SIMULATeQCD_property(RndMultipleTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(RndMultipleTest PRIVATE HALODEPTH_1=1 HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests RndMultipleTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_rnd_single_test.cpp)
add_SIMULATeQCD_executable(RndSingleTest src/testing/main_rnd_single_test.cpp)
set_SIMULATeQCD_property(RndSingleTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(RndSingleTest PRIVATE HALODEPTH_1=1 HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests RndSingleTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_SpinorHaloTest.cpp)
add_SIMULATeQCD_executable(SpinorHaloTest src/testing/main_SpinorHaloTest.cpp)
set_SIMULATeQCD_property(SpinorHaloTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(SpinorHaloTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests SpinorHaloTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_StackedSpinorTest.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(StackedSpinorTest src/testing/main_StackedSpinorTest.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(StackedSpinorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(StackedSpinorTest PRIVATE HALODEPTH_4=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests StackedSpinorTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_TaylorMeasurementTest.cpp src/modules/observables/DslashDerivative.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(TaylorMeasurementTest src/testing/main_TaylorMeasurementTest.cpp src/modules/dslash/dslash.cpp src/modules/inverter/inverter.cpp 
src/modules/HISQ/hisqSmearing.cpp src/spinor/spinorfield.cpp src/modules/observables/DslashDerivative.cpp)
set_SIMULATeQCD_property(TaylorMeasurementTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(TaylorMeasurementTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 ${ALL_NSTACK} NSTACKS_32=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests TaylorMeasurementTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_cudaAwareMPITest.cpp)
add_SIMULATeQCD_executable(cudaAwareMPITest src/testing/main_cudaAwareMPITest.cpp)
set_SIMULATeQCD_property(cudaAwareMPITest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(cudaAwareMPITest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests cudaAwareMPITest)

# This one is special because it does not need the CodeBase
set_SIMULATeQCD_gpu_backend(src/testing/main_SimpleFunctorTest.cpp)
if (USE_CUDA)
    add_executable(SimpleFunctorTest src/testing/main_SimpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    add_executable(_SimpleFunctorTest src/testing/main_SimpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    set_target_properties(SimpleFunctorTest _SimpleFunctorTest PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON RUNTIME_OUTPUT_DIRECTORY "testing")
elseif (USE_HIP_AMD)
    hip_add_executable(SimpleFunctorTest src/testing/main_SimpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    hip_add_executable(_SimpleFunctorTest src/testing/main_SimpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    set_target_properties(SimpleFunctorTest _SimpleFunctorTest PROPERTIES 
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX HIP_RESOLVE_DEVICE_SYMBOLS ON RUNTIME_OUTPUT_DIRECTORY "testing")
elseif (USE_HIP_NVIDIA)
    hip_add_executable(SimpleFunctorTest src/testing/main_SimpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    hip_add_executable(_SimpleFunctorTest src/testing/main_SimpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    set_target_properties(SimpleFunctorTest _SimpleFunctorTest PROPERTIES 
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}"  HIP_SEPARABLE_COMPILATION ON CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON RUNTIME_OUTPUT_DIRECTORY "testing")
endif()
target_compile_definitions(SimpleFunctorTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 ARCHITECTURE=${ARCHITECTURE} GIT_HASH="${GIT_HASH}")
target_compile_definitions(_SimpleFunctorTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 ARCHITECTURE=${ARCHITECTURE} GIT_HASH="${GIT_HASH}")
add_to_compound_SIMULATeQCD_target(tests SimpleFunctorTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_UtimesUdaggerTest.cpp)
add_SIMULATeQCD_executable(UtimesUdaggerTest src/testing/main_UtimesUdaggerTest.cpp)
set_SIMULATeQCD_property(UtimesUdaggerTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(UtimesUdaggerTest PRIVATE HALODEPTH_0=1 HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests UtimesUdaggerTest)


#-------------------------------------------------------------------------------------COMPILATION DEFINITIONS: PROFILING


set_SIMULATeQCD_gpu_backend(src/profiling/main_ColorElectricMagneticCloverBenchmark.cpp ${SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST})
add_SIMULATeQCD_executable(ColorElectricMagneticCloverBenchmark src/profiling/main_ColorElectricMagneticCloverBenchmark.cpp ${SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST})
set_SIMULATeQCD_property(ColorElectricMagneticCloverBenchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(ColorElectricMagneticCloverBenchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(profilers ColorElectricMagneticCloverBenchmark)

set_SIMULATeQCD_gpu_backend(src/profiling/main_FieldStrengthTensorBenchmark.cpp)
add_SIMULATeQCD_executable(FieldStrengthTensorBenchmark src/profiling/main_FieldStrengthTensorBenchmark.cpp)
set_SIMULATeQCD_property(FieldStrengthTensorBenchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(FieldStrengthTensorBenchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(profilers FieldStrengthTensorBenchmark)

set_SIMULATeQCD_gpu_backend(src/profiling/main_ForceProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/HISQ/hisqForce.cpp)
add_SIMULATeQCD_executable(ForceProfiling src/profiling/main_ForceProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/HISQ/hisqForce.cpp)
set_SIMULATeQCD_property(ForceProfiling PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(ForceProfiling PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_2=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers ForceProfiling)

set_SIMULATeQCD_gpu_backend(src/profiling/main_HBOR_benchmark.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(HBOR_benchmark src/profiling/main_HBOR_benchmark.cpp src/modules/gauge_updates/PureGaugeUpdates.cpp)
set_SIMULATeQCD_property(HBOR_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(HBOR_benchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(profilers HBOR_benchmark)

set_SIMULATeQCD_gpu_backend(src/profiling/main_InverterProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(InverterProf src/profiling/main_InverterProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(InverterProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(InverterProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers InverterProf)

set_SIMULATeQCD_gpu_backend(src/profiling/main_MrhsDSlashProf.cpp ${SOURCE_FILES_DSLASH}) 
add_SIMULATeQCD_executable(MultiRHSProf src/profiling/main_MrhsDSlashProf.cpp ${SOURCE_FILES_DSLASH}) 
set_SIMULATeQCD_property(MultiRHSProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(MultiRHSProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers MultiRHSProf)

if (NOT USE_HIP)
set_SIMULATeQCD_gpu_backend(src/profiling/main_half_prec_dslash_profile.cpp ${SOURCE_FILES_DSLASH}) 
add_SIMULATeQCD_executable(MultiRHSProf_half src/profiling/main_half_prec_dslash_profile.cpp ${SOURCE_FILES_DSLASH}) 
set_SIMULATeQCD_property(MultiRHSProf_half PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(MultiRHSProf_half PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 HALFPREC=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers MultiRHSProf_half)
endif()


#--------------------------------------------------------------------------------------COMPILATION DEFINITIONS: EXAMPLES


set_SIMULATeQCD_gpu_backend(src/examples/main_gradientFlow_example.cpp)
add_SIMULATeQCD_executable(gradientFlow_example src/examples/main_gradientFlow_example.cpp)
set_SIMULATeQCD_property(gradientFlow_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(gradientFlow_example PRIVATE HALODEPTH_3=1 DOUBLEPREC=1 COMP_R18=1 ADAPTIVE_STEPSIZE=1 ZEUTHEN_FLOW=1)
add_to_compound_SIMULATeQCD_target(examples gradientFlow_example)

set_SIMULATeQCD_gpu_backend(src/examples/main_plaquette.cpp)
add_SIMULATeQCD_executable(plaquette src/examples/main_plaquette.cpp)
set_SIMULATeQCD_property(plaquette PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(plaquette PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(examples plaquette)

set_SIMULATeQCD_gpu_backend(src/examples/main_polyakovloop.cpp)
add_SIMULATeQCD_executable(PolyakovLoop src/examples/main_polyakovloop.cpp)
set_SIMULATeQCD_property(PolyakovLoop PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(PolyakovLoop PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(examples PolyakovLoop)

set_SIMULATeQCD_gpu_backend(src/examples/main_wilson_lines_fields.cpp src/modules/gaugeFixing/gfix.cpp)
add_SIMULATeQCD_executable(WilsonLinesFields src/examples/main_wilson_lines_fields.cpp src/modules/gaugeFixing/gfix.cpp)
set_SIMULATeQCD_property(WilsonLinesFields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(WilsonLinesFields PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(examples WilsonLinesFields)

set_SIMULATeQCD_gpu_backend(src/examples/main_wilson_lines_fields_stacked_shared.cpp src/modules/gaugeFixing/gfix.cpp)
add_SIMULATeQCD_executable(WilsonLinesFieldsShared src/examples/main_wilson_lines_fields_stacked_shared.cpp src/modules/gaugeFixing/gfix.cpp)
set_SIMULATeQCD_property(WilsonLinesFieldsShared PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(WilsonLinesFieldsShared PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(examples WilsonLinesFieldsShared)


#---------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: APPLICATIONS

# This gives a cmake error when using hip, so commented it out.
#set_SIMULATeQCD_gpu_backend(src/applications/main_fluxtube.cpp)
#add_SIMULATeQCD_executable(fluxtube src/applications/main_fluxtube.cpp ${SOURCE_FILES_BASE} ${SOURCE_FILES_GFIX})
#set_SIMULATeQCD_property(fluxtube PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
#SIMULATeQCD_target_compile_definitions(fluxtube PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
##add_to_compound_SIMULATeQCD_target(applications fluxtube)

set_SIMULATeQCD_gpu_backend(src/applications/main_configConverter.cpp)
add_SIMULATeQCD_executable(configConverter src/applications/main_configConverter.cpp) 
set_SIMULATeQCD_property(configConverter PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(configConverter PRIVATE HALODEPTH_0=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(applications configConverter)

add_SIMULATeQCD_executable(CheckConf src/applications/main_CheckConf.cpp)
set_SIMULATeQCD_property(CheckConf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(CheckConf PRIVATE HALODEPTH_0=1 COMP_R18=1 SINGLEPREC=1 DOUBLEPREC=1 CPUONLY=1)
add_to_compound_SIMULATeQCD_target(applications CheckConf)

add_SIMULATeQCD_executable(CheckRand src/applications/main_CheckRand.cpp)
set_SIMULATeQCD_property(CheckRand PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(CheckRand PRIVATE HALODEPTH_0=1 CPUONLY=1)
add_to_compound_SIMULATeQCD_target(applications CheckRand)

set_SIMULATeQCD_gpu_backend(src/applications/main_GenerateQuenched.cpp) 
add_SIMULATeQCD_executable(GenerateQuenched src/applications/main_GenerateQuenched.cpp ${SOURCE_FILES_GENERATEQUENCHED})
set_SIMULATeQCD_property(GenerateQuenched PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(GenerateQuenched PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(applications GenerateQuenched)

set_SIMULATeQCD_gpu_backend(src/applications/main_gaugeFixing.cpp) 
add_SIMULATeQCD_executable(gaugeFixing src/applications/main_gaugeFixing.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(gaugeFixing PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(gaugeFixing PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(applications gaugeFixing)

set_SIMULATeQCD_gpu_backend(src/applications/main_gradientFlow.cpp)
add_SIMULATeQCD_executable(gradientFlow src/applications/main_gradientFlow.cpp ${SOURCE_FILES_GFLOW} ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(gradientFlow PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications") 
SIMULATeQCD_target_compile_definitions(gradientFlow PRIVATE COMP_R18=1 HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 DOUBLEPREC=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(applications gradientFlow)

set_SIMULATeQCD_gpu_backend(src/applications/main_measureHadrons.cpp)
add_SIMULATeQCD_executable(measureHadrons src/applications/main_measureHadrons.cpp ${SOURCE_FILES_MEASUREHADRONS})
set_SIMULATeQCD_property(measureHadrons PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(measureHadrons PRIVATE COMP_R18=1 HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R14=1 NSTACKS_1=1 NSTACKS_3=1 ${ALL_LAYOUTS})

set_SIMULATeQCD_gpu_backend(src/applications/main_polSuscRenorm.cpp)
add_SIMULATeQCD_executable(polSuscRenorm src/applications/main_polSuscRenorm.cpp src/modules/observables/PolyakovLoop.cpp)
set_SIMULATeQCD_property(polSuscRenorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(polSuscRenorm PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(applications polSuscRenorm)

if (USE_HIP_AMD)
set_SIMULATeQCD_gpu_backend(src/applications/main_rhmc.cpp)
add_SIMULATeQCD_executable(RHMC src/applications/main_rhmc.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(RHMC PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(RHMC PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(applications RHMC)
else()
set_SIMULATeQCD_gpu_backend(src/applications/main_rhmc.cpp)
add_SIMULATeQCD_executable(RHMC src/applications/main_rhmc.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(RHMC PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(RHMC PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 HALFPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 ${REALLY_ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(applications RHMC)
endif()

set_SIMULATeQCD_gpu_backend(src/applications/main_sampleTopology.cpp)
add_SIMULATeQCD_executable(sampleTopology src/applications/main_sampleTopology.cpp ${SOURCE_FILES_GFLOW} ${SOURCE_FILES_GENERATEQUENCHED})
set_SIMULATeQCD_property(sampleTopology PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(sampleTopology PRIVATE COMP_R18=1 HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 DOUBLEPREC=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(applications sampleTopology)

set_SIMULATeQCD_gpu_backend(src/applications/main_sublatticeUpdates.cpp)
add_SIMULATeQCD_executable(sublatticeUpdates src/applications/main_sublatticeUpdates.cpp ${SOURCE_FILES_SUBLATTICEUPDATES})
set_SIMULATeQCD_property(sublatticeUpdates PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(sublatticeUpdates PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(applications sublatticeUpdates)

set_SIMULATeQCD_gpu_backend(src/applications/main_wilsonLinesCorrelatorMultiGPUStacked.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/WilsonLineCorrelatorMultiGPU.cpp)
add_SIMULATeQCD_executable(wilsonLinesCorrelatorMultiGPUStacked src/applications/main_wilsonLinesCorrelatorMultiGPUStacked.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/WilsonLineCorrelatorMultiGPU.cpp)
set_SIMULATeQCD_property(wilsonLinesCorrelatorMultiGPUStacked PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(wilsonLinesCorrelatorMultiGPUStacked PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(applications wilsonLinesCorrelatorMultiGPUStacked)


#-----------------------------------------------------------------------------------------COMPILATION DEFINITIONS: TOOLS


set_SIMULATeQCD_gpu_backend(src/tools/main_createCorrelatorNorm.cpp)
add_SIMULATeQCD_executable(correlatorNorm src/tools/main_createCorrelatorNorm.cpp)
set_SIMULATeQCD_property(correlatorNorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "tools")
SIMULATeQCD_target_compile_definitions(correlatorNorm PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tools correlatorNorm)


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
