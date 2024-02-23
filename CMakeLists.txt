#
# CMakeLists.txt
#
# CMake file for programs using the multiGPU code. Compilation target definitions are organized by type, and then
# alphabetically. Please try to follow these guidelines when adding your target.
#
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

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


set(SIMULATEQCD_INCLUDE_DIRS "${MPI_INCLUDE_PATH};$ENV{MPICH_DIR}/include")
set(SIMULATEQCD_LIBRARIES "${MPI_LIBRARIES}")

if (USE_CUDA OR USE_HIP_NVIDIA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER_LOADED)
        enable_language(CUDA)
    endif()
    set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
    set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
    find_package(CUDAToolkit REQUIRED)
    set(SIMULATEQCD_INCLUDE_DIRS "${SIMULATEQCD_INCLUDE_DIRS};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
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

    #set(CMAKE_MODULE_PATH "${HIP_PATH}/hip/cmake" ${CMAKE_MODULE_PATH})
    enable_language(HIP)
    find_package(hipcub REQUIRED CONFIG PATHS "${HIP_PATH}/hipcub/")

    if(USE_HIP_AMD)
        find_package(rocprim REQUIRED CONFIG PATHS "${HIP_PATH}/rocprim")
        #find_package(rccl REQUIRED CONFIG PATHS "${HIP_PATH}/rccl")
    endif()
    set(HIP_INCLUDE_DIRS "${HIP_PATH}/include")
    set(SIMULATEQCD_INCLUDE_DIRS "${SIMULATEQCD_INCLUDE_DIRS};${HIP_INCLUDE_DIRS}")
    set(SIMULATEQCD_LIBRARIES "${SIMULATEQCD_LIBRARIES};${HIP_LIBRARIES}")
    message(STATUS "Using HIP_PATH ${HIP_PATH} with HIP_INCLUDE_DIRS: ${HIP_INCLUDE_DIRS} and HIP_INCLUDE_DIR: ${HIP_INCLUDE_DIR}")
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
    add_definitions(-DUSE_GPU_P2P)
endif()

set(USE_MARKER OFF CACHE BOOL "Set to ON to build with GPU Marker API (default = OFF)")
if (USE_MARKER)
    add_definitions(-DUSE_MARKER)
    if (USE_CUDA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnvToolsExt")
    elseif(USE_HIP)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROCM_PATH}/roctracer/lib -lroctracer64 -lroctx64")
    endif()
endif()

set(USE_TILED_MULTIRHS ON CACHE BOOL "Set to OFF to build without most Multi-RHS DSlash instantiations (default = ON)")
if(USE_TILED_MULTIRHS)
    add_definitions(-DUSE_TILED_MULTIRHS)
endif()

set(USE_NCCL OFF CACHE BOOL "Set to ON to build with NCCL/RCCL support (default = OFF)")
if (USE_NCCL)
    add_definitions(-DUSE_NCCL)
    if (USE_HIP)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lrccl")
    endif()
endif()

set(COMMS_STREAMS ON CACHE BOOL "Set to ON to build with 160 streams for communications, OFF for 2 streams (default = ON)")
if (COMMS_STREAMS)
    add_definitions(-DCOMMS_STREAMS)
endif()

# Additional compiler flags
if (USE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fPIC")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -O3 -Wno-deprecated-gpu-targets --std=c++17 -arch=sm_${ARCHITECTURE} -Xcudafe --display_error_number -prec-div=true -prec-sqrt=true --expt-relaxed-constexpr")

elseif (USE_HIP_AMD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-comment -Wno-deprecated-copy-with-user-provided-copy -fPIC -fgpu-rdc")

    add_definitions(-D__HIP_PLATFORM_AMD__)
    set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} "-O3 -std=c++17 -D__HIP_PLATFORM_AMD__ --offload-arch=${ARCHITECTURE} -fgpu-rdc")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -fgpu-rdc --hip-link --craype-prepend-opt=-Wl,-rpath=${ROCM_PATH}/deps")

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
    src/base/math/random.cpp
    src/gauge/gaugefield_device.cpp
    src/gauge/gaugefield.cpp
    src/gauge/gaugeAction.cpp
    src/base/latticeContainer.cpp)
set(SOURCE_FILES_COLORELECTRICCORRTEST
    src/modules/observables/colorElectricCorr.cpp
    src/modules/observables/polyakovLoop.cpp)
set(SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST
    src/modules/observables/colorElectricCorr.cpp
    src/modules/observables/colorMagneticCorr.cpp
    src/modules/observables/polyakovLoop.cpp)
set(SOURCE_FILES_DSLASH
        src/modules/dslash/dslash.cpp 
		src/modules/hisq/hisqSmearing.cpp 
		src/spinor/spinorfield.cpp)
set(SOURCE_FILES_GENERATEQUENCHED
    src/modules/gauge_updates/pureGaugeUpdates.cpp
    src/modules/observables/polyakovLoop.cpp)
set(SOURCE_FILES_GFIX
    src/modules/gaugeFixing/gfix.cpp
    src/modules/observables/polyakovLoop.cpp
    src/modules/gaugeFixing/polyakovLoopCorrelator.cpp
    src/modules/observables/wilsonLineCorrelator.cpp)
set(SOURCE_FILES_GFLOW
    src/modules/observables/topology.cpp
    src/modules/observables/weinberg.cpp
    src/modules/observables/blocking.cpp
    src/modules/observables/colorElectricCorr.cpp
    src/modules/observables/colorMagneticCorr.cpp)
set(SOURCE_FILES_GETMAGICTRACEANOMALY
    src/modules/gauge_updates/subLatMeas_device.cpp)
set(SOURCE_FILES_MEASUREHADRONS
    src/modules/measureHadrons/measureHadrons.cpp
    src/modules/hisq/hisqSmearing.cpp
    src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp
    src/spinor/spinorfield.cpp)
set(SOURCE_FILES_SUBLATTICEUPDATES
    src/modules/gauge_updates/luscherweisz.cpp
    src/modules/gauge_updates/subLatMeas.cpp
    src/modules/gauge_updates/subLatMeas_device.cpp)
set(SOURCE_FILES_HMC
    src/modules/rhmc/pure_gauge_hmc.cpp
    src/modules/rhmc/integrator.cpp
    src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp
    src/modules/hisq/hisqSmearing.cpp
    src/modules/hisq/hisqForce.cpp
    src/spinor/spinorfield.cpp)
set(SOURCE_FILES_RHMC
    src/modules/rhmc/rhmc.cpp
    src/modules/rhmc/integrator.cpp
    src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp
    src/modules/hisq/hisqSmearing.cpp
    src/modules/hisq/hisqForce.cpp
    src/modules/observables/polyakovLoop.cpp
    src/spinor/spinorfield.cpp)
set(ALL_NSTACK
    NSTACKS_1=1
    NSTACKS_2=1
    NSTACKS_4=1
    NSTACKS_8=1
    NSTACKS_10=1
    NSTACKS_12=1
    NSTACKS_14=1)
set(ALL_NSTACK_BLOCK
    NSTACKS_BLOCK_1=1
    NSTACKS_BLOCK_2=1
    NSTACKS_BLOCK_3=1
    NSTACKS_BLOCK_4=1
    NSTACKS_BLOCK_5=1
    NSTACKS_BLOCK_6=1
    NSTACKS_BLOCK_7=1
    NSTACKS_BLOCK_8=1)
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
    src/base/latticeContainer.cpp
    src/base/indexer/initGPUIndexer.cpp
    src/base/math/random.cpp
    src/gauge/gaugeAction.cpp
    src/gauge/gaugefield_device.cpp
    src/gauge/gauge_kernels.cpp
    src/modules/hisq/hisqForce.cpp
    src/modules/hisq/hisqSmearing.cpp
    src/modules/dslash/dslash.cpp
    src/modules/gaugeFixing/polyakovLoopCorrelator.cpp
    src/modules/gaugeFixing/gfix.cpp
    src/modules/gauge_updates/pureGaugeUpdates.cpp
    src/modules/gauge_updates/subLatMeas_device.cpp
    src/modules/gauge_updates/luscherweisz.cpp
    src/modules/inverter/inverter.cpp
    src/modules/measureHadrons/measureHadrons.cpp
    src/modules/observables/blocking.cpp
    src/modules/observables/colorElectricCorr.cpp
    src/modules/observables/colorMagneticCorr.cpp
    src/modules/observables/polyakovLoop.cpp
    src/modules/observables/topology.cpp
    src/modules/observables/weinberg.cpp
    src/modules/observables/wilsonLineCorrelator.cpp
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
        ${ALL_NSTACK} ${ALL_LAYOUTS} GIT_HASH="${GIT_HASH}"
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
FUNCTION(add_standalone_executable TARGET)
if (USE_CUDA)
        add_executable(${TARGET} ${ARGN}) # single target

        target_link_libraries("_${TARGET}")
        set_target_properties(${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON)
    elseif (USE_HIP_AMD)
        add_executable(${TARGET} ${ARGN}) # single target
        
# This might work in the future
#        add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
#        target_link_libraries("_${TARGET}" CodeBase hip::hipcub)
        set_target_properties(${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX HIP_RESOLVE_DEVICE_SYMBOLS ON PUBLIC "-L${ROCM_PATH}/roctracer/lib -lroctracer64 -lroctx64")

    elseif (USE_HIP_NVIDIA)
        add_executable(${TARGET} ${ARGN}) # single target
        
# This might work in the future
#        add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
#        target_link_libraries("_${TARGET}" CodeBase hip::hipcub)
        set_target_properties(${TARGET} PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
ENDFUNCTION()

FUNCTION(add_SIMULATeQCD_executable TARGET)
    if (USE_CUDA)
        add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
        add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
        
        target_link_libraries("_${TARGET}" CodeBase)
        set_target_properties(${TARGET} _${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" CUDA_SEPARABLE_COMPILATION ON)
    elseif (USE_HIP_AMD)
        add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
        add_executable(_${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target

# This might work in the future
#        add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
#        target_link_libraries("_${TARGET}" CodeBase hip::hipcub)
        set_target_properties(${TARGET} _${TARGET}
            PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX HIP_RESOLVE_DEVICE_SYMBOLS ON PUBLIC "-L${ROCM_PATH}/roctracer/lib -lroctracer64 -lroctx64")

    elseif (USE_HIP_NVIDIA)
        add_executable(${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target
        add_executable(_${TARGET} ${ARGN} ${SOURCE_FILES_BASE}) # single target

# This might work in the future
#        add_executable(_${TARGET} ${ARGN}) # compound target (e.g. in "tests", "applications")
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
add_custom_target(experimental)
add_custom_target(profilers)
add_custom_target(tests)
add_custom_target(tools)
add_custom_target(everything)
add_dependencies(everything
    applications
    examples
    experimental
    profilers
    tests
    tools
    )


#----------------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: TESTS


set_SIMULATeQCD_gpu_backend(src/testing/main_bulkIndexerTest.cpp)
add_SIMULATeQCD_executable(bulkIndexerTest src/testing/main_bulkIndexerTest.cpp)
set_SIMULATeQCD_property(bulkIndexerTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(bulkIndexerTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests bulkIndexerTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_colorElectricCorrTest.cpp)
add_SIMULATeQCD_executable(colorElectricCorrTest src/testing/main_colorElectricCorrTest.cpp ${SOURCE_FILES_COLORELECTRICCORRTEST})
set_SIMULATeQCD_property(colorElectricCorrTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(colorElectricCorrTest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests colorElectricCorrTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_stopwatch.cpp)
add_SIMULATeQCD_executable(stopwatchTest src/testing/main_stopwatch.cpp)
set_SIMULATeQCD_property(stopwatchTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(stopwatchTest PRIVATE HALODEPTH_1=1)
add_to_compound_SIMULATeQCD_target(tests stopwatchTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_compressionTest.cpp)
add_SIMULATeQCD_executable(compressionTest src/testing/main_compressionTest.cpp)
set_SIMULATeQCD_property(compressionTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(compressionTest PRIVATE HALODEPTH_2=1  DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 COMP_R12=1 COMP_STAGGR12=1)
add_to_compound_SIMULATeQCD_target(tests compressionTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_condensate_test.cpp)
add_SIMULATeQCD_executable(condensateTest src/testing/main_condensate_test.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(condensateTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(condensateTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14 NSTACKS_1=1 NSTACKS_10=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
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

set_SIMULATeQCD_gpu_backend(src/testing/main_dslashTest.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(dslashTest src/testing/main_dslashTest.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(dslashTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(dslashTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests dslashTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_dslashImagmuTest.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(dslashImagmuTest src/testing/main_dslashImagmuTest.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(dslashImagmuTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(dslashImagmuTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests dslashImagmuTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_dslashMultiTest.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(dslashMultiTest src/testing/main_dslashMultiTest.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(dslashMultiTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(dslashMultiTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests dslashMultiTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_generalFunctorTest.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(generalFunctorTest src/testing/main_generalFunctorTest.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(generalFunctorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(generalFunctorTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_U3R14=1 COMP_R18=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests generalFunctorTest)

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

set_SIMULATeQCD_gpu_backend(src/testing/main_gradientFlowTest.cpp src/modules/observables/topology.cpp src/modules/observables/weinberg.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(gradientFlowTest src/testing/main_gradientFlowTest.cpp src/modules/observables/topology.cpp src/modules/observables/weinberg.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
set_SIMULATeQCD_property(gradientFlowTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(gradientFlowTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 WILSON_FLOW=1 ZEUTHEN_FLOW=1 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(tests gradientFlowTest)

if (NOT USE_HIP)
set_SIMULATeQCD_gpu_backend(src/testing/main_half_prec_math_test.cpp src/spinor/spinorfield.cpp )
add_SIMULATeQCD_executable(halfPrecMathTest src/testing/main_half_prec_math_test.cpp src/spinor/spinorfield.cpp )
set_SIMULATeQCD_property(halfPrecMathTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(halfPrecMathTest PRIVATE HALODEPTH_0=1 HALODEPTHSPIN_0=1 DOUBLEPREC=1 SINGLEPREC=1 HALFPREC=1 COMP_R18=1 COMP_U3R14=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests halfPrecMathTest)
endif()

set_SIMULATeQCD_gpu_backend(src/testing/main_haloTest.cpp)
add_SIMULATeQCD_executable(haloTest src/testing/main_haloTest.cpp)
set_SIMULATeQCD_property(haloTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(haloTest PRIVATE HALODEPTH_0=1 HALODEPTH_1=1 HALODEPTH_2=1  HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests haloTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_hbor_multiple_test.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(hbor_multiple_test src/testing/main_hbor_multiple_test.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
set_SIMULATeQCD_property(hbor_multiple_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hbor_multiple_test PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hbor_multiple_test)

set_SIMULATeQCD_gpu_backend(src/testing/main_hbor_single_test.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(hbor_single_test src/testing/main_hbor_single_test.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
set_SIMULATeQCD_property(hbor_single_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hbor_single_test PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hbor_single_test)

set_SIMULATeQCD_gpu_backend(src/testing/main_hisqForce.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/hisq/hisqForce.cpp) 
add_SIMULATeQCD_executable(hisqForce src/testing/main_hisqForce.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/hisq/hisqForce.cpp) 
set_SIMULATeQCD_property(hisqForce PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqForce PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_0=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests hisqForce)

set_SIMULATeQCD_gpu_backend(src/testing/main_hisqSmearingTestMulti.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(hisqSmearingMulti src/testing/main_hisqSmearingTestMulti.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(hisqSmearingMulti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingMulti PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingMulti)

set_SIMULATeQCD_gpu_backend(src/testing/main_hisqSmearing_Create_Multi.cpp src/modules/hisq/hisqSmearing.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(hisqSmearingMultiCreate src/testing/main_hisqSmearing_Create_Multi.cpp src/modules/hisq/hisqSmearing.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(hisqSmearingMultiCreate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingMultiCreate PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingMultiCreate)

set_SIMULATeQCD_gpu_backend(src/testing/main_hisqSmearingTest.cpp src/modules/hisq/hisqSmearing.cpp)
add_SIMULATeQCD_executable(hisqSmearingTest src/testing/main_hisqSmearingTest.cpp src/modules/hisq/hisqSmearing.cpp)
set_SIMULATeQCD_property(hisqSmearingTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_hypSmearingTest.cpp src/modules/hyp/hypSmearing.cpp)
add_SIMULATeQCD_executable(hypSmearingTest src/testing/main_hypSmearingTest.cpp src/modules/hyp/hypSmearing.cpp)
set_SIMULATeQCD_property(hypSmearingTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hypSmearingTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hypSmearingTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_hisqSmearingImagmuTest.cpp src/modules/hisq/hisqSmearing.cpp)
add_SIMULATeQCD_executable(hisqSmearingImagmuTest src/testing/main_hisqSmearingImagmuTest.cpp src/modules/hisq/hisqSmearing.cpp)
set_SIMULATeQCD_property(hisqSmearingImagmuTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(hisqSmearingImagmuTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 HALODEPTH_4=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests hisqSmearingImagmuTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_inverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(inverterTest src/testing/main_inverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(inverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(inverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests inverterTest)

if (NOT USE_HIP)
set_SIMULATeQCD_gpu_backend(src/testing/main_mixedPrecInverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(mixedPrecInverterTest src/testing/main_mixedPrecInverterTest.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(mixedPrecInverterTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(mixedPrecInverterTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 HALFPREC=1 SINGLEPREC=1 NSTACKS_1=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests mixedPrecInverterTest)
endif()

set_SIMULATeQCD_gpu_backend(src/testing/main_dotProduct.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(dotProductTest src/testing/main_dotProduct.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(dotProductTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(dotProductTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests dotProductTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_linkPathTest.cpp)
add_SIMULATeQCD_executable(linkPathTest src/testing/main_linkPathTest.cpp)
set_SIMULATeQCD_property(linkPathTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(linkPathTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests linkPathTest)

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

set_SIMULATeQCD_gpu_backend(src/testing/main_pure_gauge_hmc_test.cpp src/modules/observables/polyakovLoop.cpp)
add_SIMULATeQCD_executable(pureGaugeHmcTest src/testing/main_pure_gauge_hmc_test.cpp src/modules/observables/polyakovLoop.cpp ${SOURCE_FILES_HMC})
set_SIMULATeQCD_property(pureGaugeHmcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(pureGaugeHmcTest PRIVATE HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests pureGaugeHmcTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_rhmc_test.cpp)
add_SIMULATeQCD_executable(rhmcTest src/testing/main_rhmc_test.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(rhmcTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(rhmcTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests rhmcTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_rnd_multiple_test.cpp)
add_SIMULATeQCD_executable(rndMultipleTest src/testing/main_rnd_multiple_test.cpp)
set_SIMULATeQCD_property(rndMultipleTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(rndMultipleTest PRIVATE HALODEPTH_1=1 HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests rndMultipleTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_rnd_single_test.cpp)
add_SIMULATeQCD_executable(rndSingleTest src/testing/main_rnd_single_test.cpp)
set_SIMULATeQCD_property(rndSingleTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(rndSingleTest PRIVATE HALODEPTH_1=1 HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests rndSingleTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_spinorHaloTest.cpp)
add_SIMULATeQCD_executable(spinorHaloTest src/testing/main_spinorHaloTest.cpp)
set_SIMULATeQCD_property(spinorHaloTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(spinorHaloTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests spinorHaloTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_stackedSpinorTest.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(stackedSpinorTest src/testing/main_stackedSpinorTest.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(stackedSpinorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(stackedSpinorTest PRIVATE HALODEPTH_4=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests stackedSpinorTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_taylorMeasurementTest.cpp src/modules/observables/dslashDerivative.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(taylorMeasurementTest src/testing/main_taylorMeasurementTest.cpp src/modules/dslash/dslash.cpp src/modules/inverter/inverter.cpp 
src/modules/hisq/hisqSmearing.cpp src/spinor/spinorfield.cpp src/modules/observables/dslashDerivative.cpp)
set_SIMULATeQCD_property(taylorMeasurementTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(taylorMeasurementTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 ${ALL_NSTACK} NSTACKS_32=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests taylorMeasurementTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_cudaAwareMPITest.cpp)
add_SIMULATeQCD_executable(cudaAwareMPITest src/testing/main_cudaAwareMPITest.cpp)
set_SIMULATeQCD_property(cudaAwareMPITest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(cudaAwareMPITest PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(tests cudaAwareMPITest)

# This one is special because it does not need the CodeBase
set_SIMULATeQCD_gpu_backend(src/testing/main_simpleFunctorTest.cpp)
if (USE_CUDA)
    add_executable(simpleFunctorTest src/testing/main_simpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    add_executable(_simpleFunctorTest src/testing/main_simpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    set_target_properties(simpleFunctorTest _simpleFunctorTest PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON RUNTIME_OUTPUT_DIRECTORY "testing")
elseif (USE_HIP_AMD)
    add_executable(simpleFunctorTest src/testing/main_simpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    add_executable(_simpleFunctorTest src/testing/main_simpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    set_target_properties(simpleFunctorTest _simpleFunctorTest PROPERTIES 
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}" HIP_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX HIP_RESOLVE_DEVICE_SYMBOLS ON RUNTIME_OUTPUT_DIRECTORY "testing")
elseif (USE_HIP_NVIDIA)
    add_executable(simpleFunctorTest src/testing/main_simpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    add_executable(_simpleFunctorTest src/testing/main_simpleFunctorTest.cpp src/base/communication/communicationBase_mpi.cpp src/base/gutils.cpp)
    set_target_properties(simpleFunctorTest _simpleFunctorTest PROPERTIES 
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}" LINK_FLAGS "${MPI_LINK_FLAGS}"  HIP_SEPARABLE_COMPILATION ON CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON RUNTIME_OUTPUT_DIRECTORY "testing")
endif()
target_compile_definitions(simpleFunctorTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 ARCHITECTURE=${ARCHITECTURE} GIT_HASH="${GIT_HASH}")
target_compile_definitions(_simpleFunctorTest PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 SINGLEPREC=1 ARCHITECTURE=${ARCHITECTURE} GIT_HASH="${GIT_HASH}")
add_to_compound_SIMULATeQCD_target(tests simpleFunctorTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_utimesUdaggerTest.cpp)
add_SIMULATeQCD_executable(utimesUdaggerTest src/testing/main_utimesUdaggerTest.cpp)
set_SIMULATeQCD_property(utimesUdaggerTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(utimesUdaggerTest PRIVATE HALODEPTH_0=1 HALODEPTH_1=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 SINGLEPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(tests utimesUdaggerTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_weinbergTopTest.cpp src/modules/observables/topology.cpp src/modules/observables/weinberg.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(weinbergTopTest src/testing/main_weinbergTopTest.cpp src/modules/observables/topology.cpp src/modules/observables/weinberg.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
set_SIMULATeQCD_property(weinbergTopTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(weinbergTopTest PRIVATE HALODEPTH_1=1 HALODEPTH_2=1 HALODEPTH_3=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 WILSON_FLOW=1 ZEUTHEN_FLOW=0 FIXED_STEPSIZE=1 ADAPTIVE_STEPSIZE=0)
add_to_compound_SIMULATeQCD_target(tests weinbergTopTest)

set_SIMULATeQCD_gpu_backend(src/testing/main_wilsonLinesCorrelatorTest.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/wilsonLineCorrelatorMultiGPU.cpp)
add_SIMULATeQCD_executable(wilsonLinesCorrelatorTest src/testing/main_wilsonLinesCorrelatorTest.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/wilsonLineCorrelatorMultiGPU.cpp )
set_SIMULATeQCD_property(wilsonLinesCorrelatorTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(wilsonLinesCorrelatorTest PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1 WILSON_FLOW=1 FIXED_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(tests wilsonLinesCorrelatorTest)


set_SIMULATeQCD_gpu_backend(src/testing/main_maximalCenterGaugeFixingTest.cpp)
add_SIMULATeQCD_executable(maximalCenterGaugeFixingTest src/testing/main_maximalCenterGaugeFixingTest.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(maximalCenterGaugeFixingTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "testing")
SIMULATeQCD_target_compile_definitions(maximalCenterGaugeFixingTest PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 WILSON_FLOW=1 FIXED_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(tests maximalCenterGaugeFixingTest)



#-------------------------------------------------------------------------------------COMPILATION DEFINITIONS: PROFILING


set_SIMULATeQCD_gpu_backend(src/profiling/main_colorElectricMagneticCloverBenchmark.cpp ${SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST})
add_SIMULATeQCD_executable(colorElectricMagneticCloverBenchmark src/profiling/main_colorElectricMagneticCloverBenchmark.cpp ${SOURCE_FILES_COLORELECTRICMAGNETICCLOVERTEST})
set_SIMULATeQCD_property(colorElectricMagneticCloverBenchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(colorElectricMagneticCloverBenchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(profilers colorElectricMagneticCloverBenchmark)

set_SIMULATeQCD_gpu_backend(src/profiling/main_fieldStrengthTensorBenchmark.cpp)
add_SIMULATeQCD_executable(fieldStrengthTensorBenchmark src/profiling/main_fieldStrengthTensorBenchmark.cpp)
set_SIMULATeQCD_property(fieldStrengthTensorBenchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(fieldStrengthTensorBenchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(profilers fieldStrengthTensorBenchmark)

set_SIMULATeQCD_gpu_backend(src/profiling/main_forceProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/hisq/hisqForce.cpp)
add_SIMULATeQCD_executable(forceProfiling src/profiling/main_forceProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp src/modules/hisq/hisqForce.cpp)
set_SIMULATeQCD_property(forceProfiling PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(forceProfiling PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_2=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers forceProfiling)

set_SIMULATeQCD_gpu_backend(src/profiling/main_7linkprof.cpp ${SOURCE_FILES_DSLASH})
add_SIMULATeQCD_executable(7linkprof src/profiling/main_7linkprof.cpp ${SOURCE_FILES_DSLASH})
set_SIMULATeQCD_property(7linkprof PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(7linkprof PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_2=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers 7linkprof)

set_SIMULATeQCD_gpu_backend(src/profiling/main_axpy.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(axpy src/profiling/main_axpy.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(axpy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(axpy PRIVATE HALODEPTHSPIN_4=1 HALODEPTH_2=1  SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers axpy)



set_SIMULATeQCD_gpu_backend(src/profiling/main_triad.cpp src/spinor/spinorfield.cpp)
add_SIMULATeQCD_executable(triad src/profiling/main_triad.cpp src/spinor/spinorfield.cpp)
set_SIMULATeQCD_property(triad PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(triad PRIVATE HALODEPTHSPIN_2=1 HALODEPTH_2=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers triad)

set_SIMULATeQCD_gpu_backend(src/profiling/main_hbor_benchmark.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
add_SIMULATeQCD_executable(hbor_benchmark src/profiling/main_hbor_benchmark.cpp src/modules/gauge_updates/pureGaugeUpdates.cpp)
set_SIMULATeQCD_property(hbor_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(hbor_benchmark PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(profilers hbor_benchmark)

set_SIMULATeQCD_gpu_backend(src/profiling/main_inverterProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(inverterProf src/profiling/main_inverterProfiling.cpp ${SOURCE_FILES_DSLASH} src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(inverterProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(inverterProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers inverterProf)

set_SIMULATeQCD_gpu_backend(src/profiling/main_cgprofiling.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp src/modules/inverter/inverter.cpp) 
add_SIMULATeQCD_executable(cgProf src/profiling/main_cgprofiling.cpp  src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp src/modules/inverter/inverter.cpp) 
set_SIMULATeQCD_property(cgProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(cgProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 NSTACKS_1=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers cgProf)


if (USE_TILED_MULTIRHS)
set_SIMULATeQCD_gpu_backend(src/profiling/main_mrhsDSlashProf.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp) 
add_SIMULATeQCD_executable(multiRHSProf src/profiling/main_mrhsDSlashProf.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp)
set_target_properties(multiRHSProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
target_compile_definitions(multiRHSProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 SINGLEPREC=1 DOUBLEPREC=1 NSTACKS_1=1 NSTACKS_2=1 NSTACKS_3=1 NSTACKS_4=1 NSTACKS_5=1 NSTACKS_6=1 NSTACKS_8=1 NSTACKS_10=1 NSTACKS_12=1 NSTACKS_15=1 NSTACKS_16=1 ${ALL_NSTACK_BLOCK} LAYOUT_EVEN=1 LAYOUT_ODD=1 GIT_HASH="${GIT_HASH}")
add_dependencies(profilers multiRHSProf)
else()
set_SIMULATeQCD_gpu_backend(src/profiling/main_mrhsDSlashProf.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp) 
add_SIMULATeQCD_executable(multiRHSProf src/profiling/main_mrhsDSlashProf.cpp src/modules/dslash/dslash.cpp src/spinor/spinorfield.cpp)
set_target_properties(multiRHSProf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
target_compile_definitions(multiRHSProf PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 SINGLEPREC=1 NSTACKS_1=1 NSTACKS_4=1 NSTACKS_12=1 NSTACKS_BLOCK_1=1 NSTACKS_BLOCK_2=1 NSTACKS_BLOCK_4=1 LAYOUT_EVEN=1 LAYOUT_ODD=1 GIT_HASH="${GIT_HASH}")
add_dependencies(profilers multiRHSProf)
endif()



if (NOT USE_HIP)
set_SIMULATeQCD_gpu_backend(src/profiling/main_half_prec_dslash_profile.cpp ${SOURCE_FILES_DSLASH}) 
add_SIMULATeQCD_executable(multiRHSProf_half src/profiling/main_half_prec_dslash_profile.cpp ${SOURCE_FILES_DSLASH}) 
set_SIMULATeQCD_property(multiRHSProf_half PROPERTIES RUNTIME_OUTPUT_DIRECTORY "profiling")
SIMULATeQCD_target_compile_definitions(multiRHSProf_half PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 COMP_R18=1 COMP_U3R14=1 COMP_R14=1 SINGLEPREC=1 HALFPREC=1 ${ALL_NSTACK} ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(profilers multiRHSProf_half)
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
add_SIMULATeQCD_executable(polyakovLoop src/examples/main_polyakovloop.cpp)
set_SIMULATeQCD_property(polyakovLoop PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(polyakovLoop PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(examples polyakovLoop)

set_SIMULATeQCD_gpu_backend(src/examples/main_wilson_lines_fields.cpp src/modules/gaugeFixing/gfix.cpp)
add_SIMULATeQCD_executable(wilsonLinesFields src/examples/main_wilson_lines_fields.cpp src/modules/gaugeFixing/gfix.cpp)
set_SIMULATeQCD_property(wilsonLinesFields PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(wilsonLinesFields PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(examples wilsonLinesFields)

set_SIMULATeQCD_gpu_backend(src/examples/main_wilson_lines_fields_stacked_shared.cpp src/modules/gaugeFixing/gfix.cpp)
add_SIMULATeQCD_executable(wilsonLinesFieldsShared src/examples/main_wilson_lines_fields_stacked_shared.cpp src/modules/gaugeFixing/gfix.cpp)
set_SIMULATeQCD_property(wilsonLinesFieldsShared PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples")
SIMULATeQCD_target_compile_definitions(wilsonLinesFieldsShared PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(examples wilsonLinesFieldsShared)

#---------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: EXPERIMENTAL

set_SIMULATeQCD_gpu_backend(src/experimental/main_wilsonloop.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/wilsonLineCorrelatorMultiGPU.cpp)
add_SIMULATeQCD_executable(wilsonloop src/experimental/main_wilsonloop.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/wilsonLineCorrelatorMultiGPU.cpp )
set_SIMULATeQCD_property(wilsonloop PROPERTIES RUNTIME_OUTPUT_DIRECTORY "experimental")
SIMULATeQCD_target_compile_definitions(wilsonloop PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1 WILSON_FLOW=1 FIXED_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(experimental wilsonloop)

#---------------------------------------------------------------------------COMPILATION TARGET DEFINITIONS: APPLICATIONS


set_SIMULATeQCD_gpu_backend(src/applications/main_configConverter.cpp)
add_SIMULATeQCD_executable(configConverter src/applications/main_configConverter.cpp) 
set_SIMULATeQCD_property(configConverter PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(configConverter PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 SINGLEPREC=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1)
add_to_compound_SIMULATeQCD_target(applications configConverter)

add_SIMULATeQCD_executable(checkConf src/applications/main_checkConf.cpp)
set_SIMULATeQCD_property(checkConf PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(checkConf PRIVATE HALODEPTH_0=1 COMP_R18=1 SINGLEPREC=1 DOUBLEPREC=1 CPUONLY=1)
add_to_compound_SIMULATeQCD_target(applications checkConf)

add_SIMULATeQCD_executable(checkRand src/applications/main_checkRand.cpp)
set_SIMULATeQCD_property(checkRand PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(checkRand PRIVATE HALODEPTH_0=1 CPUONLY=1)
add_to_compound_SIMULATeQCD_target(applications checkRand)

set_SIMULATeQCD_gpu_backend(src/applications/main_generateQuenched.cpp) 
add_SIMULATeQCD_executable(generateQuenched src/applications/main_generateQuenched.cpp ${SOURCE_FILES_GENERATEQUENCHED})
set_SIMULATeQCD_property(generateQuenched PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(generateQuenched PRIVATE HALODEPTH_1=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(applications generateQuenched)

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
add_SIMULATeQCD_executable(polSuscRenorm src/applications/main_polSuscRenorm.cpp src/modules/observables/polyakovLoop.cpp)
set_SIMULATeQCD_property(polSuscRenorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(polSuscRenorm PRIVATE HALODEPTH_0=1 DOUBLEPREC=1 COMP_R18=1)
add_to_compound_SIMULATeQCD_target(applications polSuscRenorm)

set_SIMULATeQCD_gpu_backend(src/applications/main_rhmc.cpp)
add_SIMULATeQCD_executable(rhmc src/applications/main_rhmc.cpp ${SOURCE_FILES_RHMC})
set_SIMULATeQCD_property(rhmc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(rhmc PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_4=1 DOUBLEPREC=1 SINGLEPREC=1 COMP_R18=1 COMP_R14=1 COMP_U3R14=1 NSTACKS_1=1 NSTACKS_10=1 NSTACKS_12=1 NSTACKS_14=1 ${ALL_LAYOUTS})
add_to_compound_SIMULATeQCD_target(applications rhmc)

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

set_SIMULATeQCD_gpu_backend(src/applications/main_wilsonLinesCorrelatorMultiGPUStacked.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/wilsonLineCorrelatorMultiGPU.cpp src/modules/hyp/hypSmearing.cpp)
add_SIMULATeQCD_executable(wilsonLinesCorrelatorMultiGPUStacked src/applications/main_wilsonLinesCorrelatorMultiGPUStacked.cpp src/modules/gaugeFixing/gfix.cpp src/modules/observables/wilsonLineCorrelatorMultiGPU.cpp src/modules/hyp/hypSmearing.cpp)
set_SIMULATeQCD_property(wilsonLinesCorrelatorMultiGPUStacked PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(wilsonLinesCorrelatorMultiGPUStacked PRIVATE HALODEPTH_2=1 HALODEPTHSPIN_2=1 DOUBLEPREC=1 COMP_R18=1 NSTACKS_1=1 LAYOUT_ALL=1 WILSON_FLOW=1 FIXED_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(applications wilsonLinesCorrelatorMultiGPUStacked)


set_SIMULATeQCD_gpu_backend(src/applications/main_maximalCenterGaugeFixing.cpp)
add_SIMULATeQCD_executable(maximalCenterGaugeFixing src/applications/main_maximalCenterGaugeFixing.cpp ${SOURCE_FILES_GFIX})
set_SIMULATeQCD_property(maximalCenterGaugeFixing PROPERTIES RUNTIME_OUTPUT_DIRECTORY "applications")
SIMULATeQCD_target_compile_definitions(maximalCenterGaugeFixing PRIVATE HALODEPTH_0=1 HALODEPTH_2=1 DOUBLEPREC=1 COMP_R18=1 WILSON_FLOW=1 FIXED_STEPSIZE=1)
add_to_compound_SIMULATeQCD_target(applications maximalCenterGaugeFixing)


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
