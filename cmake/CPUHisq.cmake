# First CPU milestone: existing HISQ tests, standard C++ and ordinary MPI.
# Kept separate from GPU target registration (many other targets use raw kernels).
if(USE_GPU_AWARE_MPI OR USE_GPU_P2P OR USE_NCCL OR USE_MARKER)
    message(FATAL_ERROR "CPU requires USE_GPU_AWARE_MPI, USE_GPU_P2P, USE_NCCL and USE_MARKER all OFF")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "CPU prototype currently supports GCC/Clang with _Float16")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(CpuHisqOptions INTERFACE)
target_compile_options(CpuHisqOptions INTERFACE -fno-fast-math -ffp-contract=off)

set(CPU_ARCHITECTURE "portable" CACHE STRING "CPU instruction set: portable, native, avx512")
set_property(CACHE CPU_ARCHITECTURE PROPERTY STRINGS portable native avx512)
if(CPU_ARCHITECTURE STREQUAL "portable")
    # Keep the compiler's default target; do not detect the configure host.
    set(_cpu_arch_flags "")
elseif(CPU_ARCHITECTURE STREQUAL "native")
    set(_cpu_arch_flags -march=native)
elseif(CPU_ARCHITECTURE STREQUAL "avx512")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        message(FATAL_ERROR "CPU_ARCHITECTURE=avx512 requires an x86-64 target")
    endif()
    # AVX-512 Foundation supplies eight double lanes. Do not assume extra
    # AVX-512 subsets or a particular Intel/AMD microarchitecture.
    set(_cpu_arch_flags -mavx512f)
else()
    message(FATAL_ERROR "Unknown CPU_ARCHITECTURE='${CPU_ARCHITECTURE}'; choose portable, native, or avx512")
endif()
if(_cpu_arch_flags)
    include(CheckCXXCompilerFlag)
    foreach(_cpu_arch_flag IN LISTS _cpu_arch_flags)
        string(MAKE_C_IDENTIFIER "CPU_SUPPORTS_${_cpu_arch_flag}" _cpu_arch_check)
        check_cxx_compiler_flag("${_cpu_arch_flag}" ${_cpu_arch_check})
        if(NOT ${_cpu_arch_check})
            message(FATAL_ERROR "Compiler does not support ${_cpu_arch_flag}")
        endif()
    endforeach()
    target_compile_options(CpuHisqOptions INTERFACE ${_cpu_arch_flags})
endif()
message(STATUS "CPU architecture: ${CPU_ARCHITECTURE}; extra target flags: ${_cpu_arch_flags}")
target_compile_definitions(CpuHisqOptions INTERFACE
    SINGLEPREC=1 DOUBLEPREC=1 HALFPREC=0
    COMP_R18=1 COMP_R14=1 COMP_U3R14=1
    ${ALL_HALODEPTHS} HALODEPTHSPIN_4=1
    ${ALL_NSTACK} ${ALL_LAYOUTS}
    GIT_HASH="${GIT_HASH}")
target_link_libraries(CpuHisqOptions INTERFACE MPI::MPI_CXX)

add_library(CpuHisqBase STATIC ${SOURCE_FILES_BASE})
target_link_libraries(CpuHisqBase PUBLIC CpuHisqOptions)
target_compile_definitions(CpuHisqBase PRIVATE HALODEPTHSPIN_0=1)
add_library(CpuHisqSmearing STATIC src/modules/hisq/hisqSmearing.cpp)
target_link_libraries(CpuHisqSmearing PUBLIC CpuHisqBase)
add_executable(cpuBackendTest src/testing/main_cpuBackendTest.cpp)
target_link_libraries(cpuBackendTest PRIVATE CpuHisqBase)
add_executable(hisqSmearingTest src/testing/main_hisqSmearingTest.cpp)
add_executable(hisqSmearingRecursiveTest src/testing/main_hisqSmearingRecursiveTest.cpp)
target_link_libraries(hisqSmearingTest PRIVATE CpuHisqSmearing)
target_link_libraries(hisqSmearingRecursiveTest PRIVATE CpuHisqSmearing)
add_library(CpuHisqFermions STATIC
    src/modules/hisq/hisqForce.cpp src/modules/dslash/dslash.cpp
    src/modules/inverter/inverter.cpp src/spinor/spinorfield.cpp)
target_link_libraries(CpuHisqFermions PUBLIC CpuHisqSmearing)
# Dslash tests need depth 0, but RHMC's integrator requires spinor depth 4.
# Keep depth-0 instantiations private so they cannot propagate into CpuRhmc.
target_compile_definitions(CpuHisqFermions PRIVATE HALODEPTHSPIN_0=1)
add_executable(hisqForce src/testing/main_hisqForce.cpp)
target_link_libraries(hisqForce PRIVATE CpuHisqFermions)
add_executable(dslashTest src/testing/main_dslashTest.cpp)
add_executable(inverterTest src/testing/main_inverterTest.cpp)
target_link_libraries(dslashTest PRIVATE CpuHisqFermions)
target_compile_definitions(dslashTest PRIVATE HALODEPTHSPIN_0=1)
target_link_libraries(inverterTest PRIVATE CpuHisqFermions)
add_library(CpuRhmc STATIC src/modules/rhmc/rhmc.cpp
    src/modules/rhmc/integrator.cpp src/modules/observables/polyakovLoop.cpp)
target_link_libraries(CpuRhmc PUBLIC CpuHisqFermions)
add_executable(rhmcTest src/testing/main_rhmc_test.cpp)
target_link_libraries(rhmcTest PRIVATE CpuRhmc)
set_target_properties(cpuBackendTest hisqSmearingTest hisqSmearingRecursiveTest hisqForce
    dslashTest inverterTest rhmcTest
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/testing")
add_custom_target(cpuHisqTests DEPENDS cpuBackendTest hisqSmearingTest hisqSmearingRecursiveTest hisqForce)
message(STATUS "CPU milestone: serial synchronous kernels; single-rank HISQ regression only")

add_custom_target(cpuRhmcTests DEPENDS cpuHisqTests dslashTest inverterTest rhmcTest)
