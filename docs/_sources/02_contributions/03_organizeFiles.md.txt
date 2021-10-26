# How to organize new files


All source and header files are stored in `src/<MeaningfulName>/...` .
For example, `src/base/*` holds source files of all base classes which are needed by most programs, e.g. the Lattice class or the CommunicationBase class.

If you want to add a new executable, be it an application, a test, a profiler, etc., you should save it as `src/<type_of_application>/main_<exec_name>.cu`. 
If you have added additional `*.cpp` or `*.cu` or header files which are needed by your main, you should save them in a new module folder, e.g. `src/modules/<exec_name>/<meaningfulName>.cpp` and add them to the source files
list in `CMakeLists.txt` (root folder):

```Cmake
set(SOURCE_FILES_<exec_name> <path/to/file1.cu> <path/to/file2.cpp>)
add_ParallelGPU_executable(<exec_name> src/<type_of_application>/main_<exec_name>.cu
                           ${SOURCE_FILES_<exec_name>})
set_ParallelGPU_property(<exec_name> PROPERTIES RUNTIME_OUTPUT_DIRECTORY "<type_of_application>")
ParallelGPU_target_compile_definitions(<exec_name> PRIVATE <your_compile_definitions>)
add_to_compound_ParallelGPU_target(<type_of_application> <exec_name>)
```

Headerfiles should NOT be listed there!
* Example programs are stored in `src/example/*`. Ideally, these programs should be as short as possible and strongly commented.
* Testing programs are stored in `src/testing/*`. These programs should check if everything works properly. 

An example configuration is stored in `test_conf/l20t20b06498a_nersc.302500`. It is a $20^4$ lattice with a $\beta=0.638$ and it is written in nersc.

Add the HaloDepth and precision which you want to use in you appication in `ParallelGPU_target_compile_definitions(...)`. Available definitions are written in define.h.

A nice example how to write GPU code may be found in 
`src/examples/main_plaquette.cu`
or 
`src/testing/main_GeneralOperatorTest.cu`.


