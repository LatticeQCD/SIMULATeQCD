# How to compile

The following software is required to compile SIMULATeQCD:

1. `cmake3` (Some versions have the "--phtread" compiler bug. Versions that definetely work are [3.14.6](https://gitlab.kitware.com/cmake/cmake/tree/) v3.14.6 or 3.19.2 
2. `C++` compiler with `C++17` support  (e.g. `g++-9`)
3. `MPI` (e.g. `openmpi-4.0.4`) and
4. `CUDA Toolkit` version 11.0 (NOT 11.1 or 11.2)

To setup the compilation, create a folder outside of the code directory (e.g. `../build/`) and *from there* call the following example script (see source:/doc/cmake.sh, you need to manually change this for your machine): 
```shell
cmake ../simulateqcd/ \
-DARCHITECTURE="70" \
-DUSE_CUDA_AWARE_MPI=ON \
-DUSE_CUDA_P2P=ON \
``` 
Here, it is assumed that your source code folder is called `simulateqcd`. *Do NOT compile your code in the source code folder!* 
You can set the path to CUDA by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR:PATH`.
`-DARCHITECTURE` sets the GPU architecture (i.e. [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) version without the decimal point). For example, "20" will compile for Fermi, "35" for Kepler, "60" for Pascal and "70" for Volta. 
Inside the build folder, you can now begin to use `make` to compile your executables, e.g. 
```shell
make NameOfExecutable
```
If you would like to speed up the compiling process, add the option `-j`, which will compile in parallel using all available CPU threads. You can also specify the number of threads manually using, for example, `-j 4`.





## Compilation on Bielefeld GPU cluster

Loading the appropriate software required to compile on the cluster can be accomplished with `module load`. In particular you can
```shell
module load compilers/gnu
module load mpi/openmpi
module load cuda
```



