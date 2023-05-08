Getting started
===============

## How to download the code

First, make sure you have activated git-lfs using `git lfs install`.
The code can then be cloned to your folder using:
```shell
git clone https://github.com/LatticeQCD/SIMULATeQCD.git
```
If you are using two-factor authentication on GitHub, you may need to use the command
```shell
git clone git@github.com:LatticeQCD/SIMULATeQCD.git
```

## Prerequisites

The following software is required to compile SIMULATeQCD:

1. [git-lfs](https://git-lfs.github.com/) to also be able to clone test configurations:
    ```shell
    # For Debian-based system
    sudo apt install git-lfs

    # For Arch-based system
    sudo pacman -S git-lfs
    ```
    and activate it by calling `git lfs install`. If you do not have superuser privileges where you are, you can use [wget](https://www.gnu.org/software/wget/) as follows:
    ```shell
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz
    tar -xf git-lfs-linux-amd64-v3.0.2.tar.gz
    PREFIX=/path/to/install/dir ./install.sh
    ```
    and you can activate it with `/path/to/install/dir/bin/git-lfs install`. Finally you will need to add `export PATH=/path/to/install/dir/bin:$PATH` to your `.bashrc`.
2. `cmake` (Some versions have the "--pthread" compiler bug. Versions that definitely work are [3.14.6](https://gitlab.kitware.com/cmake/cmake/tree/v3.14.6) or 3.19.2.)
3. `C++` compiler with `C++17` support.
4. `MPI` (e.g. `openmpi-4.0.4`).
5. `CUDA Toolkit` version 11+ or `HIP`.
6. `pip install -r requirements.txt` to build the documentation.

## Building source with CUDA
To build the source with CUDA, you need to have the `CUDA Toolkit` version 11.0 or higher installed on your machine.
To setup the compilation, create a folder outside of the code directory (e.g. `../build/`) and **from there** call the following example script:
```shell
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="70" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=ON \
```
Here, it is assumed that your source code folder is called `SIMULATeQCD`. **Do NOT compile your code in the source code folder!**
You can set the CUDA installation path manually by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR`.
`-DARCHITECTURE` sets the GPU architecture (i.e. [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) version without the decimal point). For example "60" for Pascal and "70" for Volta.
Inside the build folder, you can now begin to use `make` to compile your executables, e.g.
```shell
make NameOfExecutable
```
If you would like to speed up the compiling process, add the option `-j`, which will compile in parallel using all available CPU threads. You can also specify the number of threads manually using, for example, `-j 4`.

## Building source with HIP for NVIDIA platforms (Experimental!)

In order to build the source with HIP for NVIDIA platforms,
you need to make sure that
- HIP is properly installed on your machine
- CUDA is properly installed on your machine
- The environment variable `HIP_PATH` holds the path to the HIP installation folder
- The environment variables `CC` and `CXX` hold the path to the HIP clang compiler

To setup the compilation, create a folder outside of the code directory (e.g. `../build/`) and **from there** call the following example script:
```shell
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="70" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=OFF \
-DBACKEND="hip_nvidia" \
```
Here, it is assumed that your source code folder is called `SIMULATeQCD`. **Do NOT compile your code in the source code folder!**
You can set the HIP installation path manually by setting the `cmake` parameter `-DHIP_PATH`.
You can also set the CUDA installation path manually by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR`.
`-DARCHITECTURE` sets the GPU architecture (i.e. [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) version without the decimal point). For example "60" for Pascal and "70" for Volta.
`-DUSE_GPU_P2P=ON` is not yet supported by this backend.
Inside the build folder, you can now begin to use `make` to compile your executables, e.g.
```shell
make NameOfExecutable
```
If you would like to speed up the compiling process, add the option `-j`, which will compile in parallel using all available CPU threads. You can also specify the number of threads manually using, for example, `-j 4`.

## Building source with HIP for AMD platforms (Experimental!)

In order to build the source with HIP for AMD platforms,
you need to make sure that
- HIP is properly installed on your machine
- The environment variable `HIP_PATH` holds the path to the HIP installation folder
- The environment variables `CC` and `CXX` hold the path to the HIP clang compiler

To setup the compilation, create a folder outside of the code directory (e.g. `../build/`) and **from there** call the following example script:
```shell
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="gfx906,gfx908" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=OFF \
-DBACKEND="hip_amd" \
```
Here, it is assumed that your source code folder is called `SIMULATeQCD`. **Do NOT compile your code in the source code folder!**
You can set the HIP installation path manually by setting the `cmake` parameter `-DHIP_PATH`.
`-DARCHITECTURE` sets the GPU architecture. For example gfx906,gfx908.
`-DUSE_GPU_P2P=ON` is not yet supported by this backend.
Inside the build folder, you can now begin to use `make` to compile your executables, e.g.
```shell
make NameOfExecutable
```
If you would like to speed up the compiling process, add the option `-j`, which will compile in parallel using all available CPU threads. You can also specify the number of threads manually using, for example, `-j 4`.


## How to run


### On a cluster using `slurm`

If you are on a cluster that uses slurm, e.g. the Bielefeld GPU cluster, then, inside of your sbatch script do not use mpiexec or mpirun, but instead do
```shell
srun -n <NoGPUs> ./<program>
```

### On your local machine (desktop, laptop, ...)

Any program has to be launched using mpirun or mpiexec.
For example:
```shell
mpiexec -np <NoGPUs> ./<program>
```
where `<NoGPUs>` is the number of GPUs you want to use.
