Getting started
===============

## How to download the code

The code can be copied to your folder using this command:

```shell
git clone https://github.com/LatticeQCD/SIMULATeQCD.git
```

If you are using two-factor authentication on GitHub, you may need to use the command

```shell
git clone git@github.com:LatticeQCD/SIMULATeQCD.git
```
## How to compile

The following software is required to compile SIMULATeQCD:

1. [git-lfs](https://git-lfs.github.com/) to also be able to clone test configurations:
    ```shell
    # For Debian-based system
    sudo apt install git-lfs
     
    # For Arch-based system
    sudo pacman -S git-lfs
    ```
    and activate it by calling `git lfs install`
2. `cmake` (Some versions have the "--phtread" compiler bug. Versions that definetely work are [3.14.6](https://gitlab.kitware.com/cmake/cmake/tree/v3.14.6) or 3.19.2.
3. `C++` compiler with `C++17` support  (e.g. `g++-9`).
4. `MPI` (e.g. `openmpi-4.0.4`).
5. `CUDA Toolkit` version 11.0 (NOT 11.1 or 11.2).
6. `pip install -r requirements.txt` to build the documentation.

To setup the compilation, create a folder outside of the code directory (e.g. `../build/`) and **from there** call the following example script: 
```shell
cmake ../simulateqcd/ \
-DARCHITECTURE="70" \
-DUSE_CUDA_AWARE_MPI=ON \
-DUSE_CUDA_P2P=ON \
``` 
Here, it is assumed that your source code folder is called `simulateqcd`. **Do NOT compile your code in the source code folder!**
You can set the path to CUDA by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR:PATH`.
`-DARCHITECTURE` sets the GPU architecture (i.e. [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) version without the decimal point). For example "60" for Pascal and "70" for Volta. 
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
where `<NoGPUs>` is the number of GPU's you want to use.
