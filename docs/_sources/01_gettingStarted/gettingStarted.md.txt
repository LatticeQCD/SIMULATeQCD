# Getting started

There are two possible ways to build SIMULATeQCD. If you are running on your own laptop or desktop and have an NVIDIA GPU,
we recommend that you use the container build. The container will automatically grab all software you need.
If you are running on an HPC system or want to use AMD, we recommmend you compile manually and ensure that all needed
software already exists on the system you're using.
## Prerequisites

Before cloning anything, we recommend you get `git-lfs`. The reason we recommend this is that we have several configurations
used to test some of the SIMULATeQCD methods. These are too large to keep in a conventional repository, so we host them
using `git-lfs`. SIMULATeQCD will work without it, but you won't be able to run the tests.

You can install [git-lfs](https://git-lfs.github.com/) by 
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

## How to download the code

First, make sure you have activated git-lfs using `git lfs install`.
The code can then be cloned to your folder using:
```shell
git clone https://github.com/LatticeQCD/SIMULATeQCD.git
```
If you are using two-factor authentication on GitHub, which is likely the case if you
are interested in developing SIMULATeQCD, you will need to use the command
```shell
git clone git@github.com:LatticeQCD/SIMULATeQCD.git
```

## Build (manual) 

If you would like to use AMD, would like to make substantial contributions to SIMULATeQCD, 
or are running on an HPC system, you will need to do a manual build. 
If you have an NVIDIA GPU, are running locally, and don't plan to do much development, we recommend you
skip down to the [container build](./gettingStarted.md#build-container) section, because the container will take
care of all the hassle of finding and installing software automatically.

For the manual build, you will need to make sure the following are installed:

1. `cmake` (Some versions have the "--pthread" compiler bug. Versions that definitely work are [3.14.6](https://gitlab.kitware.com/cmake/cmake/tree/v3.14.6) or [3.19.2](https://gitlab.kitware.com/cmake/cmake/-/tree/v3.19.2?ref_type=tags).)
2. `C++` compiler with `C++17` support.
3. `MPI` (e.g. `openmpi-4.0.4`).
4. `CUDA Toolkit` version 11+ or `HIP`.
5. `pip install -r requirements.txt` to build the documentation.

### Building source with CUDA

To build the source with CUDA, you need to have the `CUDA Toolkit` version 11.0 or higher installed on your machine.
To setup the compilation, create a folder outside of the code directory (e.g. `../buildSIMULATeQCD/`) and **from there** call the following example script:
```shell
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="80" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=ON \
```
Here, it is assumed that your source code folder is called `SIMULATeQCD`. **Do NOT compile your code in the source code folder!**
You can set the CUDA installation path manually by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR`.
`-DARCHITECTURE` sets the GPU architecture (i.e. [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) version without the decimal point). For example use "70" for Volta or "80" for Ampere.

### Building source with HIP for NVIDIA platforms (Experimental!)

In order to build the source with HIP for NVIDIA platforms,
you need to make sure that
- HIP is properly installed on your machine
- CUDA is properly installed on your machine
- The environment variable `HIP_PATH` holds the path to the HIP installation folder
- The environment variables `CC` and `CXX` hold the path to the HIP clang compiler

To setup the compilation, create a folder outside of the code directory (e.g. `../buildSIMULATeQCD/`) and **from there** call the following example script:
```shell
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="80" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=OFF \
-DBACKEND="hip_nvidia" \
```
Here, it is assumed that your source code folder is called `SIMULATeQCD`. **Do NOT compile your code in the source code folder!**
You can set the HIP installation path manually by setting the `cmake` parameter `-DHIP_PATH`.
You can also set the CUDA installation path manually by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR`.
`-DARCHITECTURE` sets the GPU architecture.
`-DUSE_GPU_P2P=ON` is not yet supported by this backend.

### Building source with HIP for AMD platforms (Experimental!)

In order to build the source with HIP for AMD platforms,
you need to make sure that
- HIP is properly installed on your machine
- The environment variable `HIP_PATH` holds the path to the HIP installation folder
- The environment variables `CC` and `CXX` hold the path to the HIP clang compiler

To setup the compilation, create a folder outside of the code directory (e.g. `../buildSIMULATeQCD/`) and **from there** call the following example script:
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

### Compiling particular executables
Inside the build folder, you can now begin to use `make` to compile your executables, e.g.
```shell
make NameOfExecutable
```
If you would like to speed up the compiling process, add the option `-j`, which will compile in parallel using all available CPU threads. You can also specify the number of threads manually using, for example, `-j 4`.

You also have the option to compile certain subsets of executables. For instance `make tests` will make all the executables used for testing.
One can also compile `applications`, `examples`, `profilers`, `tools`, and `everything`. To see a full list of available executables,
look at `SIMULATeQCD/CMakeLists.txt`. 

## Build (container) 

If you just want to get something running quickly on your laptop or desktop, this is likely the easiest way to go.
### Install Podman

#### On RHEL-based (Rocky/CentOS/RHEL) systems

Before continuing make sure there are no updates pending with `sudo dnf update -y && sudo dnf install -y podman` and then reboot with `sudo reboot`. (The reboot just makes avoiding permissions/kernel issues easy because that stuff is reread on boot.)

#### On Arch-based systems

See [install instructions](https://wiki.archlinux.org/title/Podman). If you have installed Arch before the upgrade to shadow (as in /etc/shadow) 4.11.1-3 rootless podman may encounter some issues. The build script will check for these anomalies and prompt you if you need to fix them.

#### Other \*NIX Systems

If you have a non RHEL-based OS see [here](https://podman.io/getting-started/installation.html#linux-distributions) for installation instructions.

### Make sure Podman works

Run `podman run hello-world` as your user to test your privileges. 
If this does not run correctly, the container build will not function. 

If you see the error:

```
ERRO[0014] cannot find UID/GID for user u6042105: No subuid ranges found for user "u6042105" in /etc/subuid - check rootless mode in man pages.
```

this indicates someone has modified the standard user privileges or you are running an older operating system. To fix this error run `sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 <YOUR_USER> && podman system migrate`

**WARNING**: If you are SSH'ing to your server, make sure you ssh as a user and **not** root. If you SSH as root and then `su` to user, podman will issue `ERRO[0000] XDG_RUNTIME_DIR directory "/run/user/0" is not owned by the current user`. This happens because the user that originally setup `/run` is root rather than your user.

### Build the code

1. Update `podman-build/config.yml` with any settings you would like to use for your build. This includes your target output directory.
   1. You can run `<where_you_downloaded>/simulate_qcd.sh list` to get a list of possible build targets.
   2. If you want to change where the code outputs to, you need to update OUTPUT_DIRECTORY in `podman-build/config.yml`. It will create a folder called build in the specified folder.
2. Run `chmod +x ./simulate_qcd.sh && ./simulate_qcd.sh build`

## How to run

### On a cluster using `slurm`

If you are on a cluster that uses slurm, then inside of your sbatch script do not use `mpiexec` or `mpirun`, but instead do
```shell
srun -n <NoGPUs> ./<program>
```
where `<NoGPUs>` is the number of GPUs you want to use.
The `<program>` likely already points to some default parameter file. If you would like to pass your
own parameters, one way to do this is to make your own parameter file. This can then be passed
as an argument like
```shell
srun -n <NoGPUs> ./<program> /path/to/your/parameter/file
```
Example parameter files can be found in the `parameter` folder. We have tried to explain
what each parameter means in each `.param` file. You can learn more about how general parameters
are implemented [here](../02_contributions/inputParameter.md).

### On your local machine (desktop, laptop, ...)

Unless you are using one GPU, any program has to be launched using `mpirun` or `mpiexec`.
For example:
```shell
mpiexec -np <NoGPUs> ./<program> 
```
Simiarly as above, if you want to pass your own parameter file,
```shell
mpiexec -np <NoGPUs> ./<program> /path/to/your/parameter/file
```
