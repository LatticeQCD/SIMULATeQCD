# SIMULATeQCD


[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://latticeqcd.github.io/SIMULATeQCD)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/LatticeQCD/SIMULATeQCD/commits/main)


*a SImple MUlti-GPU LATtice code for QCD calculations*


SIMULATeQCD is a multi-GPU Lattice QCD framework that makes it easy for physicists to implement lattice QCD formulas while still providing
competetive performance. 

## How to Build 

There are two possible ways to build SIMULATeQCD. If you are running on your own laptop or desktop and have an NVIDIA GPU,
we recommend that you use the [container build](#compile-using-container). The container will automatically grab all software you need.
If you are running on an HPC system or want to use AMD, we recommmend you [compile manually](#compile-manually) and ensure that all needed
software already exists on the system you're using.
This README attempts to give a succinct overview of how to build and use SIMULATeQCD. If you run into problems building, first
please have a look at the [Getting Started](https://latticeqcd.github.io/SIMULATeQCD/01_gettingStarted/gettingStarted.html) section of the **[documentation](https://latticeqcd.github.io/SIMULATeQCD)**.

### Download SIMULATeQCD

You will need to install [`git-lfs`](https://git-lfs.github.com/) before continuing or you will need to use a git client which natively supports it.
This is needed for downloading configurations used in the unit tests. Then run 
```shell
git clone https://github.com/LatticeQCD/SIMULATeQCD.git
```

### Compile Using Container

To build using the container, you need to have `podman` properly configured on your system.  
More information on that you may find in the [Getting Started](https://latticeqcd.github.io/SIMULATeQCD/01_gettingStarted/gettingStarted.html) section of the documentation.
To run the container you need an NVIDIA GPU.

To build the code, you then simply

1. Update [config.yml](./podman-build/config.yml) with any settings you would like to use for your build. This includes your target output directory.
   1. You can run `<where_you_downloaded>/simulate_qcd.sh list` to get a list of possible build targets.
   2. If you want to change where the code outputs to, you need to update OUTPUT_DIRECTORY in [config.yml](./podman-build/config.yml). It will create a folder called build in the specified folder.
2. Run `chmod +x ./simulate_qcd.sh && ./simulate_qcd.sh build`

### Compile Manually

The following software is required to manually compile SIMULATeQCD:

- `cmake` (Some versions have the "--phtread" compiler bug. Versions that definitely work are [3.14.6](https://gitlab.kitware.com/cmake/cmake/tree/v3.14.6) or 3.19.2.)
- `C++` compiler with `C++17` support.
- `MPI` (e.g. `openmpi-4.0.4`).
- `CUDA Toolkit` version 11+. 
- `pip install -r requirements.txt` to build the documentation.

To setup the compilation, create a folder outside of the code directory (e.g. `../buildSIMULATeQCD/`) and **from there** call the following example script: 
```shell
cmake ../SIMULATeQCD/ \
-DARCHITECTURE="80" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=ON \
```
Here, it is assumed that your source code folder is called `SIMULATeQCD`. 
You can set the path to CUDA by setting the `cmake` parameter `-DCUDA_TOOLKIT_ROOT_DIR:PATH`.
`-DARCHITECTURE` sets the GPU architecture (i.e. [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) version without the decimal point). For example "70" for Volta or "80" for Ampere.
Inside the build folder, you can now begin to use `make` to compile your executables, e.g.
```shell
make NameOfExecutable
```
If you would like to speed up the compiling process, add the option `-j`, which will compile in parallel using all available CPU threads. You can also specify the number of threads manually using, for example, `-j 4`.

Popular production-ready executables are:
```Shell
# generate HISQ configurations
rhmc                 # Example Parameter-file: parameter/applications/rhmc.param
# generate quenched gauge configurations using HB and OR
GenerateQuenched     # Example Parameter-file: parameter/applications/GenerateQuenched.param
# Apply Wilson/Zeuthen flow and measure various observables
gradientFlow         # Example Parameter-file: parameter/applications/gradientFlow.param
# Gauge fixing
gaugeFixing          # Example Parameter-file: parameter/applications/gaugeFixing.param
```
In the [documentation](https://latticeqcd.github.io/SIMULATeQCD/03_applications/applications.html) you will find more information on how to execute these programs.

## Example: Plaquette action computation

Here we showcase a snippet of code. It is not important that you understand all details: We just
want to emphasize that these two blocks are roughly all that is required to compute the plaquette
at every site for every orientation. These blocks take care of periodic BCs, GPU parallelization,
and communication between neighboring GPUs behind the scenes.
See this [Full code example](https://github.com/LatticeQCD/SIMULATeQCD/blob/main/src/examples/main_plaquette.cu)
for a more detailed understanding.

```C++
template<class floatT, bool onDevice, size_t HaloDepth>
struct CalcPlaq {
  gaugeAccessor<floatT> gaugeAccessor;
  CalcPlaq(Gaugefield<floatT,onDevice,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){}
  __device__ __host__ floatT operator()(gSite site) {
    floatT result = 0;
    for (int nu = 1; nu < 4; nu++) {
      for (int mu = 0; mu < nu; mu++) {
        result += tr_d(gaugeAccessor.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu)));
      }
    }
    return result;
  }
};

(... main ...)
gauge.updateAll()
latticeContainer.template iterateOverBulk<All, HaloDepth>(CalcPlaq<floatT, HaloDepth>(gauge))
latticeContainer.reduce(plaq, elems);
```


## Documentation

Please check out [the documentation](https://latticeqcd.github.io/SIMULATeQCD) to learn how to use SIMULATeQCD in detail,
including how to make contributions, details for installation, and to see what kinds of modules and applications are
already available.

## Getting help and bug report
Open an [issue](https://github.com/LatticeQCD/SIMULATeQCD/issues), if...
- you have troubles compiling/running the code.
- you have questions on how to implement your own routine.
- you have found a bug.
- you have a feature request.

If none of the above cases apply, you may also send an email to lukas.mazur(at)uni-paderborn(dot)de
or clarke(dot)davida(at)gmail.com.


## Contributors

[L. Mazur](https://github.com/lukas-mazur), 
[S. Ali](https://github.com/Sajidali1031), 
[L. Altenkort](https://github.com/luhuhis), 
[D. Bollweg](https://github.com/dbollweg), 
[D. A. Clarke](https://github.com/clarkedavida), 
[G. Curell](https://github.com/grantcurell/),
[H. Dick](https://github.com/redweasel),
[J. Goswami](https://github.com/jishnuxx),
[D. Hoying](https://github.com/goracle),
[O. Kaczmarek](https://github.com/olaf-kaczmarek),
[R. Larsen](https://github.com/RasmusNL),
[M. Neumann](https://github.com/mneumann177),
[M. Rodekamp](https://github.com/Marcel-Rodekamp), 
[H. Sandmeyer](https://github.com/hsandmeyer), 
[C. Schmidt](https://github.com/schmidt74), 
[P. Scior](https://github.com/philomat), 
[H.-T. Shu](https://github.com/haitaoshu)

## Citing SIMULATeQCD

If you are using this code in your research please cite:

- *L. Mazur, Topological aspects in lattice QCD, Ph.D. thesis, Bielefeld University (2021), [https://doi.org/10.4119/unibi/2956493](https://doi.org/10.4119/unibi/2956493)*
- *L. Altenkort, D.Bollweg, D. A. Clarke, O. Kaczmarek, L. Mazur, C. Schmidt, P. Scior, H.-T. Shu, HotQCD on Multi-GPU Systems, PoS LATTICE2021, Bielefeld University (2021), [https://arxiv.org/abs/2111.10354](https://arxiv.org/abs/2111.10354)*

## Acknowledgments

- We acknowledge support by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the CRC-TR 211
'Strong-interaction matter under extreme conditions'– project number 315477589 – TRR 211.
- This work was partly performed in the framework of the PUNCH4NFDI consortium supported by DFG fund "NFDI 39/1", Germany.
- This work is also supported by the U.S. Department of Energy, Office of Science, though the Scientific Discovery through Advance
- We would also like to acknowedge enlightening technical discussions with the ILDG team, in particular H. Simma.

