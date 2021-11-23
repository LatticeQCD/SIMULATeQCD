# SIMULATeQCD


[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://latticeqcd.github.io/SIMULATeQCD)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/LatticeQCD/SIMULATeQCD/commits/main)


*a SImple MUlti-GPU LATtice code for QCD calculations*


SIMULATeQCD is a multi-GPU Lattice QCD framework that makes it simple and easy for physicists to implement lattice QCD formulas while still providing the best possible performance.


## Prerequisites

The following software is required to compile SIMULATeQCD:

1. [git-lfs](https://git-lfs.github.com/) to also be able to clone test configurations:
    ```shell
    # For Debian-based system
    sudo apt install git-lfs
     
    # For Arch-based system
    sudo pacman -S git-lfs
    ```
    and activate it by calling `git lfs install`
2. `cmake` (Some versions have the "--phtread" compiler bug. Versions that definitely work are [3.14.6](https://gitlab.kitware.com/cmake/cmake/tree/v3.14.6) or 3.19.2.)
3. `C++` compiler with `C++17` support  (e.g. `g++-9`).
4. `MPI` (e.g. `openmpi-4.0.4`).
5. `CUDA Toolkit` version 11.0 (NOT 11.1 or 11.2).
6. `pip install -r requirements.txt` to build the documentation.

## Downloading the code

First, make sure you have activated git-lfs using `git lfs install`, as mentioned above. 
The code can then be cloned to your folder using:
```shell
git clone https://github.com/LatticeQCD/SIMULATeQCD.git
```


## Building the code

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

## Example: Plaquette action computation

(See [Full code example](https://github.com/LatticeQCD/SIMULATeQCD/blob/main/src/examples/main_plaquette.cu).)

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
latticeContainer.template iterateOverBulk<All, HaloDepth>(CalcPlaq<floatT, HaloDepth>(gauge))
```


## Documentation

Please check out [the documentation](https://latticeqcd.github.io/SIMULATeQCD) to learn how to use SIMULATeQCD.

## Getting help and bug report
Open an [issue](https://github.com/LatticeQCD/SIMULATeQCD/issues), if...
- you have troubles compiling/running the code.
- you have questions on how to implement your own routine.
- you have found a bug.
- you have a feature request.

If none of the above cases apply, you may also send an email to lukas.mazur(at)uni-paderborn(dot)de.


## Main contributors

[L. Mazur](https://github.com/lukas-mazur), [H. Sandmeyer](https://github.com/hsandmeyer), [D. Bollweg](https://github.com/dbollweg), [D. Clarke](https://github.com/clarkedavida), [L. Altenkort](https://github.com/luhuhis), [P. Scior](https://github.com/philomat), H.-T. Shu, R. Larsen, M. Rodekamp, [O. Kaczmarek](https://github.com/olaf-kaczmarek), [C. Schmidt](https://github.com/schmidt74)


## Citing SIMULATeQCD

If you are using this code in your research please cite:

- *L. Mazur, Topological aspects in lattice QCD, Ph.D. thesis, Bielefeld University (2021), [https://doi.org/10.4119/unibi/2956493](https://doi.org/10.4119/unibi/2956493)*
- *L. Altenkort, D.Bollweg, D. A. Clarke, O. Kaczmarek, L. Mazur, C. Schmidt, P. Scior, H.-T. Shu, HotQCD on Multi-GPU Systems, PoS LATTICE2021, Bielefeld University (2021), [https://arxiv.org/abs/2111.10354](https://arxiv.org/abs/2111.10354)*

## Acknowledgment

- We acknowledge support by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the CRC-TR 211 'Strong-interaction matter under extreme conditions'– project number 315477589 – TRR 211.
- This work was partly performed in the framework of the PUNCH4NFDI consortium supported by DFG fund "NFDI 39/1", Germany.
- This work is also supported by the U.S. Department of Energy, Office of Science, though the Scientific Discovery through Advance Computing (SciDAC) award 
Computing the Properties of Matter with Leadership Computing Resources. 
