# Möbius domain-wall fermions (MDWF)

Möbius domain-wall fermions with a Wilson(-clover) kernel D_W(-M5), c5 = b5 - 1, in Grid/GPT conventions:

- operator: 5D Möbius operator, even/odd Schur complement Mhat = M_ee - M_eo M_oo^-1 M_oe (fused kernels at
  c_sw = 0), optional smeared kernel links (fat7 + SU(3) or U(3) projection)
- solvers: CG on Mhat^+ Mhat or Mhat Mhat^+, mixed precision (single-precision CG with double reliable updates),
  multishift CG, thick-restart Lanczos
- 2+1 flavour HMC: Pauli-Villars light pair with Hasenbusch masses, strange quark by RHMC or exact one-flavour
  algorithm (EOFA), DSDR weight, Wilson / Symanzik / Iwasaki gauge action, multi-level leapfrog / OMF2 /
  force-gradient integrators, antiperiodic fermions in time
- measurements: residual mass, screening correlators, chiral condensates, spectra of the kernel and of the 4D
  effective Dirac operator

The code is in this directory (headers only, plus `MDWFRemez.cpp`), the programs and tests in
`src/testing/main_mdwf*.cpp`, their parameter files in `parameter/tests/mdwf*.param`.

## Build

As in the top-level README; in addition the RHMC/EOFA targets need GMP and MPFR (found in the system paths,
`CMAKE_PREFIX_PATH`, `-DGMP_ROOT=<prefix>` / `-DMPFR_ROOT=<prefix>` or the EasyBuild modules; without them CMake
prints which file is missing and skips only those targets). The test configurations in `test_conf/` come with
`git lfs pull`.

```shell
cmake ../SIMULATeQCD -DARCHITECTURE=80 -DUSE_GPU_AWARE_MPI=ON -DUSE_GPU_P2P=OFF
make -j 8 mdwf          # the programs below
make -j 8 mdwftests     # the tests below
make mdwfResidualMass   # or single targets
```

Each target takes a few minutes to compile, so build only what you need. All executables are written to
`<build>/testing`.

## Run

```shell
cd <build>/testing
./mdwfResidualMass                                    # reads ../parameter/tests/mdwfResidualMass.param
./mdwfResidualMass my.param Gaugefile=/path/conf      # own parameter file, keys overridden on the command line
mpirun -np 2 ./mdwfTwoPlusOneHmcRun my.param          # several ranks: set Nodes (programs marked "ranks")
```

Every parameter file starts with a comment block that explains its keys. The shipped files are small examples
for one GPU (the two benchmarks use a 32^3 x 8 lattice).

## Programs (`make mdwf`)

| program | what it does | needs |
|---|---|---|
| `mdwfTwoPlusOneHmcRun` | 2+1 flavour HMC; the example is the DSDR setup: Iwasaki, Hasenbusch, EOFA strange, even/odd, mixed precision | Remez; ranks at c_sw = 0 |
| `mdwfResidualMass` | residual mass from 12 point-source solves, correlators along x, y, z, t; jobs can be split by spin-colour component | |
| `mdwfMeasure` | m_res, screening correlators and stochastic condensates in the GPT output format | |
| `mdwfSpectralBounds` | Lanczos estimate of the extreme eigenvalues of M^+ M, to choose the RHMC intervals | |
| `mdwfKernelSpectrum` | lowest and highest eigenvalue of H_W^2 against M5 (spectral gap of the kernel), thin or smeared links | configuration |
| `mdwfDiracModes` | low modes of H_W^2 or of the 4D effective operator D(m)^+ D(m): IPR, chirality, Ginsparg-Wilson defect | configuration |
| `mdwfOperatorBench` | time per Mhat^+ Mhat application, fused and block path, double and single precision | |
| `mdwfProfileBench` | time of the even/odd operator pieces and of the light-quark CG solves | ranks |

`scripts/mdwf/analyze_mres.py` analyses the `mdwfResidualMass` output of many configurations (jackknife).
`parameter/tests/mdwfResidualMassGridReference.param` is a regression of `mdwfResidualMass` against Grid/GPT
on a 32^3 x 8 configuration that is not part of the repository.

## Tests (`make mdwftests`)

Tests take no arguments (lattice from `parameter/tests/mdwfFifthDimTest.param`, 6^4, unless noted), run on one
GPU, print what they compare, and exit with 0 when they pass. All of them:

```shell
cd <build>/testing && bash ../scripts/mdwf/run_mdwf_tests.sh
```

| test | checks |
|---|---|
| **operator** | |
| `mdwfWilsonKernelNormalizationTest` | the kernel is D_W(-M5) in the standard normalization (plane waves on the unit field) |
| `mdwfCloverAbelianFluxTest` | sign and normalization of the clover term against an analytic abelian field |
| `mdwfMobiusCloverMappingTest` | Möbius operator with clover kernel: c_sw = 0 and b5 = 1 (Shamir) limits, adjoint identity |
| `mdwfMobiusAdjointMappingTest` | adjoint Möbius operator: b5 = 1 limit and <x, M y> = <M^+ x, y> |
| `mdwfMobiusEvenOddTest` | even/odd decomposition and Schur-complement solve against the full operator (c_sw = 0 and 0.5) |
| `mdwfKernelSmearingTest` | smeared kernel links: unit field, unitarity, gauge covariance |
| `mdwfEffectiveOperatorTest` | 4D effective operator in the free field and thick-restart Lanczos |
| **solvers** | |
| `mdwfMobiusNormalSolveTest` | CG on M^+ M |
| `mdwfSimultaneousMultiShiftCGTest` | multishift CG: exact solutions and agreement with one CG per shift |
| `mdwfRemezTest` | rational approximations (AlgRemez): reference coefficients, error, heatbath/action consistency (host only, Remez) |
| **actions and forces** | |
| `mdwfMobiusRationalActionTest` | rational action phi^+ R(M^+ M) phi against single-shift solves, b5 = 1 limit, heatbath |
| `mdwfMobiusFiniteDifferenceTest` | action derivative by finite differences |
| `mdwfMobiusForceContractionTest` | analytic force at selected links against the finite difference |
| `mdwfMobiusAllLinkForceStorageTest` | force on all links (host) against the selected-link force |
| `mdwfDeviceForceStorageTest` | force on all links on the GPU against the host version |
| `mdwfPauliVillarsActionTest` | two-flavour Pauli-Villars ratio: heatbath, force against dS, cancellation at mf = pv_mass |
| `mdwfOneFlavorRhmcActionTest` | one-flavour RHMC action and force (Remez) |
| `mdwfEvenOddActionTest` | even/odd Pauli-Villars and RHMC actions and the log det M_oo term: heatbath, forces (Remez) |
| **HMC** | |
| `mdwfMobiusHmcConventionTest` | MD conventions: gauge and fermion forces match dS along U -> exp(i eps P) U |
| `mdwfSymanzikGaugeForceTest` | Symanzik gauge force against finite differences |
| `mdwfHmcTrajectoryTest` | two-flavour HMC: reversibility, Delta H ~ eps^2 |
| `mdwfPauliVillarsHmcTrajectoryTest` | Pauli-Villars two-flavour HMC: reversibility, Delta H ~ eps^2 |
| `mdwfPauliVillarsMultiscaleTest` | two-scale (Sexton-Weingarten) integrator: reversibility, Delta H ~ eps^2 |
| `mdwfTwoPlusOneHmcTrajectoryTest` | 2+1 HMC with RHMC strange: reversibility, Delta H ~ eps^2 (Remez) |
| `mdwfEvenOddTwoPlusOneTrajectoryTest` | even/odd 2+1 HMC: reversibility, Delta H (Remez) |
| `mdwfAntiperiodicHmcTest` | antiperiodic fermions in time in the even/odd 2+1 HMC, force by finite differences (Remez) |
| `mdwfHasenbuschTest` | Hasenbusch ladder: no mass gives the single ratio, three-factor ladder, multi-level integrator (Remez) |
| `mdwfDSDRTest` | DSDR ratio and Iwasaki gauge action (`mode = unit`, 8^4); `mode = eofa`: EOFA tests, needs a configuration (Remez) |

## Notes

- `mdwfTwoPlusOneHmcRun`: the light ratio uses GPT's form (`light_form = 1`) by default, which needs c_sw = 0;
  `Gaugefile_out` is written into `measurements_dir`.
- Optional environment variables: `MDWF_REMEZ_CACHE=<dir>` keeps the Remez approximations on disk,
  `SIMULATEQCD_LEAN_HALO=1` selects the lean halo exchange (GPU-aware MPI without P2P).
- The `*.md` files in this directory are the design notes (conventions, even/odd scheme, determinant
  factorization, force checks).
