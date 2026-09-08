# Testing the HISQforceoptimize branch

This is a pre-merge validation workflow. Develop and review the optimization on
`HISQforceoptimize`; merge into `main` only after collaborators review correctness
and timing results. The current branch includes recursive **force and smearing**,
so its RHMC speedup measures their combined effect.

## Branch and review

The author creates the branch in the current working tree:

```bash
git switch -c HISQforceoptimize
git status --short
```

Review and commit the optimization, benchmark drivers, scripts, parameters and
this guide, then push the branch to your collaboration remote. There are unrelated
local Wilson-line changes and backup files in the author's working tree: do not
use `git add .` to prepare this change.

Collaborators check out the published branch:

```bash
git fetch origin
git switch --track origin/HISQforceoptimize
```

If already checked out locally, use `git switch HISQforceoptimize`.
Record `git rev-parse HEAD` and any local changes with your results. The baseline
below is the frozen commit `1732861`, not a moving `main`.

## One-time setup

Use this layout (the parent directory can be anywhere):

```text
project/
  SIMULATeQCD/                    # HISQforceoptimize checkout
  SIMULATeQCD-legacy-1732861/      # separate baseline worktree
  buildSIMULATeQCD/
  buildSIMULATeQCD_legacy_1732861/
  parameter/                     # common benchmark parameters and rational file
  test_conf/
    l528f21b6315m00282m0759_001.1610
  testrun/                       # submit all jobs here
```

Obtain the large thermalized gauge file from the collaboration; it is not bundled
with the benchmark. From `SIMULATeQCD/`, create the baseline **once** in a fresh
workspace:

```bash
git worktree add --detach ../SIMULATeQCD-legacy-1732861 1732861
cp src/testing/main_rhmcBenchmark.cpp ../SIMULATeQCD-legacy-1732861/src/testing/main_rhmcBenchmark.cpp
git -C ../SIMULATeQCD-legacy-1732861 apply --check ../SIMULATeQCD/scripts/rhmcBenchmarkLegacy.patch
git -C ../SIMULATeQCD-legacy-1732861 apply ../SIMULATeQCD/scripts/rhmcBenchmarkLegacy.patch
```

The baseline patch only adds the benchmark target, CUDA 12.8 NVTX compatibility,
and the same Hamiltonian/reversibility diagnostics already present on the branch.
It does not port recursive force or smearing. Do not apply it again to an already
prepared worktree. For the author's existing setup, just refresh the common driver:

```bash
cp src/testing/main_rhmcBenchmark.cpp ../SIMULATeQCD-legacy-1732861/src/testing/main_rhmcBenchmark.cpp
```

For a **new** project directory, populate the common inputs:

```bash
mkdir -p ../parameter/tests ../test_conf ../testrun
cp parameter/tests/rhmcBenchmark.param ../parameter/tests/
cp parameter/sample_force_test.rat ../parameter/
```

Existing setups can keep their common inputs; check them against these versioned
files. Both binaries must read the same parameters and rational coefficients.

## Build both versions

On the current V100 cluster, from `SIMULATeQCD/`:

```bash
module load compilers/gnu/12.2.1 cuda/12.8 mpi/openmpi/ompi-cuda-5.0.7

cmake -S . -B ../buildSIMULATeQCD \
  -DCMAKE_BUILD_TYPE=Release -DBACKEND=cuda -DARCHITECTURE=70 \
  -DUSE_GPU_AWARE_MPI=ON -DUSE_GPU_P2P=ON

cmake -S ../SIMULATeQCD-legacy-1732861 -B ../buildSIMULATeQCD_legacy_1732861 \
  -DCMAKE_BUILD_TYPE=Release -DBACKEND=cuda -DARCHITECTURE=70 \
  -DUSE_GPU_AWARE_MPI=ON -DUSE_GPU_P2P=ON

cmake --build ../buildSIMULATeQCD --target rhmcBenchmark rhmcTrajectoryCompare --parallel 8
cmake --build ../buildSIMULATeQCD_legacy_1732861 --target rhmcBenchmark --parallel 8
```

Use the same compiler/CUDA/MPI and CMake options for both. Never reconfigure the
recursive build directory with the legacy source. Preserve both CMake caches with
the report. The common driver prints seconds explicitly:
`RHMC trajectory time: 303.603s`. Rebuild **both** if an old binary prints minutes.

## One-command collaborator check

After building, submit from `testrun/`:

```bash
sbatch ../SIMULATeQCD/scripts/runRhmcTimingComparison.sh collaborator
```

This runs the historical small smearing and force tests, the new recursive
smearing test, a complete large-field force comparison, and a one-step RHMC
gauge-field comparison. Every check has a named PASS/FAIL result and failures
stop the job. Then it measures one complete legacy trajectory and one complete
recursive trajectory and reports the measured percentage reduction.

A successful report ends with:

```text
Collaborator correctness checks: PASS
Legacy RHMC time:    <measured>s
Recursive RHMC time: <measured>s
RHMC speedup: <measured>x
RHMC trajectory time reduction: <measured>% (single timing pair)
Collaborator check: PASS
```

No target gain (such as 15%) is hardcoded, and a slowdown is reported honestly.
This checks the specified regression cases; it is not a claim that every RHMC
property has been proved. Full reversibility/energy scaling review remains part
of the author workflow below. A single timing pair is a quick local confirmation,
not the number to quote as a precise performance result.

The historical `hisqSmearingTest` and `hisqForce` drivers retain their commit
1732861 logic, inputs, tolerances and normalization, with only their verdicts
renamed PASS/FAIL. The new `hisqSmearingRecursiveTest` is separate and runs
correctness checks by default; `--benchmark` enables its original smearing
timing loops. Large force dumps live in the job's result directory.

In addition to the RHMC targets, build the test targets from `SIMULATeQCD/`:

```bash
cmake --build ../buildSIMULATeQCD \
  --target hisqSmearingTest hisqSmearingRecursiveTest hisqForce hisqForceBenchmark hisqForceLargeCompare \
  --parallel 8
cmake --build ../buildSIMULATeQCD_legacy_1732861 --target hisqForceBenchmark --parallel 8
```

If preparing a fresh baseline without the force benchmark target:

```bash
cp src/testing/main_hisqForceBenchmark.cpp ../SIMULATeQCD-legacy-1732861/src/testing/
git -C ../SIMULATeQCD-legacy-1732861 apply --include=CMakeLists.txt ../SIMULATeQCD/scripts/hisqForceBenchmarkTarget.patch
```

Reconfigure the legacy build with the same options above before building the
new target. Its NVTX compatibility patch is already supplied by the RHMC setup.

For a new common input directory, copy the versioned test parameters:

```bash
cp parameter/tests/hisqForce.param parameter/tests/hisqSmearingTest.param \
   parameter/tests/hisqForce_bench.param parameter/tests/hisqSmearingRecursiveTest.param \
   ../parameter/tests/
```

Also provide the historical `gauge12750`, `force_reference`, and
`smearing_reference_conf` fixtures under `../test_conf/` (real data, not Git LFS
pointer files). Do not regenerate or replace those references. The collaborator
runner explicitly fixes the historical tests to an 8x8x8x4 lattice, even if your
old local smearing parameter file was changed for a large benchmark.

## Author correctness checks before timing

Submit each check separately from `testrun/`, wait for it, and inspect its logs
before continuing. Each job requests 20 minutes, below the 30-minute devel limit.

```bash
cd ../testrun
sbatch ../SIMULATeQCD/scripts/runRhmcTimingComparison.sh short
# After reviewing that result:
sbatch ../SIMULATeQCD/scripts/runRhmcTimingComparison.sh reverse
# After reviewing reversibility:
sbatch ../SIMULATeQCD/scripts/runRhmcTimingComparison.sh scaling
# Finally inspect one complete trajectory per implementation:
sbatch ../SIMULATeQCD/scripts/runRhmcTimingComparison.sh validate
```

- `short`: one MD step, full final gauge-field comparison at `1e-6`.
- `reverse`: one step forward/backward, full returned gauge-field comparison,
  and maximum gauge/momentum reversibility violations.
- `scaling`: fixed trajectory length 0.02 with step sizes 0.02, 0.01, 0.005;
  inspect the Hamiltonian errors for both implementations.
- `validate`: complete trajectories, initial/final observables, Hamiltonian
  diagnostics, and acceptance status.

Each process reloads the same gauge configuration and seeds the RNG again.
These checks use the production float RHMC driver. Same seed does not prove
bitwise-identical pseudofermions after changed smearing arithmetic; examine the
initial diagnostics as well as the final results. Observable or Delta-H agreement
alone is not a correctness proof. Scaling and validation print diagnostics for
review; exit status zero is not an automatic physics-validation verdict.
Retain the existing small historical regressions and full-force comparison in the
review evidence, and rerun smearing/force correctness after the smearing change.

Full logs and gauge dumps are kept under `rhmc_results/JOB_ID/`.
The displayed path in each Slurm output identifies the directory. No measurement
or gauge dump was added to the region timed around `HMC.update()`.

## Author timing: three repetitions with one submission

After correctness review:

```bash
sbatch --array=1-3%1 ../SIMULATeQCD/scripts/runRhmcTimingComparison.sh timing
```

Slurm prints an array job ID, for example `50100`. Only one array task runs at a
time; each task runs one legacy/recursive pair and has its own 20-minute limit.
Execution order is legacy/recursive, recursive/legacy, legacy/recursive.
There are no `afterok` dependencies between repetitions. A failure in one task
does not prevent the others from running; the report still requires all three.

When all three tasks complete successfully:

```bash
bash ../SIMULATeQCD/scripts/summarizeRhmcTiming.sh rhmc_results/50100
```

Replace `50100` with your actual array ID. This prints all six internal times,
both medians, speedup, percent time reduction, timing ranges and each pair's gain. Scheduler job duration is never
used. Different arrays use different directories; do not rebuild binaries or
change input files during an array. If the paired gains vary materially, collect
additional independent arrays before quoting a stable percentage; three samples
do not establish a confidence interval.

If the site disallows arrays, submit `timing 1`, `timing 2`, `timing 3` one at a
time and export the same absolute `RHMC_RESULTS_DIR` for that set. Use a new
directory for every new comparison.

## Other sites or directory names

Override partition/time on the `sbatch` command line. The runner accepts
`RHMC_LEGACY_BINARY`, `RHMC_RECURSIVE_BINARY`, and `RHMC_COMPARE_BINARY` as absolute
paths. The comparator otherwise lives beside the recursive binary.
Set `RHMC_MODULES` to the site's space-separated module names, or to an empty
string when submitting with an already configured environment. Submit from the
run directory so the common `../parameter` and `../test_conf` paths resolve.
Use one MPI rank and one V100 for the controlled comparison.

## Recovering a broken dependency chain

Inspect the prerequisite job's output and error files first. A completed
trajectory can still leave a failed shell job if output parsing fails. The old
timer printed minutes after two minutes and the old parser rejected that format.
Do not recover timing values by converting that string: its minute count is
rounded. Use the new seconds record after rebuilding both drivers.

Cancel only the obsolete dependent jobs (for example `scancel 49833 49834`),
then resubmit with the array command after fixing the underlying failure.

## Report for pre-merge review

Attach the branch commit and local diff, baseline commit/benchmark patch, build
options and toolchain, input and rational-file checksums, GPU model, correctness
logs, and the median timing report. Report force-only timings separately from
RHMC timings with force plus smearing. Publish this evidence on the branch's
pull/merge request, obtain collaborator review, then merge into `main`.
