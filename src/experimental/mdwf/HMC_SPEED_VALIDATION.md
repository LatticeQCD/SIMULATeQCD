# MDWF HMC speed-ups: stack-grid launch, action from the noise, chrono action solve: validation

This note covers three changes on branch `mdwf-hmcspeed-final`, based on `mdwf-clover-kernel` c287dfe:

| commit | change | switch | default |
|---|---|---|---|
| 631744d | start-of-trajectory action from the noise; chrono initial guess for the end-of-trajectory light action solve | `action_from_noise`, `action_chrono` (parameter file) | off |
| 955c950, fcf6a3a | 5D operator kernels, mixed-CG vector updates and the 5D dot product launched with Ls stacked into the grid instead of 24/48-thread blocks | `MDWF_STACK_GRID` (environment) | on (`MDWF_STACK_GRID=0` gives the old launch) |

None of them changes the Markov chain. The stack-grid launch is bit-identical to the old launch. With the two
action switches on, the molecular dynamics and the random numbers stay the same. Only the start- and end-of-trajectory
action solves change, and they move ΔH by much less than the solver tolerance.

## What the action switches do

- `action_from_noise = 1`: for every pseudofermion factor whose heatbath makes S = |η|² exact, H at the start of a
  trajectory takes S = |η|² instead of a solve. These factors are the light two-flavour ratio factors (both
  `light_form`s) and the DSDR ratios, at c_sw = 0. EOFA and the strange RHMC still solve, because their heatbath is
  rational.
- `action_chrono = 1`: the end-of-trajectory light action solve starts from the factor's chronological force-solve
  history. The history is only read, never written, so the MD is unchanged. Needs `chrono_size > 0`.

The two only change the **rounding of S** at the trajectory end points, at the level of the action solver
tolerance (`action_error`). The pass rule below therefore asks for an unbiased ΔH change far below 1 % of an
acceptance probability, and identical MD.

## Tests

### 1. Small volumes (PC2, H100)

- 8^4, Ls 24 and Ls 16, b1633 action: the stack-grid launch gives an **identical** ΔH (−0.0738335736096 at Ls 24,
  0.0196197545156 at Ls 16). Ls 24 time per trajectory: 22.7 s → 12.1 s.
- 16^3 × 8, 4 ranks: identical with the stack grid. With both action switches on: the same accepts, identical plaquettes,
  |ΔΔH| ≤ 2.5e-6.
- 64^3 × 8, 16 H100, real b1633_ml40 configuration, Ls 24: time per CG iteration 15.38 ms → 12.51 ms (−19 %).

### 2. Production regression (JUPITER, GH200, 32^3 × 8, 1 node = 4 GPUs, Nodes 1 2 2 1)

One trajectory from a production configuration with the production parameter file and seed, compared with the row the
old binary (c287dfe) wrote into `hmc.dat`. Columns: traj, accept, ΔH, exp(−ΔH), plaquette, kinetic, gauge action,
fermion action, CG iterations, s/traj.

| run | ensemble | switches | ΔH | rest of the row | s/traj |
|---|---|---|---|---|---|
| reference (c287dfe) | b1633_ml40, Ls 24, traj 26 | | 0.362644553185 | | 4704.6 |
| `MDWF_STACK_GRID=0` | same | | 0.362644553185 | identical | 4704.4 |
| `MDWF_STACK_GRID=1` | same | | 0.362644553185 | identical | 2023.0 (−57 %) |
| stack grid + both action switches | same | `action_from_noise=1 action_chrono=1` | 0.362520039082 | plaquette, kinetic, gauge identical; fermion action differs in the 12th digit | 1795.0 (−62 %) |
| reference (2707fd4) | b1671_ml80, Ls 16, traj 289 | | −0.977638334036 | | 1958.4 |
| `MDWF_STACK_GRID=0` | same | | −0.977638334036 | identical | 2132.5 (+8.9 %) |
| `MDWF_STACK_GRID=1` | same | | −0.977638334036 | identical | 1440.2 (−26.5 %) |

Operator benchmark of the same job: every launch configuration gives |Mx_new − Mx_old|² = 0 in double and single
precision. Ls 16 in single precision: 1.125 → 0.587 ms per M̂†M̂ application with block 64.

### 3. Paired long test of the action switches (JUPITER, b1633_ml40, 2026-11-14)

**Design.** Six production streams of b1633_ml40 (32^3 × 8, Ls 24, β = 1.633) were restarted from exactly the
configuration and seed that the production job of the same day restarted from. The paired run used the production
parameter file plus `action_from_noise=1 action_chrono=1`, and ran 6 h on one node. The MD is identical, so as long as
every accept decision agrees, the two chains stay bit-identical. Each trajectory then gives one **paired** sample
ΔΔH = ΔH(switches) − ΔH(production) at the same gauge field.

**Pass rule** (fixed before the run):

1. Every paired trajectory has the same accept flag, plaquette, kinetic energy and gauge action before the update
   (MD and RNG unchanged).
2. max |ΔΔH| < 1e-2, so no acceptance probability changes by more than 1 %.
3. Mean ΔΔH is consistent with 0 within 3σ, and |mean ΔΔH| < 1e-3 (no bias in H).
4. No FATAL error and no CG failure.
5. Time per trajectory is at least 5 % below the paired production trajectories.

**Result.**

| stream | paired traj | identity breaks | mean ΔΔH | max \|ΔΔH\| | s/traj switches | s/traj production | change |
|---|---|---|---|---|---|---|---|
| 002 | 1 | 0 | −5.35e-05 | 5.35e-05 | 2018 | 2279 | −11.4 % |
| 005 | 10 | 0 | −4.21e-05 | 1.45e-04 | 1923 | 2179 | −11.7 % |
| 009 | 10 | 0 | −1.26e-05 | 1.35e-04 | 1918 | 2134 | −10.1 % |
| 013 | 10 | 0 | −4.10e-05 | 1.19e-04 | 1894 | 2130 | −11.1 % |
| 016 | 10 | 0 | −3.32e-05 | 1.10e-04 | 1906 | 2195 | −13.2 % |
| 020 | 9 | 0 | −8.09e-06 | 1.27e-04 | 1925 | 2162 | −11.0 % |
| **pooled** | **50** | **0** | **−2.83e-05 ± 0.97e-05** | **1.45e-04** | | | **−11.4 %** |

Verdict: **PASS** on all five criteria. Stream 002 has only one pair because the production node failed at start-up
and the job was requeued, so its production trajectories came later.

- The mean is 2.9σ from zero. That is within the rule, but it is a real ~3e-5 shift, two orders of magnitude below
  criterion 3 and far below the statistical error of ⟨exp(−ΔH)⟩.
- Its likely origin: S = |η|² is exact, while the solved start action carries the solver error. A residual of order
  `action_error` biases S in one direction.

**Production.** From 2026-11-14 the JUPITER Ls 24 production (b1633) runs with both switches on. Ls 16 stays off until
the switches are tested at Ls 16.

## How to repeat the paired test

1. Pick N streams. For each one, take the configuration and seed that the next production segment will start from.
   Submit the paired job together with the production round, before production writes newer configurations.
2. Run the same binary with the production parameter file and
   `Gaugefile=<start> traj_offset=<traj> seed=<production seed> action_from_noise=1 action_chrono=1`.
   Write into a separate `measurements_dir` and delete its configurations afterwards.
3. Pair the rows of the two `hmc.dat` files by trajectory number:
   - Columns 2 (accept), 5 (plaquette), 6 (kinetic) and 7 (gauge) must agree exactly.
   - Accumulate d = column 3 (ΔH) of the test minus the production value, then report mean ± sd/√n and max |d|.
   - Compare the last column (s/traj).

The JUPITER scripts are `SQCDruns/hmcspeed_test/ACTIONOPT.sh` (`submit`, `status`) and `job_actionopt.sbatch`. They are
not part of the repository.
