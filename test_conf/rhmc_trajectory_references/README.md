# Frozen legacy RHMC trajectory references

These eight fixtures belong to the existing `rhmcTest`, not a new executable.
`scripts/runTests_2.bash` runs the checks after the historical RHMC tests.
Collaborators need only their optimized build and the normal test inputs.
Failures produce named FAIL output and a nonzero executable exit status.
Missing or incompatible fixtures are failures, not silently skipped tests.

## Provenance

- Legacy force/smearing baseline: commit `1732861` with the common reference
  test driver.
- Starting field: `test_conf/gauge12750`, lattice 8x8x8x4, one MPI rank.
- Input settings: `parameter/tests/rhmcTrajectoryReference.param`.
- Float and double, mu0=0/0.4 (imaginary chemical potential), no_md=1/10.
- Every case restarts from the same input field and trajectory RNG seed 1337.
- Metropolis disabled: the fixture is the evolved field, not a rejected-field
  rollback. `cgMixedPrec=0`; float is not the optional mixed-precision solver.
- CHI_UD uses ten sources from a fresh independent RNG seed 424242.
- Plaquette/Polyakov/CHI_UD measurements and NERSC writes are untimed.
- Three-row, 64-bit NERSC field files; `.txt` stores the case identity,
  plaquette, real/imaginary Polyakov, ten CHI_UD samples and informational time.
- References are copied unchanged from the legacy implementation; the
  optimized implementation never generates its own oracle.

Input SHA256:

```text
41bfba2281d9360b7820f5acdedaa5bb6645995bcdda17f0301f6d9461043c0d  test_conf/gauge12750
628cf3bfb79d0a2b0025d30df09a4a6ae46a2cf06af5f88d190903d2d3a5148e  parameter/sample_force_test.rat
085a9b87bda0798c31cb21feef0c0da4c827215ad8a1d60c27b8945d8b375728  parameter/tests/rhmcTrajectoryReference.param
```

## Numerical-regression policy

| Arithmetic | MD steps | Field tolerance | Plaquette/Polyakov absolute tolerance | Each CHI_UD absolute tolerance |
|---|---:|---:|---:|---:|
| float | 1 | 1e-6 | 1e-7 | 1e-5 |
| float | 10 | 1e-5 | 1e-7 | 1e-5 |
| double | 1 | 1e-10 | 1e-11 | 1e-5 |
| double | 10 | 1e-9 | 1e-11 | 1e-5 |

The field check is the existing `compare_fields`; a separate full-field maximum
absolute component difference is also reported. All observables must be finite.
Both zero- and imaginary-mu cases use the same policy.

These limits are empirical regression envelopes, not physics-error bounds
or evidence of bitwise equality. Strict comparisons used to select them found
maximum float field differences of 2.09e-7/1.25e-6 for 1/10 steps, double
field differences of 1.03e-11/2.72e-10, and CHI_UD differences up to 6.79e-6
(float) and 1.51e-6 (double). Gauge observable differences stayed below
1.5e-8 (float) and 6.8e-13 (double). Validate on each target architecture.

CHI_UD is a separate finite-tolerance inversion and smearing measurement; its
error is not necessarily explained by trajectory integration alone. Its looser
limit is deliberate and visible, including in double precision. This small
numerical regression supplements, not replaces, force/smearing comparisons and
historical reversibility tests. It is not a statistical physics validation.

Neither time nor Delta H nor a target speedup determines PASS. Strict author
checks remain available via `--check-trajectory-reference` with explicit
field/observable tolerances. Do not regenerate these fixtures using optimized
force merely to make a test pass.
