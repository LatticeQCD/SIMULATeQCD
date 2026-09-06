# MDWF Direction-Independent Force and MPI Convention Review

This note reviews the boundary after the validated single-rank all-link
random-direction checks.  It does not add a direction-independent force
matrix, choose a production projection or HMC force sign, update momenta,
extend the tests to MPI, or modify existing gauge, HISQ, RHMC, indexer, or
communication code.

## Validated input to this review

The current tests establish one scalar identity for a fixed, deterministic
left direction on every link:

```text
sum_ell dS_ell(H_ell)
  = D_Wilson + D_clover
  = D_acc
  = D_fd
```

on a single rank.  The `c_sw = 0` and nonzero-`c_sw` tests also show that the
direction-dependent matrices are stored on all `4 * volume` bulk links and
round-trip their defining contractions.

Those results do not reconstruct the response to an arbitrary link direction.
In particular,

```text
A_ell[H] = dS_ell(H) H / Re tr(H H)
```

depends on the direction used to create it.  It is not a reusable link
derivative, an `ipdot` field, or evidence for an MPI ownership rule.

## Existing code conventions observed

The following facts come directly from the current SIMULATeQCD implementation:

- `do_evolve_Q` in `src/modules/rhmc/integrator.cpp` evolves a link as
  `U <- exp(i dt p) U`.  Gauge evolution is therefore left multiplied.
- `SU3::gauss()` creates Hermitian traceless momentum matrices.
- `do_evolve_P` updates the momentum as
  `p <- p - i dt ipdot`.
- The gauge force and the final HISQ force call `SU3::TA()` before their result
  is consumed as `ipdot`.
- `SU3::TA()` implements
  `(A - A^dagger) / 2` followed by trace subtraction.  It does not contain an
  extra factor of two.
- The HISQ finalizer forms `TA(2 U F)`.  Its link placement and factor of two
  are specific evidence about the HISQ implementation, not an MDWF
  normalization rule.
- Gauge-force and HISQ entry points overwrite their final bulk output.
  HISQ adds many intermediate path contributions internally before the final
  overwrite/projection.
- `SiteComm::updateAll()` extracts owned inner-boundary values and writes
  received values into outer halos.  The injection uses assignment, not
  addition, so `updateAll()` is a halo refresh and not a reverse-halo
  accumulation operation.
- The integrator evolves only bulk momentum links.  It does not require
  refreshed `ipdot` halos for the final `evolveP` loop.

These observations constrain a future MDWF interface, but they do not by
themselves prove the MDWF derivative normalization or its final sign.

## Direction-independent matrix definition

The next scalar-to-matrix step should regard each link derivative as a linear
functional on `su(3)` and define its unique test-local algebra representative
`K_ell` by the left-variation identity

```text
U_ell(epsilon) = exp(epsilon H) U_ell,
H^dagger = -H,
tr(H) = 0,

dS_ell(H) = Re tr(H K_ell),
K_ell^dagger = -K_ell,
tr(K_ell) = 0.
```

For every anti-Hermitian traceless `H`, not just the direction used by the
all-link test.  This definition preserves the contraction convention already
used by the MDWF finite-difference and analytic helpers.  `K_ell` is a
direction-independent contraction representative for testing.  Defining it
does not choose how a production kernel should construct and project a raw
spinor-bilinear matrix, and it does not identify `K_ell` with `ipdot`.

The Wilson and clover matrices should remain separately visible:

```text
K_ell = K_ell,Wilson + K_ell,clover.
```

Fifth-direction coupling, mass terms, and fifth-dimensional coefficients
remain gauge independent and contribute zero.

The first reconstruction test must span all eight real dimensions of
`su(3)`.  The current three-generator subset is insufficient.  Use a
test-local, documented eight-generator basis and measure

```text
G_ab = Re tr(H_a H_b).
```

Solve the Gram system for the coefficients of `K_ell`; do not assume the
basis is orthonormal or insert a factor of two from a textbook normalization.
For selected links, verify all of:

```text
dS_ell(H_a) = Re tr(H_a K_ell),       a = 1,...,8
dS_ell(sum_a c_a H_a)
             = Re tr((sum_a c_a H_a) K_ell)
```

using independent finite differences and irregular coefficient combinations.
The reconstructed matrix must predict directions that were not used to build
it.

## Left and right orientation

The first matrix should use the left convention above because it matches
`do_evolve_Q` and the validated all-link perturbation.  A right-oriented
matrix must not be mixed into the same interface.

For

```text
U(epsilon) = U exp(epsilon H_R),
```

the equivalent left generator is

```text
H_L = U H_R U^dagger.
```

Consequently, matrices satisfying the same trace pairing should obey

```text
K_R = U^dagger K_L U,
Re tr(H_R K_R) = Re tr(H_L K_L).
```

This covariance relation should become a selected-link regression before a
right-oriented representation is supported.  It is not a reason to convert
orientations implicitly inside the first implementation.

## Projection boundary

For anti-Hermitian traceless `H`,

```text
Re tr(H B_ell) = Re tr(H TA(B_ell))
```

for any raw matrix `B_ell`.  Thus scalar action variations determine only the
anti-Hermitian traceless part of a future raw bilinear.  Its Hermitian and
trace components are invisible to the validated directional derivatives.
The eight-direction reconstruction above directly produces the unique
algebra representative `K_ell`; it cannot reconstruct the invisible parts of
`B_ell`.

The next patch should reconstruct and test `K_ell` without declaring it to be
production `ipdot` or choosing a raw-bilinear projection path.  After
eight-direction reconstruction passes, a separate production-matrix gate
should:

1. derive the MDWF raw bilinear `B_ell` without borrowing the HISQ form;
2. apply the existing `SU3::TA()` exactly once;
3. compare `TA(B_ell)` with the independently reconstructed `K_ell`;
4. check anti-Hermiticity, tracelessness, and all basis/mixed contractions;
5. compare against the existing gauge-force storage convention with a
   convention-only test;
6. reject any unexplained factor of two, link multiplication, or sign.

The existing HISQ expression `TA(2 U F)` must not be copied.  Whether an MDWF
raw bilinear needs multiplication by `U` depends on how that bilinear is
defined.  With `K_ell` defined directly by the left trace identity above, the
link placement is already fixed for the independent reference matrix.

## Accumulation semantics

Linearity requires contributions from fifth slices, rational terms, and the
Wilson/clover split to add into one link matrix.  The implementation should
make two different operations explicit:

```text
zero/overwrite the destination for a new force evaluation
add one Wilson, clover, slice, or rational contribution internally
```

The first isolated API should overwrite a caller-supplied destination after
explicit initialization.  It must not depend on its previous contents.
Internal helpers may add to a zeroed workspace.  This matches the existing
force entry points' final overwrite behavior while preventing stale-field
errors.

Any later API that adds an MDWF force to an already populated integrator field
must use a separately named additive operation and a test with a nonzero
sentinel destination.  That integration decision is outside this patch.

## HMC sign boundary

The existing integrator establishes the storage/update relationship

```text
U <- exp(i dt p) U,
p <- p - i dt ipdot,
```

with Hermitian `p` and anti-Hermitian `ipdot`.  It does not replace an MDWF
sign derivation.

Conditionally, if a future projected MDWF matrix `F` satisfies exactly

```text
dS(H) = Re tr(H F)
```

for the left variation and no hidden normalization is present, then setting
`H = i delta_p` gives the algebraic mapping expected by the integrator.
That conditional calculation is not sufficient to select the production
sign, because the matrix construction, rational-action sign, and force API
may each absorb a sign or normalization.

Before HMC/RHMC wiring, add an isolated impulse-convention test that:

1. computes `dS/d epsilon` by a left finite difference;
2. contracts the candidate projected matrix with the identical direction;
3. applies the candidate through the literal `p <- p - i dt ipdot` rule;
4. verifies the expected Hamiltonian force response for both `+dt` and `-dt`;
5. compares the convention with the existing gauge-force path.

Only that test should promote the matrix to `ipdot` and freeze the final sign.
A later short-trajectory reversibility and `Delta H` test is an integration
gate, not a substitute for the local sign test.

## MPI ownership and halo boundary

The first MPI implementation should use an owner-computes rule:

```text
each rank writes only its owned bulk links;
for each owned link, it enumerates every Wilson and clover occurrence;
neighbor gauge and spinor values are read through refreshed halos.
```

This shape counts every physical link exactly once and avoids remote writes.
It is preferable to an origin-computes path loop that produces contributions
for links owned by another rank, because the current `updateAll()` operation
cannot sum those remote contributions back to their owners.

The owner-computes implementation must:

- derive deterministic test directions from periodic global coordinates;
- update required gauge and spinor halos before evaluating boundary stencils;
- confirm that the existing halo depth covers every clover occurrence;
- store and inspect force matrices on owned bulk links only;
- use the existing global sum reduction only for scalar diagnostics;
- treat a force-field `updateAll()` after construction as an optional halo
  refresh for later readers, never as force accumulation.

If owner-computes cannot express the clover derivative without remote writes,
stop and design an explicit reverse accumulation operation.  Do not change
`SiteComm`, `GIndexer`, or the gauge-field layout as an incidental MDWF patch.

## Required MPI validation

MPI validation should compare one- and multi-rank runs representing the same
global gauge field, source, and direction field.  At minimum it must check:

- the global number of owned links is exactly `4 * global_volume`;
- every global link has one owner and no duplicate contribution;
- selected interior and rank-boundary link matrices agree component by
  component;
- Wilson, clover, and total global contractions agree separately;
- the direction-independent contraction agrees with the action finite
  difference;
- basis and mixed-direction checks include rank-boundary links;
- results are decomposition independent within justified floating-point
  tolerances;
- action and force-workspace residuals remain controlled.

A passing single-rank periodic-boundary probe is not MPI evidence.  A matching
global scalar alone is also insufficient because ownership errors can cancel;
selected-link matrices and per-part diagnostics are required.

## Current isolated projection gate

The single-rank selected-link reconstructions pass for both `c_sw = 0` Wilson
and separate nonzero-`c_sw` Wilson/clover matrices.  The raw left-oriented
Wilson bilinear gate also passes: after all fifth-slice and rational
contributions are accumulated, one `SU3::TA()` agrees with the independent
`c_sw = 0` reconstruction at roundoff while removing a demonstrably nonzero
invisible component.

The separate clover raw-projection gate now also passes.  It derives the
Hermitian field-strength sensitivity, transports it through each clover-leaf
occurrence of the selected link, and accumulates a genuinely unprojected raw
clover matrix.  One `SU3::TA()` matches the independent clover reconstruction
and preserves all eight basis contractions plus the held-out mixed direction
at roundoff.

The selected-link test-only gate now combines the independently derived raw
Wilson and clover matrices and validates projection linearity:

```text
TA(B_Wilson + B_clover)
  = TA(B_Wilson) + TA(B_clover)
  = K_Wilson + K_clover
  = K_total.
```

It retains separate raw/projected component diagnostics, basis and held-out
mixed contractions, and full-matrix comparisons.  The complete chain passes
at roundoff without an extra factor of two, link multiplication, or sign.

Do not yet:

- call the projected result a production force field or `ipdot`;
- choose the HMC force sign;
- combine the Wilson and clover raw matrices into an all-link field;
- expose the test-only all-link overwrite scaffold as a production API;
- add MPI ownership or communication;
- edit RHMC/HMC, gauge force, HISQ force, `SiteComm`, or `GIndexer`.

The single-rank all-link direction-independent storage contract is recorded in
`ALL_LINK_DIRECTION_INDEPENDENT_STORAGE.md`.  It specifies bulk-only coverage,
explicit destination overwrite behavior, zeroed internal Wilson/clover
buffers, addition over slices/terms/path occurrences, and nonzero-sentinel
tests that distinguish overwrite from any future additive entry point.

The document's `c_sw = 0` Wilson-only storage gate is implemented and
cluster-validated as a test-only bulk overwrite scaffold.  All `5184` bulk
links are finalized exactly once; zero, nonzero-sentinel, and repeated writes
agree; and the stored deterministic-direction contraction matches the
validated `157.723` scalar derivative.  It does not call the stored matrices
`ipdot`, choose the HMC sign, or imply an MPI ownership rule.  The
nonzero-`c_sw` all-link extension with separate raw Wilson/clover buffers,
direct clover path-factor attribution, and one projection of their sum is now
implemented and cluster-validated as the next test-only gate.  All `5184`
bulk links receive exactly `72` clover path additions, the independent
Wilson/clover/total contractions agree with the stored matrices at roundoff,
and the zero/sentinel/repeated overwrite checks are exact.
