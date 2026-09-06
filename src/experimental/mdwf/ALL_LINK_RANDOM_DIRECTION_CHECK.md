# MDWF All-Link Random-Direction Contraction Check

This note is design-only.  It does not add an all-link force kernel, choose a
projection convention, define the HMC force sign, update momenta, or wire MDWF
into RHMC/HMC.

## Purpose

The validated one-link tests establish the scalar action derivative and sparse
storage conventions for:

- the Wilson path at `c_sw = 0`,
- the Wilson plus clover path at nonzero `c_sw`,
- interior and periodic-boundary/halo-touching probes.

The next check should perturb every owned 4D gauge link simultaneously along a
deterministic anti-Hermitian traceless direction and compare the centered
finite-difference action derivative with the sum of the analytic link
contractions.  Its purpose is to expose all-link indexing, link ownership,
periodic-image, and accumulation errors before any production force convention
is selected.

## Directional derivative

For each owned link `ell = (x, mu)`, construct a fixed matrix `H_ell` satisfying

```text
H_ell^dagger = -H_ell,
tr(H_ell) = 0,
-Re tr(H_ell H_ell) = 1.
```

Use one documented perturbation convention for the first check:

```text
U_ell(+) = exp(+epsilon H_ell) U_ell,
U_ell(-) = exp(-epsilon H_ell) U_ell.
```

This is a left-multiplied test direction only.  It does not decide whether a
future production force should use a left- or right-oriented representation.
Right multiplication can be added as a later independent regression.

The numerical derivative is

```text
D_fd = [S(U(+)) - S(U(-))] / (2 epsilon),
```

where every owned bulk link is perturbed in the same action evaluation.

For the existing rational force workspace, the analytic derivative remains

```text
D_an = -2 sum_i alpha_i Re[eta_i^dagger (delta_H M) chi_i].
```

The implementation must continue to report

```text
D_an = D_Wilson + D_clover,
```

with `D_clover = 0` in the first `c_sw = 0` gate.  The fifth-direction and mass
terms remain gauge independent and contribute zero.

## Deterministic direction field

The direction must be reproducible and independent of device scheduling:

1. Derive coefficients from global periodic coordinates and `mu`, not from
   local site indices or traversal order.
2. Combine the anti-Hermitian traceless generators already used by the MDWF
   finite-difference harness.
3. Normalize every `H_ell` with `-Re tr(H_ell H_ell)`.
4. Reject a zero or non-finite norm.
5. Use the same direction field for `U(+)`, `U(-)`, and the analytic
   contraction.

A fixed integer hash or explicit coordinate formula is preferable to a runtime
random-number generator.  The word "random" in this check means a
deterministic, irregular direction over all links.  The current three-generator
subset is sufficient for this first indexing/contraction check, but it is not
complete eight-generator Lie-algebra coverage.

## Direction-dependent accumulator representation

The one-link scaffold stores only the already-computed component along its
selected generator.  The all-link correctness check may use the same restricted
idea.  If `d_ell` is the analytic scalar contribution for direction `H_ell`,
store

```text
A_ell[H] = d_ell H_ell / Re tr(H_ell H_ell),
```

so that

```text
Re tr(H_ell A_ell[H]) = d_ell,
D_acc = sum_ell Re tr(H_ell A_ell[H]).
```

`A_ell[H]` is explicitly direction dependent.  It is not the full
direction-independent derivative matrix, not a projected algebra force, and
not an `ipdot` field.  It must not be reused with another direction or exposed
as a production force interface.  This restriction lets the test validate
all-link storage and summation without prematurely choosing projection,
orientation, normalization, or HMC sign conventions.

## Correctness-first implementation shape

Keep the first implementation under `src/experimental/mdwf/` and
`src/testing/`:

1. Add a test-local or MDWF-only deterministic direction functor.
2. Perturb all bulk links into separate `gaugePlus` and `gaugeMinus` fields,
   then refresh gauge halos.
3. Build the force workspace once on the unperturbed gauge field.  Do not solve
   again for every link.
4. Evaluate the Wilson variation for all links and all fifth slices.
5. For nonzero `c_sw`, keep the clover path-factor variation separate and
   attribute each active path-factor contribution to its periodic physical
   link.
6. Write the direction-dependent matrices into a caller-supplied
   gauge-field-like test object.
7. Inspect every owned bulk link and reduce the scalar sum.

The analytic code should preload or reuse host/device accessors once.  Recalling
the current selected-link helper separately for every link would repeatedly
copy the same spinors and would obscure ownership errors.

No `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout change is
needed.

## First validation scope

The first code patch should be the `c_sw = 0` Wilson gate on the existing
single-rank `6^4`, `Ls = 8` test lattice.  It should verify:

- exactly `4 * volume` owned bulk links are inspected;
- every direction matrix and accumulated matrix is finite;
- the direction norm is valid on every link;
- the sum of absolute per-link contributions is nonzero;
- `D_acc` agrees with the direct analytic sum;
- `D_an` agrees with `D_fd`;
- the rational action and force-workspace solves converge;
- the existing one-link `c_sw = 0` test remains passing.

Use a short centered-difference epsilon sweep, initially

```text
epsilon in {1e-3, 3e-4, 1e-4},
```

and log the derivative stability, action imaginary relative size, maximum
shifted residual, and analytic/finite-difference differences.  Existing
one-link tolerances may be used as initial guidance, but any all-link tolerance
must be justified by the observed epsilon plateau and solver residuals.

Only after this gate passes should the same test shape add nonzero `c_sw`,
retain separate Wilson/clover totals, and confirm a finite clover response.

## Boundary and MPI risks

Perturbing every link automatically includes links on periodic boundaries, but
the initial single-rank test does not validate distributed ownership.  It must
not be reported as a multi-rank halo validation.

A later multi-rank extension must explicitly verify:

- direction coefficients are derived from global coordinates;
- each physical link is owned and counted exactly once;
- clover path contributions through halo images are routed to the owning link;
- gauge halos are refreshed after the simultaneous perturbation;
- local analytic sums use the same global reduction as the action comparison.

Do not solve ownership by changing indexer or gauge-field layouts.

## Required diagnostics

On failure, report at least:

- `c_sw`, `Ls`, epsilon, and multiplication side;
- number of inspected and invalid links;
- direction-norm minimum and maximum;
- Wilson, clover, total analytic, accumulated, and finite-difference values;
- absolute and relative differences;
- epsilon-sweep stability;
- maximum action and force-workspace residuals;
- action imaginary relative size.

## Deferred production decisions

This check deliberately does not decide:

- whether a production field stores an unprojected derivative or `TA(...)`;
- whether accumulation overwrites or adds;
- whether links are represented in a left- or right-oriented convention;
- the conversion from `dS/d epsilon` to the existing `ipdot` convention;
- the sign used by `p <- p - i epsilon ipdot`;
- RHMC/HMC integration, smearing, or multishift optimization.

Those decisions require a direction-independent matrix derivation and a
deliberate review of the existing integrator conventions after the all-link
scalar check is validated.
