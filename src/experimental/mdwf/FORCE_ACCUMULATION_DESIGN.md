# MDWF Force-Accumulation Design Boundary

This note is architecture-only.  It does not add an MDWF force kernel, gauge-force accumulation, momentum updates, HMC/RHMC wiring, HISQ reuse, smearing, or production force code.

## Purpose

The MDWF scalar force-contraction checks now compare

```text
dS / d epsilon
```

against selected-link finite differences for:

- the Wilson path at `c_sw = 0`,
- the Wilson + clover path at nonzero `c_sw`,
- interior and boundary/halo-touching clover probes.

The next boundary is to define how a future production accumulator should turn the validated scalar derivative into a force field, without guessing sign, projection, storage, or link-orientation conventions.

## Existing SIMULATeQCD conventions to respect

The existing gauge/HISQ/RHMC path suggests these conventions must be matched deliberately before MDWF is wired into HMC:

- Gauge-force helpers return anti-Hermitian traceless matrices by calling `TA()` after multiplying the link by its staple-like derivative.
- The RHMC integrator stores the force-like object in a `Gaugefield<floatT, onDevice, HaloDepth>` named `ipdot`.
- The momentum update uses `p <- p - i * stepsize * ipdot`.
- Existing HISQ force code builds intermediate force fields and finalizes through a projected `2 * U * Force`-like object, but this is an architectural reference only, not an MDWF implementation template.

Therefore the MDWF accumulator must not assume that the scalar action derivative sign is already the final `ipdot` sign.  The scalar checks fix `dS/depsilon`; the HMC momentum convention fixes how that derivative is converted into `ipdot`.

## Force workspace input

For each rational force term, the already validated workspace provides

```text
chi_i = (M^dagger M + sigma_i)^(-1) phi
eta_i = M chi_i
```

The derivative contribution is

```text
delta S_i = -2 alpha_i Re[ eta_i^dagger (delta M) chi_i ].
```

The future force accumulator should consume only:

- the gauge field,
- the MDWF operator parameters,
- the explicit rational force coefficients,
- the stored `(chi_i, eta_i)` fields,
- an externally supplied force field or one-link force probe.

It should not differentiate CG, own pseudofermions, choose rational determinant powers, update momenta, or call RHMC/HMC.

## Accumulation target

The first production-facing object should be a caller-supplied gauge-force field with one matrix per 4D link.  For safety, the first implementation should support a sparse one-link mode before an all-link mode:

```text
selected link and generator
  -> scalar contraction from accumulated force
  -> compare with validated finite difference
```

Only after the sparse mode matches the scalar contraction tests should an all-link accumulator be enabled.

## Operator derivative split

The future accumulator should keep the same derivative decomposition as the scalar tests:

1. Wilson hopping contribution.
2. Clover contribution through the existing Wilson/clover path.
3. Fifth-direction coupling contribution, identically zero for gauge force.
4. Mass and fifth-dimensional coefficient contribution, identically zero for gauge force.

This split should remain visible in code and tests until signs and normalizations are fully validated.

## Minimal implementation ladder

1. Keep this design note as the boundary after scalar contractions.
2. Add a one-link force-accumulator mock that writes a single selected link and verifies the scalar contraction against the existing analytic helper.
3. Add a `c_sw = 0` Wilson one-link accumulator check against `mdwfWilsonForceContractionCsw0Test`.
4. Add a nonzero-`c_sw` Wilson + clover one-link accumulator check against `mdwfCloverForceContractionNonzeroTest`.
5. Add an all-link random-direction contraction check by contracting the accumulated force with a deterministic anti-Hermitian traceless perturbation field.
6. Only then discuss HMC/RHMC integration and the final momentum-update sign.

## Open convention checks before code

Before writing production force accumulation, inspect and document:

- whether MDWF should store pre-projection derivative matrices or already projected `ipdot` matrices,
- whether the accumulator should overwrite or add into an existing force field,
- whether the final projection should be `TA(U * staple)` or an equivalent convention derived from the scalar left/right perturbation,
- whether force fields need halo updates after accumulation,
- how link orientation and boundary images are represented for halo-touching clover terms,
- which sign converts `dS/depsilon` into the existing `p <- p - i eps ipdot` update.

## Non-goals

- No RHMC/HMC edits.
- No momentum update.
- No HISQ force or smearing reuse.
- No production Wilson or clover force kernel in this step.
- No changes to `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout.
