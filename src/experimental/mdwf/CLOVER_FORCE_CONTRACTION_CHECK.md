# MDWF Nonzero-c_sw Clover-Path Force-Contraction Check Design

This note is architecture-only.  It does not add a clover-force kernel, gauge-force accumulation, momentum updates, HMC/RHMC wiring, HISQ reuse, smearing, or production force code.

## Purpose

The validated `c_sw = 0` Wilson-path scalar contraction now fixes the single-link perturbation convention for the MDWF rational action.  The next physics-sensitive step is to extend the scalar contraction to nonzero `c_sw` by adding only the clover-path derivative through the existing Wilson/clover route.

The check should still produce one scalar action derivative for selected links and generators.  It should not accumulate a force field.

## Existing clover path to mirror

The MDWF clover operator currently routes through:

```text
applyMDWFCloverOperator
  -> applyMDWFCloverWilsonSlice
      -> preCalcFmunu
          -> FieldStrengthTensor / plaqClover
      -> DiracWilsonEvenOdd2
      -> DiracWilsonEvenEven2
  -> applyMDWFFifthDimCoupling
```

The clover matrices are stored as `Vect18` upper/lower 6x6 Hermitian blocks.  `preCalcFmunu` builds these blocks from the existing `FieldStrengthTensor`, multiplies the `sigma_mu_nu F_mu_nu` part by `-0.5 * c_sw`, then adds the mass diagonal.  The current MDWF apply path uses the upper/lower clover matrices, not the stored inverse matrices.

For `c_sw = 0`, the gauge derivative of the clover term is zero, and the existing Wilson-path contraction remains the regression gate.

## Scalar derivative target

For the rational action

```text
S = phi^dagger R(N) phi,  N = M^dagger M,
```

the shifted force workspace provides

```text
chi_i = (N + sigma_i)^(-1) phi
eta_i = M chi_i
```

using the same nonzero-`c_sw` MDWF operator as the finite-difference action.  The scalar derivative target is

```text
dS/depsilon = sum_i -2 alpha_i Re[
    eta_i^dagger (dM_Wilson/depsilon + dM_clover/depsilon) chi_i
]
```

where `dM_Wilson/depsilon` is the already validated Wilson hopping derivative, evaluated with the nonzero-`c_sw` force workspace, and `dM_clover/depsilon` is the new local clover derivative.  The fifth-direction coupling is gauge independent and contributes zero.

Do not compare the nonzero-`c_sw` finite-difference derivative against the old `c_sw = 0` Wilson scalar contraction.  Nonzero `c_sw` changes both the operator and the shifted solution fields.

## Clover derivative scope

For a selected link `U(x, mu)`, the clover derivative is local in spin/color at each affected site but nonlocal over the clover stencil:

```text
dC_upper/lower(y) / depsilon
  = -0.5 * c_sw * sum_{rho < sigma}
      sigmaF_upper/lower_signs(rho, sigma) * dF_{rho sigma}(y) / depsilon
```

The mass diagonal has zero gauge derivative.  The signs and upper/lower block structure must match `preCalcFmunu` exactly, not a continuum convention written independently.

`dF_{rho sigma}(y)` must mirror `FieldStrengthTensor`:

```text
F_{rho sigma}(y) =
  (-i / 8) * (Q_{rho sigma}(y) - Q_{rho sigma}(y)^dagger)
  - trace part
```

with `Q` supplied by `plaqClover`.  A link perturbation contributes wherever the selected link appears in one of the four plaquette products in `plaqClover(y, rho, sigma)`.

## Proposed first runnable check

The first nonzero-`c_sw` contraction test should be named something like:

```text
mdwfCloverForceContractionNonzeroTest
```

It should:

1. keep the same deterministic gauge/source setup as the finite-difference action tests,
2. evaluate and store the finite-difference action derivative before preparing the force workspace,
3. prepare `MDWFFermionForceWorkspace` with nonzero `c_sw`,
4. compute the Wilson hopping scalar derivative with the existing test-local Wilson helper and the nonzero-`c_sw` workspace,
5. compute the clover scalar derivative with a new test-local helper that mirrors `preCalcFmunu` and `FieldStrengthTensor`,
6. compare `Wilson + clover` against the nonzero-`c_sw` finite-difference derivative,
7. log the Wilson part, clover part, total analytic derivative, finite-difference derivative, residuals, selected link, generator, perturbation side, and epsilon.

The finite-difference-before-workspace ordering is intentional; the `c_sw = 0` Wilson sweep exposed that ordering as important for stable action comparisons.

## Implementation ladder

1. Preserve `mdwfWilsonForceContractionCsw0Test` as the gate.
2. Add a test-local clover derivative helper only; do not edit `DWilson.h` or `FieldStrengthTensor.h` in the first clover patch.
3. First validate one interior selected link and one generator to avoid boundary-condition ambiguity.
4. Add a second probe only after the first nonzero-`c_sw` scalar contraction passes.
5. Only after scalar Wilson+clover contractions pass, discuss production force-field accumulation and projection conventions.

## Risks and unknowns

- The clover stencil has multiple affected sites for one selected link; missing one plaquette orientation will give a plausible but wrong scalar derivative.
- The `-0.5 * c_sw` factor and upper/lower block signs must match `preCalcFmunu` exactly.
- The trace subtraction in `FieldStrengthTensor` must be differentiated with the same normalization.
- Left/right link perturbations must be inserted at the exact position where the selected link appears in each plaquette product.
- The current apply path stores clover inverse fields but does not use them; if a future preconditioned clover path uses the inverse matrices, this derivative design must be revisited.
- This check fixes an action derivative convention, not the final HMC momentum-update sign.

## Non-goals

- No RHMC/HMC edits.
- No force-field accumulation.
- No momentum update.
- No HISQ force or smearing reuse.
- No changes to `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout.
- No production clover-force kernel in the first scalar-contraction patch.
