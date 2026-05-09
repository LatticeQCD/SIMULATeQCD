# MDWF Finite-Difference Harness Design

This note is design-only.  It does not add gauge-force accumulation, momentum updates, HMC/RHMC wiring, smearing, or edits to existing force modules.

## Purpose

The finite-difference harness is the bridge between the validated MDWF rational action and any future MDWF force kernel.  Its job is to compare a numerical directional derivative of the action,

```text
dS/d epsilon ~= [S(U_+) - S(U_-)] / (2 epsilon),
```

against a future analytic force contraction along the same single-link perturbation.  The harness should exist before implementing force accumulation so that sign, normalization, projection, and link-orientation conventions are pinned down by tests rather than guessed.

## Inputs

The first harness should be explicit and small:

- fixed gauge field `U`,
- fixed MDWF source or pseudofermion field `phi`,
- fixed rational action coefficients,
- MDWF operator parameters including `Ls`, mass/fifth coefficients, and `c_sw`,
- one selected 4D site and direction `mu`,
- one anti-Hermitian traceless generator direction `T_a`,
- perturbation size `epsilon`,
- solver tolerance and max iterations.

The harness should use the existing `computeMDWFRationalAction` path for action evaluation.  It must not call or depend on a force implementation.

## Perturbation convention

Use a single-link group perturbation with the convention documented in the test:

```text
U_+(x,mu) = exp(+epsilon T_a) U(x,mu)
U_-(x,mu) = exp(-epsilon T_a) U(x,mu)
```

where `T_a` is anti-Hermitian and traceless.  The first patch should choose one simple generator and one fixed link.  A later sweep can test multiple generators, directions, and sites.

The harness must explicitly record whether the perturbation is left-multiplied or right-multiplied.  Existing force conventions may use one or the other; this harness should not silently choose a convention and then hide it.

## Action evaluation

For each perturbation:

1. Copy the base gauge field to a work gauge.
2. Apply exactly one link perturbation.
3. Rebuild or refresh any MDWF operator object that holds gauge references.
4. Evaluate `S(U_\pm) = phi^\dagger R(M^\dagger M[U_\pm]) phi`.
5. Record solver convergence and shifted residuals.

The finite-difference estimate is

```text
deltaS_fd = (S(U_+) - S(U_-)) / (2 epsilon).
```

Use the real part of the action for the derivative.  The imaginary part should remain numerically tiny and should be logged as a diagnostic.

## Epsilon strategy

The first harness should not assume one `epsilon` is trustworthy.  Use a short sweep, for example:

```text
epsilon in {1e-3, 3e-4, 1e-4}
```

and report:

- `S(U_+)`,
- `S(U_-)`,
- `deltaS_fd`,
- imaginary-action relative size,
- shifted-solve max residual,
- variation stability between neighboring `epsilon` values.

If the derivative changes wildly across the sweep, do not proceed to analytic force comparison.

## Expected staged tests

1. **Gauge-independent null harness**: use a mock or fifth-direction-only action where `M` has no gauge dependence.  The finite-difference derivative should be zero within numerical tolerance.
2. **MDWF `c_sw = 0` action harness**: perturb a fixed random gauge and evaluate the finite-difference action derivative with clover disabled.
3. **MDWF nonzero `c_sw` action harness**: repeat only after the `c_sw = 0` harness is stable.
4. **Analytic-force comparison**: compare the finite-difference derivative against a future force contraction.  This is not part of the first harness patch.

The null harness should come first because it catches accidental gauge mutation, stale operator references, and source-field changes.

## Minimal software shape

The first code scaffold should stay under `src/experimental/mdwf/` and `src/testing/`:

```cpp
MDWFFiniteDifferenceProbe {
    site;
    mu;
    generator_id;
    epsilon;
    multiplication_side;
}

MDWFFiniteDifferenceResult {
    action_plus;
    action_minus;
    derivative;
    max_shifted_residual;
    action_imag_relative;
}

evaluateMDWFFiniteDifferenceAction(result, probe, base_gauge, phi, coefficients, parameters);
```

This is a harness for action derivatives only.  It should not allocate a force field and should not expose a momentum update.

The initial scaffold is `MDWFFiniteDifferenceHarness.h`.  Its first test is `mdwfFiniteDifferenceNullTest`, which perturbs a gauge link but evaluates a gauge-independent mock action through `computeMDWFRationalAction`, so the centered derivative must vanish.

## Gauge-field risks to inspect before coding

Before writing the code scaffold, inspect existing gauge-field helpers for:

- host/device accessor patterns for one-link updates,
- how to copy gauge fields without aliasing,
- whether gauge halos must be refreshed after a single-link mutation,
- link storage order and compression mode for `R18`,
- how existing tests construct simple SU(3) perturbations,
- whether `Gaugefield::random` leaves links projected to SU(3) in the expected representation.

If there is no safe one-link update helper, the first implementation should add a tiny test-local perturbation functor rather than changing global gauge-field APIs.

## What this harness must not do

- Do not accumulate gauge force.
- Do not create a momentum field.
- Do not call `src/modules/rhmc/`.
- Do not call existing HISQ force or smearing code.
- Do not change Wilson/clover or MDWF operator behavior.
- Do not infer the final force sign from finite differences alone; the final comparison must include the integrator convention.
