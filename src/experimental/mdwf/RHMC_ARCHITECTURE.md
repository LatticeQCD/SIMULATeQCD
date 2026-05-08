# MDWF RHMC Architecture Notes

This note is architecture-only.  It records how a future MDWF RHMC layer should be designed after the isolated MDWF operator, normal-equation solver, shifted solver, multishift scaffold, and rational wrapper have been validated.  It does not propose editing the existing HISQ RHMC, HMC integrator, or force code yet.

## Current boundary

The existing RHMC implementation is HISQ-specific:

- `src/modules/rhmc/rhmc.h` owns HISQ smeared gauge fields, HISQ dslash objects, HISQ pseudofermions, and the HISQ force/integrator path.
- `src/modules/rhmc/integrator.h` and `src/modules/rhmc/integrator.cpp` call `HisqForce` directly during momentum updates.
- `src/modules/rhmc/rhmcParameters.h` defines the existing HISQ rational-coefficient parameter convention.
- `src/modules/inverter/inverter.h` treats `NStacks` as independent right-hand sides in the existing CG/multishift solvers.

The MDWF scaffold deliberately does not use that solver interpretation.  In the MDWF layer, `Spinorfield<..., 12, Ls>` stores one coupled 5D spinor, and the stack index is the physical fifth coordinate `s`, not an independent RHS.

## Validated MDWF-side ingredients

The current experimental MDWF layer has the following solver-facing pieces:

- `MDWFLinearOperator.h`: raw coupled 5D operator application `M`.
- `MDWFAdjointOperator.h`: explicit scaffold for `M^\dagger`.
- `MDWFNormalOperator.h`: explicit composition `N = M^\dagger M`.
- `MDWFCoupledSolverAdapter.h`: coupled 5D dot products and norms, summing over all `Ls` slices.
- `MDWFCoupledCG.h`: isolated coupled-5D CG.
- `MDWFShiftedNormalOperator.h`: explicit `(M^\dagger M + sigma)` operator.
- `MDWFCoupledMultiShiftCG.h`: correctness-first multishift interface, currently implemented as independent coupled-5D solves per shift.
- `MDWFRationalOperator.h`: explicit `c0 x + sum_i numerator_i (A + shift_i)^(-1) x`.

These are pre-RHMC building blocks only.  They do not define determinant powers, pseudofermion heatbath conventions, Metropolis action conventions, force terms, or trajectory integration.

## Recommended integration direction

Do not retrofit MDWF into the existing HISQ `rhmc` class first.  That class is coupled to HISQ-specific smearing, staggered phases, pseudofermion layout, and force construction.  A safer path is to add an MDWF-specific action layer that can later be connected to an integrator through a narrow interface.

The first MDWF RHMC design should introduce concepts in this order:

1. `MDWFRationalCoefficients` mapping from existing rational files or a new MDWF-specific rational file section.
2. `MDWFPseudofermionField` as an MDWF 5D spinor, not an existing multi-RHS staggered spinor.
3. `MDWFPseudofermionHeatbath` applying the chosen heatbath rational approximation to Gaussian 5D noise.
4. `MDWFFermionAction` computing `phi^\dagger R(M^\dagger M) phi` with explicit normal operators.
5. Only after action/heatbath validation, `MDWFFermionForce` for the gauge derivative of the chosen MDWF action.
6. Only after force validation, an HMC/RHMC trajectory wrapper that can call the MDWF action and force.

The first useful interface shape is therefore action-like, not integrator-like:

```cpp
generatePseudofermion(phi, gauge, rng, heatbath_rational);
double action(phi, gauge, action_rational);
```

No momentum updates or gauge evolution are needed for this first interface.

## Rational-coefficient conventions

Existing RHMC rational files use coefficients in the form

```text
r(x) = r_const + sum_i r_num[i] / (x + r_den[i])
```

The MDWF rational scaffold already uses the same mathematical form.  The open design question is semantic, not algebraic: which determinant power each coefficient set represents for the chosen MDWF action, Hasenbusch factorization, and number of flavors.

Do not silently reuse HISQ labels such as `r_1f`, `r_2f`, or `r_bar_*` for MDWF unless the determinant factorization is deliberately matched and documented.  A future MDWF parameter layer should make the exponent and role explicit, for example:

- heatbath rational for pseudofermion generation,
- action rational for Hamiltonian evaluation,
- force rational for molecular-dynamics force.

## Solver requirements

All MDWF solves used by RHMC must use the coupled 5D vector algebra:

- Dot products must sum over 4D sites, spin/color, and all `Ls` slices.
- `Ls` must never be interpreted as independent RHS.
- CG must be applied only to Hermitian positive-definite operators, normally explicit `M^\dagger M` or shifted `M^\dagger M + sigma`.
- Nonzero `c_sw` runs must preserve the validated `c_sw = 0` normal-equation regression.

The current `MDWFCoupledMultiShiftCG` is correctness-first and solves shifts independently.  That is acceptable for architecture and validation, but performance RHMC will eventually need a true simultaneous multishift recurrence or another production-grade shifted-solve strategy.

## Smearing policy

Do not use HISQ smearing for MDWF.  HISQ smearing is staggered-action-specific, includes staggered phase/Naik structure, and is tied to the HISQ force chain.

The first MDWF dynamical path should use the gauge links directly through the Wilson/clover kernel:

```text
MDWF operator -> Wilson/clover gauge links -> no optional smearing
```

Optional MDWF gauge-link smearing, such as HYP or stout, is a separate physics choice.  If enabled later, it must be introduced with:

- a clearly named smeared-gauge wrapper,
- a `c_sw = 0` and unsmeared regression path,
- a documented force chain rule before HMC/RHMC trajectories use it.

## What not to touch yet

Do not edit these areas for the next MDWF RHMC step:

- `src/modules/rhmc/`
- `src/modules/hisq/`
- `src/modules/inverter/`
- existing HMC/RHMC applications or tests,
- fermion force code.

The next safe code patch should stay under `src/experimental/mdwf/` and add only a pseudofermion/action scaffold with mock or fixed-gauge tests.  It should not advance gauge fields, compute forces, or call the existing RHMC integrator.

## Suggested next validation sequence

1. Validate `mdwfRationalCoefficientAdapterTest` on the cluster.
2. Add a pseudofermion heatbath mock test with a controlled diagonal normal operator.
3. Add an MDWF normal-operator action test on a fixed random gauge field.
4. Compare action rational results against repeated shifted solves, preserving the existing `mdwfRationalNormalMdwfTest` behavior.
5. Only then design the force interface.
