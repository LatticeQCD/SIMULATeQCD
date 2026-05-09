# MDWF Wilson-Path Force-Contraction Check Design

This note is architecture-only.  It does not add a Wilson force kernel, gauge-force accumulation, momentum updates, HMC/RHMC wiring, HISQ reuse, smearing, or clover-force code.

## Purpose

The validated finite-difference harness now measures the single-link action derivative

```text
dS/d epsilon ~= [S(U_+) - S(U_-)] / (2 epsilon)
```

for the MDWF rational action.  The validated analytic mock also checks that a local single-link matrix contraction can match the same left/right perturbation convention.  The next physics-sensitive bridge is a Wilson-path contraction check that compares the finite-difference derivative against the local analytic variation of the MDWF Wilson hopping term, using the already prepared rational-force workspaces.

This check should still stop before production force accumulation.  Its output is one scalar analytic derivative for one selected link and generator, not a full force field.

## Inputs

The first Wilson-path check should be explicit and small:

- fixed gauge field `U`,
- fixed MDWF pseudofermion/source field `phi`,
- force rational coefficients,
- action rational coefficients only if the same test also evaluates `S(U_\pm)`,
- MDWF parameters: `Ls`, mass, fifth-direction coefficients, and `c_sw = 0`,
- one selected 4D site, direction `mu`, and anti-Hermitian traceless generator `T_a`,
- finite-difference `epsilon`,
- solver tolerance and max iterations.

Use `MDWFFermionForceWorkspace` to compute and store

```text
chi_i = (M^\dagger M + sigma_i)^(-1) phi
eta_i = M chi_i
```

for each force-rational term.  Do not reinterpret `Ls` as independent RHS.

## Mathematical target

For the action rational

```text
R(N) = c0 + sum_i alpha_i (N + sigma_i)^(-1),  N = M^\dagger M,
S = phi^\dagger R(N) phi,
```

the shifted-term variation is

```text
delta S_i = -2 alpha_i Re[ eta_i^\dagger (delta M) chi_i ],
eta_i = M chi_i.
```

For the first Wilson-path check, set `c_sw = 0`.  Then `delta M` should include only the 4D Wilson gauge-link hopping contribution on each fifth slice.  The fifth-direction coupling is gauge independent and should contribute zero.

The scalar compared to finite difference is

```text
dS/depsilon = sum_i -2 alpha_i Re[ eta_i^\dagger (dM/depsilon) chi_i ].
```

No force sign for the integrator is fixed here.  This is an action derivative check.

## Local Wilson-link derivative

`MDWFWilsonSlice` currently applies the Wilson path through the existing `DWilson.h` functors.  For a link `U(x,mu)`, the unclovered Wilson kernel contributes to `M psi` at site `x` through the forward hop to `x+mu`.  Because the MDWF wrapper applies this 4D Wilson kernel independently on each fifth slice, the local contraction should sum over `s = 0..Ls-1`.

The first check should implement a test-local derivative helper that mirrors the existing Wilson functor convention rather than modifying `DWilson.h`.  For the selected link and left perturbation

```text
U(epsilon) = exp(epsilon T_a) U,
dU/depsilon |0 = T_a U,
```

and for right perturbation

```text
U(epsilon) = U exp(epsilon T_a),
dU/depsilon |0 = U T_a.
```

The helper should form only the selected-link contribution to `(dM/depsilon) chi_i`, then contract it with `eta_i`.  It should not write a gauge-force field.

## Test scope

The first runnable test should be named something like:

```text
mdwfWilsonForceContractionCsw0Test
```

It should:

1. construct the same fixed gauge/source setup used by the finite-difference action tests,
2. evaluate the finite-difference action derivative with `c_sw = 0`,
3. build `MDWFFermionForceWorkspace` using the matching force rational coefficients,
4. compute one scalar Wilson-path analytic contraction for the same link/generator,
5. compare finite-difference and analytic derivatives with tolerances based on solver residual and epsilon stability,
6. log both derivatives, absolute/relative difference, max shifted residual, and the selected link/generator convention.

Keep this as a test-local scalar contraction.  Do not accumulate into `Gaugefield` force storage.

## Expected validation ladder

1. `mdwfAnalyticForceContractionMockTest`: already validates single-link contraction sign/orientation on a controlled linear action.
2. `mdwfWilsonForceContractionCsw0Test`: compare the Wilson-path scalar contraction to the `c_sw = 0` finite-difference MDWF rational action.
3. Sweep at least two generators or link directions after the first selected-link case passes.
4. Only after the `c_sw = 0` Wilson-path contraction is stable, add a nonzero-`c_sw` clover-path contraction design.
5. Only after Wilson and clover scalar contractions pass, discuss full force-field accumulation and projection conventions.

## Risks and unknowns

- The exact sign and normalization must follow `DWilson.h` as used through `MDWFWilsonSlice`; do not infer it from continuum formulas alone.
- `applyMDWFCloverWilsonSlice` uses a different path from `applyMDWFWilsonSlice`; keep the first check at `c_sw = 0`.
- The backward-hop dependence of a selected link must be handled carefully if the derivative helper accumulates contributions from neighboring output sites.  The first implementation should inspect the exact `gamma5DiracWilson` stencil and choose one convention deliberately.
- The rational coefficients used for force must correspond to the action rational being finite-differenced; otherwise the scalar comparison will fail for a bookkeeping reason, not a force reason.
- This check still does not define the HMC momentum update sign.

## Non-goals

- No RHMC/HMC edits.
- No force-field accumulation.
- No momentum update.
- No HISQ force or smearing reuse.
- No clover-force derivative in the first Wilson-path check.
- No changes to `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout.
