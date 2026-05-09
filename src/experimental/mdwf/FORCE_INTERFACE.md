# MDWF Force Interface Design Notes

This note is architecture-only.  It deliberately does not add force code, HMC/RHMC wiring, momentum updates, smearing, or changes to existing HISQ/RHMC/force modules.

## Current validated boundary

The experimental MDWF path has validated the following pre-force pieces:

- 5D spinor storage through `Spinorfield<..., 12, Ls>` with `Ls` as the physical fifth dimension.
- Raw MDWF operator application `M`.
- Explicit adjoint scaffold `M^\dagger`.
- Explicit normal operator `N = M^\dagger M`.
- Shifted normal solves `(N + sigma_i)^{-1}`.
- Rational application `R(N) phi`.
- Rational action evaluation `phi^\dagger R(N) phi`.
- Agreement between rational action evaluation and repeated shifted solves.

The next interface must use these ingredients without reinterpreting `Ls` as independent RHS and without calling the existing HISQ RHMC force path.

## Force target

For an action rational of the form

```text
R(N) = c0 + sum_i alpha_i (N + sigma_i)^(-1),
N = M^\dagger M,
S_f = phi^\dagger R(N) phi,
```

define shifted solution fields

```text
chi_i = (N + sigma_i)^(-1) phi,
eta_i = M chi_i.
```

The formal gauge variation of each shifted term is

```text
delta S_i = - alpha_i chi_i^\dagger (delta N) chi_i
          = - 2 alpha_i Re[ eta_i^\dagger (delta M) chi_i ],
```

assuming the implemented `M^\dagger` is the adjoint of `M` in the coupled 5D inner product.  This identity is the key software boundary: the force layer should not differentiate the CG solver.  It should solve for `chi_i`, apply `M` once to get `eta_i`, then pass `(eta_i, chi_i)` to a kernel-specific derivative accumulator for `delta M / delta U`.

## Minimal interface shape

The first MDWF force-facing layer should be an isolated object under `src/experimental/mdwf/`, not an edit to `src/modules/rhmc/`:

```cpp
MDWFFermionForceInput {
    gauge;
    phi;
    mdwf_operator_parameters;
    force_rational_coefficients;
    solver_tolerance;
    max_iter;
}

MDWFFermionForceOutput {
    force_field;
    shifted_solve_diagnostics;
    action_consistency_diagnostics;
}

computeMDWFFermionForce(output, input);
```

This interface should accumulate into a gauge-force field supplied by the caller.  It should not own a trajectory, update gauge links, update momenta, parse RHMC files, or decide determinant powers.

## Operator derivative split

The MDWF operator derivative should be decomposed by source of gauge dependence:

1. **4D Wilson hopping term**: gauge-link derivative of the existing Wilson-kernel path, summed over all fifth slices `s`.
2. **Clover term**: nonzero `c_sw` derivative through the existing Wilson/clover path only.  Do not duplicate clover storage in MDWF.
3. **Fifth-direction coupling**: no direct gauge-link derivative in the current scaffold because it is local in `s` and gauge independent.
4. **Mass/fifth coefficients**: no gauge force contribution unless later physics changes introduce gauge dependence.
5. **Optional smearing**: out of scope for the first force design; if added later, it requires an explicit chain rule and unsmeared regression path.

This split is important because it preserves the validated `c_sw = 0` path and keeps clover force work localized to the Wilson/clover derivative route.

## Solver and workspace requirements

The force layer must use coupled-5D solves:

- Solve each shifted system with the MDWF coupled solver adapter.
- Keep `chi_i` as one 5D field, not `Ls` right-hand sides.
- Compute `eta_i = M chi_i` with the same forward MDWF operator used in the action.
- Reuse workspaces explicitly; do not hide allocations inside inner force kernels once the interface stabilizes.
- Record shifted-solve residuals and iteration counts for force diagnostics.

The current `MDWFCoupledMultiShiftCG` is correctness-first and internally performs independent shifted solves.  That is acceptable for initial force-interface validation, but production RHMC may need a true simultaneous multishift implementation later.

## Sign and normalization risks

The force sign must be fixed by the integrator convention before wiring to HMC/RHMC.  The formula above gives the variation of `S_f`; whether the momentum update adds `-dS/dU` or accumulates the opposite sign depends on existing gauge-force conventions.

Before implementing kernels, inspect the existing force modules for:

- anti-Hermitian traceless projection convention,
- force-field storage normalization,
- gauge-link direction ordering,
- MPI halo/update assumptions for force accumulation,
- whether existing force kernels accumulate `force += contribution` or overwrite,
- whether the force object stores algebra elements or full matrices.

Do not infer these conventions from the action tests; action correctness does not determine force sign.

## Validation ladder

The first force work should proceed in this order:

1. **Architecture note only**: this file.
2. **Workspace-only scaffold**: allocate/solve/store `(chi_i, eta_i)` for force rational coefficients, with no gauge-force accumulation.
3. **Gauge-independent null test**: fifth-direction-only mock operator gives zero gauge force.
4. **Finite-difference harness**: compare action change against a single-link gauge perturbation before any HMC wiring.
5. **`c_sw = 0` Wilson-path force**: validate finite differences using the unclovered Wilson-kernel contribution.
6. **Nonzero `c_sw` clover-path force**: add clover derivative only after the `c_sw = 0` finite-difference test is stable.
7. **Only after force finite differences pass**: discuss integrator/RHMC coupling.

## Explicit non-goals for the next patch

Do not implement any of the following in the next code patch:

- HMC/RHMC trajectory integration,
- momentum updates,
- existing `src/modules/rhmc/` edits,
- HISQ force reuse,
- HISQ smearing reuse,
- optional MDWF smearing,
- production multishift optimization,
- fermion-force kernels before the force workspace and finite-difference harness are designed.
