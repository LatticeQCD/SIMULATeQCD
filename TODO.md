# MDWF TODO

## Current status

- [x] Stage 1: represent MDWF 5D spinors with `Spinorfield<..., 12, Ls>`.
- [x] Stage 2: implement fifth-direction-only coupling.
- [x] Stage 2 validation: `mdwfFifthDimTest` passes on the cluster for `All`, `Even`, and `Odd` layouts with `Ls = 8`.
- [x] Stage 3: apply the existing 4D Wilson kernel independently on each fifth-dimensional slice.
- [x] Stage 3 validation: `mdwfWilsonSliceTest` passes on the cluster with `Ls = 8`.
- [x] Stage 4: combine 4D Wilson slice application with fifth-direction coupling into an MDWF operator skeleton.
- [x] Stage 4 validation: `mdwfOperatorSkeletonTest` passes on the cluster with `Ls = 8`.
- [x] Stage 5: add clover through the Wilson-kernel path.
- [x] Stage 6: validate `c_sw = 0` against the unclovered Wilson/domain-wall path.
- [x] Stage 5/6 validation: `mdwfCloverCsw0Test` passes on the cluster with `Ls = 8`.
- [x] Nonzero `c_sw` sanity: `mdwfCloverNonzeroTest` passes on the cluster and detects a finite clover response.
- [x] Workspace validation: `mdwfOperatorWorkspaceTest` passes on the cluster with `Ls = 8`.
- [x] Stage 7 scaffold: add a non-solving `MDWFLinearOperator` wrapper around the MDWF operator workspace.
- [x] Stage 7 scaffold validation: `mdwfLinearOperatorTest` passes on the cluster with `Ls = 8`.
- [x] Coupled-5D solver-adapter scaffold: add `MDWFCoupledSolverAdapter` with 5D matvec and aggregated 5D inner products.
- [x] Coupled-5D solver-adapter validation: `mdwfCoupledSolverAdapterTest` passes on the cluster with `Ls = 8`.
- [x] Coupled-5D CG scaffold: add isolated `MDWFCoupledCG` using `MDWFCoupledSolverAdapter` primitives.
- [x] Coupled-5D CG identity validation: `mdwfCoupledCGIdentityTest` passes on the cluster with `Ls = 8`.
- [x] Coupled-5D CG diagonal mock scaffold: add an `A = 2 I` positive-definite test target.
- [x] Coupled-5D CG diagonal mock validation: `mdwfCoupledCGDiagonalTest` passes on the cluster with `Ls = 8`.
- [x] Coupled-5D CG slice-diagonal mock scaffold: add `A_s = 1 + 0.1 s` with exact solution `x_s = b_s / A_s`.
- [x] Coupled-5D CG slice-diagonal mock validation: `mdwfCoupledCGSliceDiagonalTest` passes on the cluster with `Ls = 8`.
- [x] Coupled-5D CG fifth-neighbor SPD mock scaffold: add open-boundary `A = 2 I - 0.25 T_s` with host tridiagonal exact solve.
- [x] Coupled-5D CG fifth-neighbor SPD mock validation: `mdwfCoupledCGFifthNeighborTest` passes on the cluster with `Ls = 8`.
- [x] First safe MDWF-operator CG scaffold: add `mdwfCoupledCGMdwfCsw0Test` with `c_sw = 0` operator regression and zero-RHS CG early exit.
- [x] First safe MDWF-operator CG validation: `mdwfCoupledCGMdwfCsw0Test` passes on the cluster with `Ls = 8`.
- [x] Normal-operator scaffold: define `N = M^\dagger M` as an explicit composition of supplied forward and adjoint operators.
- [x] Normal-operator scaffold validation: `mdwfNormalOperatorDiagonalTest` passes on the cluster with `Ls = 8`.
- [x] Explicit MDWF adjoint scaffold: add `MDWFAdjointLinearOperator` for `M^\dagger` using gamma5-hermitian Wilson/clover and transposed fifth-direction coupling.
- [x] Raw-MDWF normal-equation solve scaffold: add `mdwfNormalMdwfCsw0SolveTest` with explicit `M^\dagger M`, `c_sw = 0`, adjoint identity check, and nonzero source.
- [x] Raw-MDWF normal-equation solve validation: `mdwfNormalMdwfCsw0SolveTest` passes on the cluster with `Ls = 8`.
- [x] `c_sw = 0` normal-equation behavior is preserved as the required regression gate before any nonzero-`c_sw` normal solve.
- [x] Nonzero `c_sw` normal-equation sanity scaffold: add `mdwfNormalMdwfNonzeroCswSanityTest` with explicit `M^\dagger M`, nonzero-adjoint identity check, and finite-response comparison against `c_sw = 0`.
- [x] Nonzero `c_sw` normal-equation sanity validation: `mdwfNormalMdwfCsw0SolveTest` and `mdwfNormalMdwfNonzeroCswSanityTest` pass on the cluster with `Ls = 8`.
- [x] Nonzero `c_sw` normal-equation CG solve scaffold: add `mdwfNormalMdwfNonzeroCswSolveTest` with explicit `M^\dagger M`, nonzero-adjoint identity check, positive-Rayleigh sanity check, and residual check.
- [x] Nonzero `c_sw` normal-equation CG solve validation: `mdwfNormalMdwfNonzeroCswSolveTest` passes on the cluster with `Ls = 8`, `c_sw = 0.5`, 35 iterations, relative residual `6.99408e-09`, and adjoint relative difference `1.24017e-16`.
- [x] Shifted normal-operator scaffold: add `MDWFShiftedNormalOperator` for explicit `(M^\dagger M + sigma) x` without RHMC/HMC integration.
- [x] Shifted normal-equation solve scaffold: add `mdwfShiftedNormalSolveTest` for `sigma = 0.1`, covering both `c_sw = 0` and `c_sw = 0.5`.
- [x] Shifted normal-equation solve validation: `mdwfShiftedNormalSolveTest` passes on the cluster with `Ls = 8`, `sigma = 0.1`, 34 iterations for both `c_sw = 0` and `c_sw = 0.5`, and exact zero-shift agreement.
- [x] Coupled-5D multishift-CG scaffold: add `MDWFCoupledMultiShiftCG` with a multi-shift interface that preserves the coupled 5D inner product and keeps `Ls` out of the RHS interpretation.
- [x] Coupled-5D multishift mock-SPD tests: add `mdwfCoupledMultiShiftCGMockTest` for `A = 2 I` and fifth-slice diagonal `A_s = 1 + 0.1 s`.
- [x] Coupled-5D multishift mock validation: `mdwfCoupledMultiShiftCGMockTest` passes on the cluster with `Ls = 8` for 4 shifts on both mock SPD operators.
- [x] Coupled-5D multishift MDWF normal scaffold: add `mdwfMultiShiftNormalMdwfTest`, comparing multishift solutions against repeated `MDWFShiftedNormalOperator` single-shift solves for shifts `{0.0, 0.1, 0.3}`.
- [x] Coupled-5D multishift MDWF normal validation: `mdwfMultiShiftNormalMdwfTest` passes on the cluster with `Ls = 8`, `c_sw = 0.5`, 3 shifts, exact agreement with single-shift solves, and max relative residual `9.0239e-09`.
- [x] Coupled-5D rational-operator scaffold: add `MDWFRationalOperator` for explicit `c0 + sum_i numerator_i / (A + shift_i)`, backed by `MDWFCoupledMultiShiftCG`.
- [x] Coupled-5D rational mock-SPD test scaffold: add `mdwfRationalMockTest` for `A = 2 I` and fifth-slice diagonal `A_s = 1 + 0.1 s`.

## Next stages

- [ ] Coupled-5D rational mock validation: compile and run `mdwfRationalMockTest` on the cluster before applying rational operators to MDWF normal systems.
- [ ] Only after the operator and solver adapter are correct, discuss RHMC integration and optional smearing.

## Stage 5 notes

- Preserve the `c_sw = 0` regression test before changing nonzero-clover behavior.
- Route clover only through the existing Wilson-kernel path; do not duplicate clover storage or alter MDWF fifth-direction coupling.
- Keep the first clover patch local to the MDWF wrapper/test layer unless an existing Wilson interface requires a minimal extension.
- The first nonzero-`c_sw` test should only verify a localized, nonzero difference from `c_sw = 0` on a nontrivial gauge field; it is not a full physics-correctness validation.

## Guardrails

- Keep MDWF changes under `src/experimental/mdwf/` plus focused tests in `src/testing/`.
- Do not use existing multi-RHS CG as a true coupled MDWF solver.
- `MDWFLinearOperator` intentionally exposes `apply()` but deletes `applyMdaggM()` until a true coupled 5D solver adapter exists.
- `MDWFCoupledSolverAdapter` must aggregate `dotProductStacked` over all `Ls` slices before any solver uses the result.
- `MDWFCoupledCG` is isolated from `src/modules/inverter/` and must not be wired into RHMC/HMC until the coupled operator is validated.
- `MDWFNormalOperator` composes supplied forward/adjoint operators; it must not be used with the raw MDWF operator as its own adjoint unless that hermiticity is explicitly proven.
- `MDWFAdjointLinearOperator` is part of the experimental scaffold; validate adjoint identities before trusting any normal-equation solve.
- Nonzero-`c_sw` normal-equation tests must preserve `mdwfNormalMdwfCsw0SolveTest` as a passing baseline and must continue using an explicitly supplied `M^\dagger M`.
- `MDWFCoupledMultiShiftCG` is a correctness-first scaffold with independent coupled-CG solves per shift; do not treat it as an optimized simultaneous multishift recurrence yet.
- `MDWFRationalOperator` consumes explicit rational coefficients only; it does not define RHMC determinant powers, pseudofermion conventions, or force terms.
- Do not touch RHMC/HMC/force/HISQ code for the operator scaffold.
- Do not change `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout for the scaffold.
