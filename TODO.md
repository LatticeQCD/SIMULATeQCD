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

## Next stages

- [ ] Validate `mdwfOperatorWorkspaceTest` on the cluster.
- [ ] Stage 7: only after the operator is correct, discuss solver/RHMC integration and optional smearing.

## Stage 5 notes

- Preserve the `c_sw = 0` regression test before changing nonzero-clover behavior.
- Route clover only through the existing Wilson-kernel path; do not duplicate clover storage or alter MDWF fifth-direction coupling.
- Keep the first clover patch local to the MDWF wrapper/test layer unless an existing Wilson interface requires a minimal extension.
- The first nonzero-`c_sw` test should only verify a localized, nonzero difference from `c_sw = 0` on a nontrivial gauge field; it is not a full physics-correctness validation.

## Guardrails

- Keep MDWF changes under `src/experimental/mdwf/` plus focused tests in `src/testing/`.
- Do not use existing multi-RHS CG as a true coupled MDWF solver.
- Do not touch RHMC/HMC/force/HISQ code for the operator scaffold.
- Do not change `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout for the scaffold.
