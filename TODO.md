# MDWF TODO

## Current status

- [x] Stage 1: represent MDWF 5D spinors with `Spinorfield<..., 12, Ls>`.
- [x] Stage 2: implement fifth-direction-only coupling.
- [x] Stage 2 validation: `mdwfFifthDimTest` passes on the cluster for `All`, `Even`, and `Odd` layouts with `Ls = 8`.
- [ ] Stage 3 validation: compile and run `mdwfWilsonSliceTest` on the cluster.

## Next stages

- [ ] Stage 3: apply the existing 4D Wilson kernel independently on each fifth-dimensional slice.
- [ ] Stage 4: combine 4D Wilson slice application with fifth-direction coupling into an MDWF operator skeleton.
- [ ] Stage 5: add clover through the Wilson-kernel path.
- [ ] Stage 6: validate `c_sw = 0` against the unclovered Wilson/domain-wall path.
- [ ] Stage 7: only after the operator is correct, discuss solver/RHMC integration and optional smearing.

## Guardrails

- Keep MDWF changes under `src/experimental/mdwf/` plus focused tests in `src/testing/`.
- Do not use existing multi-RHS CG as a true coupled MDWF solver.
- Do not touch RHMC/HMC/force/HISQ code for the operator scaffold.
- Do not change `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout for the scaffold.
