# SIMULATeQCD MDWF Development Guide for Codex

## Project Context

This repository is a local editing and inspection copy for developing Möbius Domain Wall Fermions (MDWF) in SIMULATeQCD.  The target implementation is MDWF with a clover-improved Wilson kernel.

The authoritative compile and runtime validation environment is the HPC cluster.  Do not assume a local MacBook build is valid, and do not claim that a change compiles or runs unless a cluster build/run log confirms it.

## Development Priorities

1. Physics correctness.
2. Minimal, reviewable patches.
3. Preservation of existing Wilson/domain-wall behavior when `c_sw = 0`.
4. Clear staging and frequent cluster validation.
5. Locality and debuggability over abstraction or cleverness.

## General Guardrails

- Do not modify unrelated files.
- Do not rename public interfaces, files, classes, kernels, or data layouts unless explicitly requested.
- Do not change `Spinorfield`, `GIndexer`, `SiteComm`, or global memory layout for MDWF scaffolding.
- Do not touch RHMC/HMC, force, or HISQ code unless the user explicitly asks for that layer.
- Do not introduce broad refactors while implementing MDWF pieces.
- Do not silently change conventions, signs, normalizations, layouts, or boundary behavior.
- Do not use existing multi-RHS CG as a true coupled MDWF solver; `Ls` is the physical fifth dimension, not independent right-hand sides.

## MDWF Implementation Strategy

- Keep MDWF code under `src/experimental/mdwf/`.
- Keep focused MDWF tests under `src/testing/`.
- Use `Spinorfield<floatT, onDevice, Layout, HaloDepth, 12, Ls>` as the current 5D MDWF spinor representation.
- Interpret `12` as 4 spin × 3 color.
- Interpret `Ls` as the physical fifth dimension only inside the MDWF wrapper layer.
- Preserve the existing 4D Wilson/clover path and route clover through that path.
- Keep fifth-direction coupling gauge independent.
- Validate `c_sw = 0` before trusting nonzero-clover behavior.

## Current Validated Scaffold Direction

The MDWF development path is staged:

1. 5D spinor representation.
2. Fifth-direction coupling only.
3. 4D Wilson Dslash applied slice-by-slice.
4. MDWF operator skeleton.
5. Clover through the Wilson-kernel path.
6. `c_sw = 0` regression against the unclovered path.
7. Coupled 5D solver interfaces without treating `Ls` as RHS.
8. Normal-equation, shifted, multishift, rational, pseudofermion/action scaffolds.
9. Finite-difference and scalar force-contraction checks before production force accumulation.

Do not skip staging.  One conceptual change per patch is preferred.

## Force and RHMC Boundary

- Force-related work must remain isolated until scalar contraction and finite-difference checks justify the next step.
- A validated scalar derivative is not automatically the final HMC force sign.
- Do not wire MDWF into RHMC/HMC until the operator, solver adapter, rational action, workspace, finite-difference harness, and force-accumulation conventions are deliberately reviewed.
- Optional smearing is out of scope for the first MDWF implementation unless there is an explicit physics reason.

## Patch Expectations

When changing code:

- State the purpose of the change.
- Identify affected files/functions.
- Keep the patch small and reversible.
- Prefer new MDWF wrapper/test files over changing shared infrastructure.
- Avoid speculative generalization.
- Update `TODO.md` when adding or validating a staged MDWF step.
- Mention risks, especially sign, dagger, even/odd, halo, boundary, and normalization assumptions.

## Validation Expectations

Local validation may include static inspection and diff checks only.  Cluster validation should be requested explicitly with concrete commands.

Useful cluster build pattern:

```sh
make <target> -j24
```

Focused MDWF test binaries are usually run from the cluster build `testing/` directory, for example:

```sh
./mdwfOneLinkForceAccumulatorMockTest
```

Never report cluster success unless the user provides the corresponding cluster output.

## Debugging Order

When fixing failures:

1. Compiler/runtime error message.
2. Missing includes, template instantiations, namespaces, and CMake target wiring.
3. CUDA launch configuration and indexing.
4. Spin/color/stack layout.
5. Halo and boundary assumptions.
6. Dagger/hermiticity/sign conventions.
7. Physics consistency.

Do not assume a compiled result is physically correct.
