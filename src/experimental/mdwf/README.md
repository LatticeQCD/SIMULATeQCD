# MDWF Scaffold

This directory is for a minimal Mobius/domain-wall Wilson-kernel prototype.  The first scaffold only names the 5D spinor representation; it does not implement an MDWF operator, fifth-direction coupling, solver integration, boundary-condition handling, or clover physics.

## 5D spinor representation

`MDWFSpinor<floatT, onDevice, Layout, HaloDepth, Ls>` is an alias for:

```cpp
Spinorfield<floatT, onDevice, Layout, HaloDepth, 12, Ls>
```

The `12` components are the Wilson spin-color components, `4 spin x 3 color`.  The final template parameter uses the existing `Spinorfield` stack storage, but inside this MDWF layer it represents the physical fifth dimension `Ls`, not independent right-hand sides.

This choice keeps the first MDWF patch local:

- It reuses existing `Spinorfield`, `gSiteStack`, halo exchange, precision, and even/odd storage.
- It does not change `GIndexer`, `SiteComm`, `Spinorfield`, or the global memory layout.
- It allows existing Wilson/clover kernels to be inspected for slice-by-slice reuse before adding MDWF physics.

## Solver warning

Existing multi-RHS CG treats `NStacks` as independent right-hand sides and performs stack-wise reductions and coefficients.  A coupled MDWF operator must not use that solver path as a true 5D solver until a 5D vector algebra layer reduces over both 4D sites and the fifth dimension.

## Planned stages

1. Stage 1: 5D spinor representation.
2. Stage 2: fifth-direction coupling only.
3. Stage 3: 4D Wilson Dslash slice-by-slice.
4. Stage 4: MDWF operator skeleton.
5. Stage 5: clover through Wilson path.
6. Stage 6: `c_sw = 0` validation.

## Proposed first smoke test

If a later patch adds a non-intrusive experimental target, it should only instantiate a small field such as `MDWFSpinor<double, true, All, 2, 8>` and verify construction/indexing.  It should not call CG, Wilson Dslash, clover, RHMC/HMC, force code, or any physics operator.
