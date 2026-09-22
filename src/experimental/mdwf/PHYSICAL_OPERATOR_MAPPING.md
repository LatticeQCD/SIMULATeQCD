# MDWF Physical Operator Mapping Review

This note is design-only. It maps the existing gauge-independent fifth-direction
stencil (`MDWFFifthDimCoefficients`) and the per-slice Wilson-kernel `mass` to
the physical domain-wall parameters `M5`, `mf`, `b5`, `c5`, following the
`M(mf; theta)` notation and staged plan already recorded in
`MDWF_DETERMINANT_FACTORIZATION.md`. It implements stage 1 of that plan's
"Staged implementation and acceptance gates" section:

> Review the physical operator mapping: distinguish kernel height from wall
> mass; define `b5/c5`, projector/wall signs, normalization, spacetime phases,
> and same-kernel PV.

No code changes are made here. The existing fifth-direction helper, Wilson
kernel, and all currently-passing MDWF tests are unaffected. No production
`M5`, `mf`, `b5`, `c5`, or spacetime boundary condition is selected by this
document; it records what is already fixed by the existing code and proposes
a mapping for review before any implementation.

## 1. What the existing code already fixes (confirmed by direct inspection)

### 1.1 Spinor basis and chirality

`Gamma5MultVec` (`src/experimental/fullSpinor.h`) is `diag(+1, +1, -1, -1)` on
the four spin indices, i.e. a chiral (Weyl) basis where spin indices `{0, 1}`
span the `gamma5 = +1` eigenspace and `{2, 3}` span the `gamma5 = -1`
eigenspace. `Vect12` packs `(spin, color)` with color varying fastest inside
each spin block, so `Vect12` components `0..5` are spin `{0,1}` (all colors)
and `6..11` are spin `{2,3}` (all colors).

`mdwfProjectPlus` (`MDWFFifthDim.h`) keeps components `0..5` and
`mdwfProjectMinus` keeps components `6..11`. Therefore:

```text
mdwfProjectPlus  == P_+ == (1 + gamma5) / 2
mdwfProjectMinus == P_- == (1 - gamma5) / 2
```

This already matches the standard domain-wall projector convention; no change
is proposed here.

### 1.2 Fifth-direction stencil topology

`MDWFFifthDimCoupling::operator()` computes, for slice `s`:

```text
(D5 psi)_s = diagonal * psi_s
           + forward_coeff  * P_- psi_{s_forward}
           + backward_coeff * P_+ psi_{s_backward}
```

with `forward_coeff = forward_boundary` only at `s = Ls - 1` (wrapping to
`s_forward = 0`), and `backward_coeff = backward_boundary` only at `s = 0`
(wrapping to `s_backward = Ls - 1`); otherwise the interior `forward_hop` /
`backward_hop` values are used. This topology (forward neighbor entering
through `P_-`, backward neighbor through `P_+`, with distinct wraparound
coefficients) matches the standard Shamir/Möbius fifth-direction stencil
structure. No change is proposed here.

### 1.3 Per-slice Wilson kernel mass normalization

`gamma5DiracWilson` / `DiracWilsonEvenOdd2` (`src/experimental/DWilson.h`)
compute, per 4D site with `r = 1`:

```text
(D_W psi)(x) = 2 * mass * psi(x)
             - (1/2) * sum_mu [ (1 - gamma_mu) U_mu(x)   psi(x+mu)
                               + (1 + gamma_mu) U_mu(x-mu)^dagger psi(x-mu) ]
```

For `U = 1` this gives, in momentum space,

```text
D_W(p) = [2 * mass - sum_mu cos(p_mu)] + i * sum_mu gamma_mu sin(p_mu)
```

so `D_W(p=0) = 2 * mass - 4`. Comparing to the standard Wilson-fermion form
`D_W(p) = (m_std + 4r) - r * sum_mu cos(p_mu) + i * sum_mu gamma_mu sin(p_mu)`
(`r = 1`), the existing `mass` argument relates to the standard bare Wilson
mass `m_std` by

```text
mass = (m_std + 4) / 2                      i.e.   m_std = 2 * mass - 4
```

This is an existing normalization fact, not a proposal; it is required to
translate `M5` into the `mass` argument below. `applyMDWFWilsonSlice` applies
`gamma5DiracWilson` then `gamma5` again, so the net per-slice kernel is
`D_W` itself (the two `gamma5` multiplications cancel); the clover path
(`applyMDWFCloverWilsonSlice`) applies the equivalent even/even + even/odd
split directly. Both paths use the same `mass`/`csw` normalization.

## 2. Proposed mapping

**Correction (superseding the first version of this document):** the first
version of this section claimed that general Möbius (`c5 != 0`) "does not fit
the current gauge-independent fifth stencil" and "would require a new,
gauge-dependent fifth-direction coupling." That claim was wrong, and was
reached without checking an actual reference implementation. Section 2.3
below replaces it, after reading the real RBC/UKQCD/Grid Möbius source
(`Grid/qcd/action/fermion/implementation/CayleyFermion5DImplementation.h`,
`paboyle/Grid`, functions `M5D`, `Meooe5D`, `M`, fetched 2026-09-22). General
Möbius does *not* need `MDWFFifthDimCoupling` itself to become gauge
dependent; it needs one extra gauge-independent fifth-direction mixing step
*before* the existing per-slice `D_W` kernel, using the same coupling
mechanism already validated by `mdwfFifthDimTest`. See Section 2.3.

### 2.1 Shamir special case (`b5 = 1`, `c5 = 0`)

The Shamir kernel is the special case of Section 2.3 with `b5 = 1`, `c5 = 0`.
It remains useful as an exact regression target for the general
construction (Section 2.3 reduces to it identically when `c5 = 0`), and is
implemented separately in `MDWFPhysicalMapping.h` /
`mdwfShamirFifthDimMappingTest`. Using the standard Furman-Shamir domain-wall
operator,

```text
(M psi)(x,s) = D_W(x; -M5) psi_s(x)
             + psi_s
             - P_- psi_{s+1}   (interior, no wrap)
             - P_+ psi_{s-1}   (interior, no wrap)
             + mf * P_- psi_0       at s = Ls - 1  (forward wrap)
             + mf * P_+ psi_{Ls-1}  at s = 0        (backward wrap)
```

the proposed mapping onto the existing scaffold inputs is:

```text
mass              = 2 - M5 / 2          (from Section 1.3, m_std = -M5)
diagonal          = 1
forward_hop       = -1
backward_hop      = -1
forward_boundary  = +mf
backward_boundary = +mf
```

Only the sign of the wraparound coefficients differs from the interior hop;
that sign flip is the defining structural feature of the domain-wall
boundary condition (it is what localizes the light chiral mode at the two
walls). `M5` is conventionally in `(0, 2)` (commonly `~1.8` in this
normalization); `mf` is the physical quark mass in lattice units.

### 2.2 Pauli-Villars endpoint

`MDWF_DETERMINANT_FACTORIZATION.md` records the reference convention
`mf = 1` for the Pauli-Villars regulator. At `mf = 1` the wraparound
coefficients (`+1`) have the same magnitude as the interior hops (`-1`, only
opposite sign), so the fifth-direction operator loses the asymmetric
wall structure that produces a light mode; this is the expected PV behavior
and is a consistency check on the mapping above, not a new convention.

### 2.3 General Möbius (RBC/UKQCD convention), by composition

Grid's `CayleyFermion5D<Impl>::M` (unpreconditioned Möbius/DWF Dslash)
assembles the operator as (transcribed from the functions named above; `Din`
is a temporary 5D field, `DW` is the per-slice 4D Wilson kernel at kernel
height `M5`, `Pminus`/`Pplus` are `mdwfProjectMinus`/`mdwfProjectPlus`):

```text
Din_s = bs * psi_s + cs * Pminus(psi_{s+1}) + cs * Pplus(psi_{s-1})     (interior)
      at s = Ls-1: the Pminus(psi_0) coefficient is -mass_minus * cs, not +cs
      at s = 0:    the Pplus(psi_{Ls-1}) coefficient is -mass_plus * cs, not +cs

chi_s = DW(Din)_s + psi_s
      - Pminus(psi_{s+1}) - Pplus(psi_{s-1})                            (interior)
      at s = Ls-1: the Pminus(psi_0) coefficient is +mass_minus, not -1
      at s = 0:    the Pplus(psi_{Ls-1}) coefficient is +mass_plus, not -1
```

with `mass_plus = mass_minus = mf` for the standard (non-Hasenbusch) case,
and, for plain (non-`z`) Möbius, constant `bs = b5`, `cs = c5` (the
`zMobius` generalization allows `s`-dependent `bs[s]`/`cs[s]` via a
Zolotarev-type rescaling; that is not covered here). This is exactly the
`D_plus(s) = 1 + b5 D_W`, `D_minus(s) = c5 D_W - 1` structure already named
in `MDWF_DETERMINANT_FACTORIZATION.md`, just regrouped so that `D_W` is
applied once, after linearly combining neighboring slices, rather than
appearing algebraically inside each hop coefficient.

Crucially, the `Din` construction above and the final `chi` shift term are
each, on their own, an instance of the *existing* gauge-independent
`MDWFFifthDimCoefficients` stencil — the same stencil validated by
`mdwfFifthDimTest` — just with two different coefficient sets:

```text
Din coefficients (feed to the existing MDWFFifthDimCoupling):
  diagonal          = b5
  forward_hop       = c5
  backward_hop      = c5
  forward_boundary  = -mf * c5
  backward_boundary = -mf * c5

Final shift coefficients: exactly mdwfShamirFifthDimCoefficients(mf)
  diagonal = 1, forward_hop = -1, backward_hop = -1,
  forward_boundary = backward_boundary = mf
```

So the full Möbius forward operator is:

```text
Din  = applyMDWFFifthDimCoupling(psi, DinCoefficients)
chi  = applyMDWFWilsonSlice(Din, mass = mdwfShamirKernelMass(M5), csw)
chi += applyMDWFFifthDimCoupling(psi, mdwfShamirFifthDimCoefficients(mf))
```

`mdwfShamirFifthDimCoefficients(mf)` has `diagonal = 1`, so this second
coupling call already produces `psi + [shift terms]` in one combined output
(exactly Grid's `axpby(chi,1,1,chi,psi)` step followed by `M5D(psi,chi)`'s
`diag = 1` regrouped by associativity into a single call); `psi` must be
added exactly once, either via a separate step with a `diagonal = 0` shift
coefficient set or, as above, folded into this one call — never both, or it
is double-counted.

**No change to `MDWFFifthDimCoupling`, `MDWFWilsonSlice`, or any existing
gauge-independence guardrail is needed.** `MDWFFifthDimCoupling` stays
exactly as validated; it is simply called twice with different constant
coefficients, with the existing, unmodified per-slice `D_W` kernel applied
to the result of the first call. At `b5 = 1`, `c5 = 0`, `Din` reduces
exactly to `psi` (`forward_hop = backward_hop = 0`, boundaries `= 0`), so
`DW(Din) = DW(psi)` and the whole construction reduces identically to the
Shamir case in Section 2.1 — an exact, checkable regression target.

`b5 - c5 = 1` (RBC/UKQCD convention, per project decision) is a convention
choice, not a mathematical requirement of the construction above; it fixes
one degree of freedom and leaves the "Möbius scale" `b5 + c5` free. This
document does not select a numerical `b5`/`c5`/scale; that remains a
separate physics decision. Clover (`csw != 0`) is not addressed in this
section; the `c_sw = 0` composition above should be validated first, per
the project's existing staging discipline.

Grid's `M()` also has a `// add i q_mu gamma_mu here` step (`addQmu`) for an
optional twisted-mass-like term; that is not part of the standard Möbius
construction used here and is not implemented.

## 3. What this document does not establish

- It does not implement the mapping in code (no `mass`/coefficient
  computation helper exists yet).
- It does not fix a numerical `M5` or `mf`, or a production `Ls`.
- It does not address spacetime boundary conditions/phases (periodic,
  antiperiodic, or twisted) for the 4D directions; those are independent of
  the fifth-direction mapping above and remain unreviewed.
- It does not validate the mapping against gauge-dependent domain-wall
  physics (chiral zero mode, residual mass, spectral flow); that needs
  propagator/eigenvalue computations and is a separate, later, more
  expensive validation step.
- It does not change force, RHMC, or HMC code, and does not authorize
  treating any resulting operator as production-ready.

## 4. Implemented next step, and a rejected check

A `gamma5`/`R5` Hermiticity check
(`M^dagger = R5 gamma5 M gamma5 R5`, where `R5` reflects `s -> Ls - 1 - s`)
against `MDWFAdjointOperator.h` was the first idea for validating this
mapping, but direct derivation shows it has no discriminating power here:
`MDWFFifthDimAdjointCoupling` is built as the explicit transpose-adjoint of
`MDWFFifthDimCoupling` for *any* five scalar coefficients (its own file
comment says so), so the block-matrix computation
`(D5)^dagger = R5 D5 R5` holds for arbitrary `(diagonal, forward_hop,
backward_hop, forward_boundary, backward_boundary)` — including wrong or
asymmetric ones — not only for the Shamir values from Section 2.1. It is a
correct statement about the existing adjoint scaffold's self-consistency
(already established generically by the normal-equation adjoint-identity
tests), not an independent check on whether *these particular numbers* are
the Shamir operator.

What was implemented instead, in `MDWFPhysicalMapping.h` (the `(M5, mf) ->
(mass, diagonal, forward_hop, backward_hop, forward_boundary,
backward_boundary)` helper from Section 2.1) and `mdwfShamirFifthDimMappingTest`:

1. An independent recomputation of the mapping formulas directly from this
   document's Section 1.3/2.1 text, compared against the header, to catch a
   transcription mistake between the two.
2. A direct check of the `mf = 1` PV magnitude-symmetry property from
   Section 2.2 on the produced coefficients, and that a generic
   physical-like `mf` does not share it.
3. A wiring check that feeds the Shamir-mapped coefficients through the
   already-validated `applyMDWFFifthDimCoupling` and the fifth-direction
   stencil oracle from `mdwfFifthDimTest`, with a deliberately wrong
   boundary sign as a sensitivity control (confirming the comparison would
   actually catch that class of mistake).

This is still a design-stage arithmetic/wiring check, not a gauge-dependent
physics validation; it does not touch clover, CG, RHMC/HMC, or force code,
and does not change `MDWFFifthDimCoupling`, `MDWFOperator`, or this header's
own behavior once written. A later, more expensive step (propagator or
eigenvalue-based, at `c_sw = 0` first, then with clover) remains the way to
validate the mapping against actual domain-wall physics rather than just
its arithmetic and wiring.

## 5. General Möbius implementation (this patch)

Following Section 2.3, `MDWFMobiusMapping.h` adds:

- `mdwfMobiusDinCoefficients(b5, c5, mf)`: the `Din`-construction
  coefficients from Section 2.3, built on the existing
  `MDWFFifthDimCoefficients`/`MDWFFifthDimCoupling` — no new coupling class.
- `MDWFMobiusOperatorParameters(M5, mf, b5)`: bundles `mass =
  mdwfShamirKernelMass(M5)` (unchanged from Section 1.3, independent of
  `b5`/`c5`), `c5 = b5 - 1` (enforcing the project's chosen RBC/UKQCD
  `b5 - c5 = 1` convention), `dinCoeff`, and `shiftCoeff =
  mdwfShamirFifthDimCoefficients(mf)`.
- `applyMDWFMobiusOperator(...)`: the three-step composition from Section 2.3
  (`Din`, `DW(Din)`, `+= shift` — where `shift` already includes `psi` once,
  via `shiftCoeff.diagonal = 1`), at `c_sw = 0` only, reusing
  `applyMDWFFifthDimCoupling` and `applyMDWFWilsonSlice` unmodified. `Din`'s
  halo is refreshed before the Wilson kernel reads its spatial neighbors.

`mdwfMobiusFifthDimMappingTest` validates:

1. At `b5 = 1` (so `c5 = 0`), `Din` reduces exactly to `psi`, and
   `applyMDWFMobiusOperator` reproduces `applyMDWFOperator` fed
   `mdwfShamirFifthDimCoefficients`/`mdwfShamirKernelMass` bit-for-bit — the
   exact-regression check promised in Section 2.3.
2. At a generic `b5` (with `c5 = b5 - 1`), the output is detectably
   different from the `b5 = 1` baseline, confirming `b5`/`c5` are actually
   wired in and not silently ignored.

This does not implement `zMobius` (`s`-dependent `bs[s]`/`cs[s]`), clover
(`csw != 0`) for the Möbius path, or any numerical `M5`/`mf`/`b5` production
choice. It does not change `MDWFFifthDimCoupling`, `MDWFWilsonSlice`,
`MDWFOperator`, or `MDWFPhysicalMapping.h`. It does not validate against
gauge-dependent domain-wall physics (see Section 3); that remains a later,
more expensive step. Cluster-validated (`Ls = 8`, `M5 = 1.8`, `mf = 0.05`):
`maxDinDiff = 0`, `maxShamirRegressionDiff = 0`, `maxGenericDetectionDiff =
69.3069` at `b5 = 1.5`; see `TODO.md`.

## 6. General Möbius adjoint (this patch)

`MDWFMobiusMapping.h` also adds the adjoint, by the same composition
principle: since `M = (D_W . Din) + Shift`,

```text
M^dagger = Din^dagger . D_W^dagger + Shift^dagger
```

`D_W^dagger = gamma5 D_W gamma5` is the standard Wilson gamma5-Hermiticity
already used by `MDWFAdjointOperator.h`. `Din^dagger` and `Shift^dagger`
both reuse the existing, already-validated `MDWFFifthDimAdjointCoupling`
(the formal transpose-adjoint of `MDWFFifthDimCoupling` for *any* five
coefficients) applied to `dinCoeff` and `shiftCoeff` respectively. No new
adjoint machinery is introduced: `applyMDWFMobiusAdjointOperator`,
`MDWFMobiusAdjointOperatorWorkspace`, and the `MDWFMobiusLinearOperator`/
`MDWFMobiusAdjointLinearOperator` wrapper pair (matching the existing
`MDWFLinearOperator`/`MDWFAdjointLinearOperator` interface so they plug
into `MDWFCoupledSolverAdapter`/`MDWFNormalOperator` later) are all built
from it.

At `b5 = 1` (`c5 = 0`), `Din^dagger` is the identity (same reasoning as the
forward operator's `Din`), so this reduces exactly to the existing Shamir
adjoint (`MDWFAdjointLinearOperator` / `applyMDWFAdjointCloverOperator` at
`c_sw = 0`).

`mdwfMobiusAdjointMappingTest` validates:

1. At `b5 = 1`, the general adjoint reproduces the existing Shamir adjoint
   bit-for-bit — the adjoint counterpart of Section 5's forward-operator
   regression check.
2. At a generic `b5 = 1.5`, the coupled-5D adjoint identity
   `<x, M y> = <M^dagger x, y>` holds, using the same aggregated dot product
   (`MDWFCoupledSolverAdapter::dotProduct5D`) already validated for the
   Shamir/normal-operator scaffold.

This does not implement `zMobius`, clover for the Mobius path, or any
CG/RHMC/HMC wiring; it does not validate against gauge-dependent
domain-wall physics. Cluster-validated (`n2dgx01`, commit `fb95e13`,
`Ls = 8`, `M5 = 1.8`, `mf = 0.05`): `mass = 1.1`, `maxAdjointRegressionDiff`
(`b5 = 1` vs Shamir adjoint) `= 3.19744e-14`, `adjointRelDiff` (`b5 = 1.5`
coupled-5D identity) `= 9.85783e-17`; see `TODO.md`.

## 7. General Möbius clover extension (this patch)

Following the existing clover guardrail ("route clover only through the
existing Wilson-kernel path; do not duplicate clover storage or alter MDWF
fifth-direction coupling"), `MDWFMobiusMapping.h` adds
`applyMDWFMobiusCloverOperator` / `MDWFMobiusCloverOperatorWorkspace` /
`MDWFMobiusCloverLinearOperator` and their adjoint counterparts
(`applyMDWFMobiusAdjointCloverOperator` /
`MDWFMobiusAdjointCloverOperatorWorkspace` /
`MDWFMobiusCloverAdjointLinearOperator`), purely alongside the existing
`c_sw = 0` classes from Sections 5–6, which are left completely untouched.

The change is minimal: in the forward operator, the single `D_W(Din)` step
now calls the existing `applyMDWFCloverWilsonSlice` instead of
`applyMDWFWilsonSlice`, with an explicit `csw`; in the adjoint, the
`D_W^dagger(gamma5(x)) = gamma5(D_W(gamma5(x)))` step does the same. `Din`
/ `Din^dagger`, the fifth-direction shift term, and
`MDWFFifthDimAdjointCoupling` (which has no `csw` dependence) are
unchanged.

Two properties make this a strong, independently checkable extension
rather than a new derivation:

- At `c_sw = 0`, `applyMDWFCloverWilsonSlice` is already proven to
  reproduce `applyMDWFWilsonSlice` exactly, generically in its input (the
  Stage 5/6 `mdwfCloverCsw0Test` regression); so the new clover-capable
  Mobius classes at `csw = 0` are expected to reduce exactly to the
  existing, already cluster-validated plain Mobius classes at any `b5`.
- At `b5 = 1` (`c5 = 0`), `Din`/`Din^dagger` still reduce to the identity
  (unchanged from Section 5/6), so the clover Mobius forward/adjoint
  operators at `b5 = 1` must reduce exactly to the plain Shamir clover
  operators (`MDWFLinearOperator` / `MDWFAdjointLinearOperator`, which
  already route every `apply()` through the clover-capable path at any
  `csw`) at the same `csw`.

`mdwfMobiusCloverMappingTest` checks both reductions explicitly (rather
than assuming them), plus a nonzero-`c_sw` finite-response sanity check
at a generic `b5` and the coupled-5D adjoint identity
`<x, M y> = <M^dagger x, y>` at a generic `b5` and nonzero `csw`. This does
not implement `zMobius`, CG/RHMC/HMC wiring, or force code, and does not
validate against gauge-dependent domain-wall physics. Cluster-validated
(`n2dgx01`, `Ls = 8`, `M5 = 1.8`, `mf = 0.05`, `c_sw = 0.5`): `c_sw = 0`
forward/adjoint regression diffs `5.68434e-14` / `4.9738e-14`, `b5 = 1`
Shamir-clover forward/adjoint regression diffs `0` / `0`, `c_sw` response
`14.2722`, adjoint identity relative difference `1.49694e-16`; see `TODO.md`.
