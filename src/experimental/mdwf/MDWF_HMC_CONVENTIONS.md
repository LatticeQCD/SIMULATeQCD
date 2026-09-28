# MDWF HMC Conventions Review

Step 1 of the MDWF RHMC plan: fix, from direct reading of SIMULATeQCD's HMC
code, how the validated MDWF force matrices enter SIMULATeQCD's molecular
dynamics.  This note is design-only; the numerical check is
`mdwfMobiusHmcConventionTest`.  It does not edit `src/modules/rhmc/`,
`src/gauge/`, or any HISQ code.

## 1. SIMULATeQCD HMC conventions (confirmed from the code)

Momenta (`Gaugefield::gauss` -> `SU3::gauss`, `base/math/su3.h`): each link
momentum `P` is **Hermitian and traceless**, `P = sum_a p_a lambda_a`
(Gell-Mann `lambda_a`, `tr(lambda_a lambda_b) = 2 delta_ab`), with each
`p_a` drawn from `exp(-p_a^2)`.  That is the distribution
`exp(-(1/2) tr P^2)`, consistent with the kinetic term below.

Hamiltonian (`pure_gauge_hmc::get_Hamiltonian`):

```text
H = (1/2) sum_links tr_d(P, P) + S_g,     tr_d(A, B) = Re tr(A B)
```

Gauge action used in `Delta H` (the energy density that `Metropolis()`
reduces), with the MILC normalization noted in `gauge_force`:

```text
S_g = -(3 beta / 5) * GaugeAction::symanzik()
    = -(beta / 3) sum_x sum_{mu<nu} Re tr P_{mu nu}(x)
      + (beta / 60) sum_x (rectangles) Re tr R(x)
```

The value logged as `glue = -beta * symanzik()` differs from this by a factor
`5/3`; only the energy-density form enters `Delta H`.

Link update (`do_evolve_Q`, `integrator.cpp`): **left multiplication**

```text
U -> exp(i epsilon P) U,   followed by su3unitarize()
```

Momentum update (`do_evolve_P`):

```text
P -> P - i epsilon ipdot
```

so `ipdot = i dP/dtau` is traceless anti-Hermitian.

## 2. The force rule

Define, for any action `S`, the left-variation raw matrix `B_l` of link `l` by

```text
U_l -> exp(epsilon H) U_l:   dS/d epsilon = Re tr(H B_l),   H in su(3)
```

and `K_l = TA(B_l)` (traceless anti-Hermitian part, `SU3::TA()`).  Energy
conservation `dH/dtau = 0` with `dU_l/dtau = i P_l U_l` (that is,
`H = i P_l`) gives

```text
dH/dtau = sum_l [ tr(P_l dP_l/dtau) + Re tr(i P_l K_l) ] = 0
  =>  dP_l/dtau = -i K_l   =>   ipdot_l = K_l.
```

**Rule: `ipdot_l = K_l` exactly, with no extra sign or factor.**  `TA()` does
not change `Re tr(H B)` for `H` in su(3), so projecting once per link is exact.

Analytic cross-check against SIMULATeQCD's gauge force: the plaquette part of
`gauge_force` is `TA(r_1 * g_c1 * U V_plaq)` with
`r_1 g_c1 = (beta/3)(3/5)(-5/3) = -beta/3`, and the left-variation raw matrix
of `-(beta/3) sum Re tr P_{mu nu}` is `B = -(beta/3) U V_plaq` (with `U V_plaq`
the plaquettes starting at link `l`); the six rectangle staples carry
`r_1 g_c2 = +beta/60`, matching `+(beta/60) sum Re tr R`.  So on paper
`gauge_force = TA(B) = K` follows this rule.  The numerical check (Section 5)
confirms the rule for the MDWF force but found the full Symanzik identity off
by 2.4%, which is under investigation.

## 3. What this means for the MDWF force

The single-rank all-link storage
(`overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero`, on an
`MDWFMobiusForceWorkspaceView` for Möbius) writes exactly `K_l = TA(B_W + B_C)`
in this left convention, validated against finite differences with
`dS(H) = Re tr(H K_l)`.  Therefore:

- the MDWF fermion contribution to `ipdot` is the stored `K_l`, unchanged;
- total `ipdot = ipdot_gauge + ipdot_fermion` (linear), or each force applied
  with its own `P -> P - i epsilon ipdot` step, as the existing integrator
  does;
- the rational numerators already carry the `-2 a_i` weights; no extra factor
  of 2 (the HISQ `TA(2 U F)` expression is not used) and no extra link
  multiplication.

Two-flavour pseudofermion `S_f = phi^\dagger (M^\dagger M)^{-1} phi` is the
rational case `c0 = 0`, numerator `{1}`, shift `{0}`; its heatbath is
`phi = M^\dagger eta` with `eta` distributed as `exp(-eta^\dagger eta)`.
`Spinorfield::gauss` (`Vect::gauss`, `base/math/vect.h`) gives each complex
component `|z|^2` exponentially distributed with mean 1, which is that
distribution, so `<eta^\dagger eta> = 12 * Ls * V`.  A wrong noise
normalization would not show up in `Delta H` or reversibility tests (it
changes the sampled determinant power, not energy conservation), so it is
checked explicitly.

## 4. Consequences and constraints for the MDWF HMC driver

- `do_evolve_Q`/`do_evolve_P` are file-local functors in `integrator.cpp` and
  the integrator's update methods are private; an MDWF driver under
  `src/experimental/mdwf/` reproduces these two formulas exactly (they are
  three lines each) rather than editing `src/modules/rhmc/`.
  `gauge_force` (`gaugeActionDeriv.h`), `GaugeAction::symanzik()`, and
  `Gaugefield::gauss` are header/library functions and are reused directly.
- The gauge action is SIMULATeQCD's tree-level Symanzik action.  RBC/UKQCD
  MDWF ensembles typically use Iwasaki or DBW2; choosing the production gauge
  action is a separate physics decision.
- The MDWF force storage is single-rank and host-side; the first HMC is
  therefore single-GPU, small-volume validation only.  A GPU force kernel and
  MPI ownership are later steps.

## 5. First cluster run (commit `0ff7e24` plus the uncommitted test)

- Fermion rule confirmed: `dS_f/dtau + sum tr(P(-i K)) = 0` to relative
  `1.25e-07` (control, `M5 = -2`) and `3.00e-07` (physical-like,
  `M5 = 1.8`, `mf = 0.05`), `b5 = 1.5`, `c_sw = 0.5`.  **The stored MDWF
  matrices are the fermion `ipdot` with no extra sign or factor.**
- Momenta Hermitian (violation `0`) and traceless (`4.4e-16`); noise
  `<eta^\dagger eta> / (12 Ls V) = 0.997706` (expected `1 +- 0.0028`).
- **Open:** the full Symanzik identity with `gauge_force` and
  `S_g = -(3 beta/5) symanzik()` gave `dS_g/dtau = -759.438` versus
  `sum tr(P(-i gauge_force)) = 777.655` (relative `2.4e-2`): right sign, 2.4%
  mismatch, although a line-by-line reading finds the plaquette and all six
  rectangle staples with coefficients matching the action.  The test was
  split to localize it: the plaquette identity with the independent
  `gaugeActionDerivPlaq` now gates the convention reading, and the rectangle
  part (`gauge_force` minus the plaquette force), the full identity, and the
  comparison `gauge_force = -(beta/5) symanzikGaugeActionDeriv` are reported
  as diagnostics.  The MDWF HMC driver must not use the Symanzik gauge action
  until this is resolved; a plaquette (Wilson) gauge action whose identity is
  confirmed is the fallback.

Second run (split gauge check): the plaquette identity holds to relative
`5.7e-08` (`-799.931` versus `799.931`), confirming the convention reading.
The whole Symanzik discrepancy sits in the rectangle term: the rectangle
action rate is `+40.493`, but the rectangle part of `gauge_force` predicts
`-22.276`, a non-uniform `~0.55` ratio rather than a normalization factor.
`gauge_force` and `symanzikGaugeActionDeriv` agree to `2.7e-15`.  Until the
rectangle term is localized and resolved, the MDWF HMC driver uses the
plaquette (Wilson) gauge action.

**Resolved (`mdwfSymanzikGaugeForceTest`, job `34554300`).** The mismatch is
a host-only indexer bug, not a force error: `GIndexer::site_up_2dn(s, mu, nu)`
is `site_move<1, -2>` (`s + mu - 2 nu`) on the GPU, but its host fallback is
`site_up_dn_dn(s, mu, mu, nu)` (`s - nu`). `gauge_force` uses it in the one
1x2 rectangle staple below the link, and the convention test evaluated
`gauge_force` through a host accessor. Evaluated on the device, as
SIMULATeQCD's HMC does, `gauge_force` is exact: `dS_g/dtau = -759.438` versus
`759.438` (relative `5.3e-8`), rectangle part `40.493` versus `-40.493`
(`1.5e-7`). The host force plus a correction for exactly that staple equals
the device force to `2.8e-15`, and reproduces the old host numbers (`777.655`,
rectangle `-22.2759`). The MDWF HMC driver therefore offers the Symanzik
action (`MDWFHmcParameters::symanzik_gauge`) with `gauge_force` evaluated on
the device only. Never evaluate `gauge_force` or `symanzikGaugeActionDeriv`
on the host until `site_up_2dn`'s host path is fixed upstream.

## 6. Numerical confirmation (`mdwfMobiusHmcConventionTest`)

For Gaussian `P` from `Gaugefield::gauss`, along `U -> exp(i epsilon P) U`:

1. Gauge: `dS_g/dtau` (centered difference of `-(3 beta/5) symanzik()`) plus
   `dT/dtau = sum tr(P (-i ipdot_g))`, `ipdot_g = gauge_force`, must vanish.
   This checks the reading of SIMULATeQCD's conventions against its own
   validated gauge force.
2. Fermion: `dS_f/dtau` (centered difference of the Möbius clover rational
   action) plus `sum tr(P (-i K))` from the stored MDWF matrices must vanish.
3. `P` is Hermitian and traceless, and `<eta^\dagger eta> / (12 Ls V) = 1`
   within statistical error.
