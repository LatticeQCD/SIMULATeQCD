# MDWF Physical/PV/Hasenbusch Determinant Contract Review

Status: design-only review, not an approved production physics convention.
This note follows the cluster-validated single-rank Wilson/clover contraction
storage gates. It changes no operator, solver, force, parameter parser,
communication, integrator, or RHMC implementation. The original documentation
step claimed no runtime validation. Existing `c_sw = 0` behavior remains the
regression baseline.

Follow-up isolated scaffold: `MDWFDeterminantFactorMetadata.h` implements only
test labels, explicit determinant exponents, and exact mass-ladder validation.
`mdwfDeterminantFactorizationMockTest` checks scalar/commuting diagonal
telescoping and exact heatbath/action powers for arbitrary positive mock
eigenvalues. Equal masses require an explicit test-only policy. Neither the
opaque operator identity nor the boundary-mass labels establish a physical
operator mapping. The user-supplied cluster run on `n2dgx01` reports
`MDWF determinant factorization mock test passed`. Rational sets, spectral
intervals, integrator assignment, and noncommuting ratio applications remain
deferred.

Build the focused target with `make mdwfDeterminantFactorizationMockTest -j24`
in the cluster build directory, then run
`./mdwfDeterminantFactorizationMockTest` from its `testing/` directory. No
parameter file or MPI initialization is required for this scalar test. Expected
coverage is eight ladders, four diagonal components, two identity factors, and
29 rejected metadata cases, including a one-ULP continuity gap. The supplied
runtime output contains only the success message; no separate build log or
numerical diagnostics were supplied, so no quantitative error is recorded.

### Dense noncommuting follow-up (mock cluster runtime validated)

`mdwfDenseHpdDeterminantRatioMockTest` uses arbitrary nonsingular complex 2x2
operators and the mathematical representative `C = Q^dagger Q`,
`Q = Ma Mb^-1`. It checks determinant equality against both normal determinants
and `|det Ma|^2 / |det Mb|^2`, agreement with the congruence form, positive
eigenvalues, nonzero normal-operator commutators, and non-Hermiticity of the
naive quotient `Na Nb^-1`. The naive quotient is never used for spectral powers.

Exact 2x2 spectral-projector powers test `p = 0.5, 1, 1.5`, covariance
`H H^dagger = C^p`, full-matrix `H^dagger C^-p H = I`, and deterministic
complex-vector actions. A rotated known spectrum independently checks the
power helper; direct inversion and Cholesky covariance provide independent
`p = 1` references. Equal operators give identity with nonzero constant action.
Only determinants, not products of HPD factor matrices or factor actions,
are asserted to telescope. Invalid inputs and a reversed heatbath sign are
negative controls. The helper's absolute singularity thresholds are for these
well-conditioned tiny mocks only, not a production numerical policy.

No dense-mock representation is added to the earlier diagonal-only metadata
enum, and no physical operator mapping or production ratio is approved.
There are no gauge fields, halos, even/odd factors, rational approximations,
iterative solves, RNG distribution tests, derivatives, HMC signs, or MPI tests.
Build `make mdwfDenseHpdDeterminantRatioMockTest -j24` in the cluster build
directory and run `./mdwfDenseHpdDeterminantRatioMockTest` from `testing/`.
Expected coverage is five factors (four nontrivial, one identity), fifteen
power cases, and six rejected inputs; a numerical success line is printed
directly to standard output.

The user-supplied run on `n2dgx01` reports success with all five factors,
fifteen power cases, and six rejected inputs. Minimum eigenvalue is
`0.47293599833893374`, minimum normal commutator norm is `1.5238142167942526`,
and minimum naive-quotient Hermiticity violation is `0.063691378215171832`.
Maximum ratio-determinant, factor-matrix, and heatbath/action differences are
`4.4408920985006262e-16`, `4.9747569773351775e-16`, and
`1.3322676295501878e-15`, respectively. No separate build log was supplied.
This validates the controlled dense algebra, not a production ratio,
physical/PV operator mapping, rational/iterative application, determinant
phase treatment, or MPI ownership.

### Dense rational application follow-up (mock cluster runtime validated)

The separate `mdwfDenseHpdRationalApplicationMockTest` target enables an
isolated compile-time test mode in the existing dense test source. It reruns
the exact dense gate before the rational tests, without moving private helpers
into a new public interface or changing the original exact-only target path.
It uses the existing explicit coefficient adapter, but direct dense inverses,
not `MDWFRationalOperator::apply()` or coupled/multishift CG.

For `0 < beta < 1`, substitution in the
[beta integral](https://dlmf.nist.gov/5.12.E3), together with the
[reflection formula](https://dlmf.nist.gov/5.5.E3), gives
`x^-beta = sin(pi beta)/pi * integral_0^infinity t^-beta/(x+t) dt`.
Setting `t = exp(s)` and applying trapezoid quadrature generates positive
residues and positive shifts independent of any mock eigenvalues. Mock grids
use spacing `0.25`, with truncated ranges `[-24,24]` (193 terms) and
`[-112,112]` (897 terms). Scalar checks sample 257 points in `[0.25,2]` for
`beta = 0.25, 0.5, 0.75`, requiring refinement improvement and relative error
below `5e-12` on this sampled grid. This is not a certified uniform or minimax
error bound and is not a recommended production coefficient set.

The final heatbath is explicitly composed as `C r_(1-p/2)(C)`; action is
`r_p(C)` for `p = 0.5`, a zero-shift direct inverse for `p = 1`, and
`C^-1 r_(p-1)(C)` for `p = 1.5`. Intermediate inverse-fraction coefficients
alone are not the heatbath power. These compositions avoid large cancelling
constants/residues from converting positive or greater-than-one powers into
a single partial-fraction sum.

Dense partial-fraction matrices are compared against scalar spectral
evaluation and exact powers. Shifted-solve residuals are recorded separately
from scalar approximation, matrix-power, covariance, full heatbath/action,
fixed-pseudofermion action against the exact kernel, and generated-action
errors. Six matrices (the five previous ratios and a
rotated matrix at interval endpoints) give eighteen power cases. Constant and
signed-residue handling, coarse truncation, unsupported requests, and an
out-of-interval spectrum are controls. Identity factors retain nonzero
Gaussian action, now agreeing within the approximation budget rather than
exact roundoff. Declared budgets are mock-only, not RHMC acceptance targets.

Build `make mdwfDenseHpdRationalApplicationMockTest -j24` and run
`./mdwfDenseHpdRationalApplicationMockTest` from the build `testing/` directory.
Rebuild/run `mdwfDenseHpdDeterminantRatioMockTest` as the exact-only regression.
The user-supplied run on `n2dgx01` passes the included exact gate with unchanged
diagnostics and the rational gate with six matrices, eighteen power cases,
and five rejected inputs. The subsequent user-supplied standalone exact-only
`mdwfDenseHpdDeterminantRatioMockTest` run on `n2dgx01` also passes with
identical coverage and all diagnostics unchanged from the recorded exact
gate. No separate build log was supplied for either follow-up run.

Recorded rational diagnostics:

- Minimum coarse scalar relative error: `9.7915471952703115e-06`.
- Maximum refined scalar relative error: `8.8029583622528662e-13`.
- Maximum dense shifted-solve residual: `9.9301366129890925e-16`.
- Maximum heatbath matrix relative error: `7.3114626450804549e-13`.
- Maximum action-kernel matrix relative error: `1.6881839809412886e-15`.
- Maximum full heatbath/action consistency relative error: `1.4499310473410385e-12`.
- Maximum fixed-phi action relative error: `2.4424906541753444e-15`.
- Maximum generated-action relative error: `1.7202905766566801e-12`.

These validate the controlled quadrature and direct dense applications.
Physical mass/PV mapping, production coefficient sets, odd-flavor phase
treatment, coupled-CG/multishift accuracy, derivatives, force signs, and
device/MPI ownership remain unvalidated by this gate.

## 1. What the current operator actually exposes

`MDWFLinearOperator::apply()` calls `MDWFOperatorWorkspace::applyClover()`.
The current operator is the sum of a slice-wise Wilson/clover application and
the independent fifth-direction stencil:

```text
(M_scaffold psi)_s = W_clover(kernel_mass, c_sw) psi_s
                  + diagonal * psi_s
                  + forward(s)  * P_minus psi_(s+1)
                  + backward(s) * P_plus  psi_(s-1).
```

The wraparound hops use `forward_boundary` and `backward_boundary`, rather
than the respective interior hop coefficients. `Ls` remains one physical
fifth dimension; it is never a list of independent right-hand sides.

| Existing input | Meaning established by the implementation | Mapping still required |
| --- | --- | --- |
| `mass`, `setMass()` | Passed to the 4D Wilson/clover kernel | Wilson normalization and relation to domain-wall height `M5`; not automatically physical quark mass `mf` |
| `diagonal` | Additional gauge-independent diagonal term | Relation to the chosen 5D operator normalization |
| `forward_hop`, `backward_hop` | Gauge-independent projected interior hops | Signs, scale, and fifth-coordinate orientation |
| `forward_boundary`, `backward_boundary` | Gauge-independent wall-to-wall projected hops | Formula relating both coefficients to `mf`, including signs |
| `csw` | Passed through the Wilson/clover kernel | Kernel normalization and identical use in physical/PV operators |
| `Ls` | Compile-time physical fifth extent | Approved production extent; existing tests mostly use `Ls = 8` |
| Spacetime boundary conditions | No named MDWF boundary-phase input | Periodic/antiperiodic phases and their forward/adjoint handling |

Do not implement a Hasenbusch mass chain by repeatedly calling `setMass()`
unless a separate derivation establishes that this is the intended mass
parameter. Changing domain-wall quark mass normally changes the wall hops;
changing kernel mass instead changes the bulk operator and its regulator.

### General Mobius boundary

The reference 5D formulation uses `D_plus(s) = 1 + b5(s) D_W` and
`D_minus(s) = c5(s) D_W - 1`, with projected `D_minus` off-diagonal blocks
and boundary blocks multiplied by the quark mass. Thus general nonzero `c5`
does not fit the current gauge-independent fifth stencil by merely choosing
its five scalar coefficients. This observation does not invalidate tests of
the current scaffold; it limits what those tests establish.

The project guardrail keeps the existing fifth-direction helper gauge
independent. Do not silently change it. Before adding a general Mobius path,
review an explicit wrapper/composition and its adjoint, kernel normalization,
projector placement, determinant factors, and gauge derivative. Any
transformation used to obtain a different representation must account for
its determinant, including whether it cancels in the physical/PV ratio.
The current force derivation must not be reused unchanged if new
kernel-dependent fifth-neighbor terms are introduced.

Reference for the operator blocks and the `mf = 1` regulator convention:
[Brower, Neff, Orginos, sections 2.1-2.2](https://arxiv.org/html/1206.5214v2).
The audit/mapping requirements above are conclusions from the local code.

## 2. Physical and Pauli-Villars operator identity

Use `M(mf; theta)` to denote the future explicitly defined physical operator.
Its common identity `theta` contains kernel height/sign/normalization,
`b5(s)`, `c5(s)`, `Ls`, `c_sw`, gauge links, projector orientation, spacetime
boundary conditions, and any approved representation/preconditioning.

The physical and PV operators must share this identity, with only the
approved domain-wall mass changed. In the reference boundary-mass convention
the PV endpoint is `mf = 1`; this is not a prescription to pass `1` to the
existing Wilson-kernel `mass` argument. A different convention requires an
explicit mapping and cancellation proof, not a freely selected regulator.

Define the unpreconditioned normal operator:

```text
N(m) = M(m; theta)^dagger M(m; theta)
R_sector = [det N(m_physical) / det N(m_PV)]^p
p = Nf / 2.
```

This describes the magnitude of the corresponding `Nf`-flavor determinant.
For odd-flavor use, positivity or treatment of the determinant sign/phase
must be established; taking a normal-operator square root does not prove it.
A degenerate two-flavor block has `p = 1`; a one-flavor block has `p = 1/2`.
The actual flavor content is not selected by this note.

PV is gauge dependent through the common Wilson/clover kernel. Its gauge
derivative cannot be omitted just because the wall mass is gauge independent.
If even/odd preconditioning is introduced later, derive any local-block
determinant and force terms, especially for clover; a Schur-complement solve
alone is not the entire determinant unless the omitted factors cancel.

## 3. Multiple Hasenbusch factors

For an approved increasing mass ladder ending at the PV regulator,

```text
m0 = m_physical < m1 < ... < mk = m_PV
R_sector = product(i = 0,...,k-1)
           [det N(mi) / det N(mi+1)]^p.
```

This is an identity of determinants. Adjacent endpoints cancel and the PV
denominator occurs exactly once. A one-factor ladder is the unsplit ratio;
do not append another PV determinant to an already PV-terminated ladder.
Different flavor sectors may have different ladders and exponents.

Proposed factor metadata, not a new C++ interface in this patch:

```text
sector_id, factor_id
common_operator_identity
numerator_boundary_mass, denominator_boundary_mass
determinant_exponent
positive_operator_representation
heatbath_rational, action_rational, force_rational
spectral_interval_and_error_for_each_role
solve_limits_and_tolerances_for_each_role
integration_group_label                 // scheduling deferred
```

A future validator must check finite masses/exponents, positive exponent,
shared operator identity, ordered production ladders, exact endpoint
connectivity, physical start, PV end, and unique factor labels. Equal-mass
factors are useful explicit test cases but should not silently enter a
production ladder. The mass list, number of factors, or a HISQ coefficient
label must not implicitly choose the number of flavors.

## 4. A determinant ratio is not automatically a CG operator

`N(a)` and `N(b)` generally do not commute. The quotient `N(a) N(b)^(-1)`
need not be Hermitian in the coupled 5D inner product. Determinant telescoping
does not imply matrix telescoping, and does not authorize CG or fractional
powers of this naive quotient. Diagonal mocks alone cannot expose this error.

One mathematical HPD example, for nonsingular `M(a)` and `M(b)`, is:

```text
Q_ab = M(a) M(b)^(-1)
C_ab = Q_ab^dagger Q_ab
     = M(b)^(-dagger) N(a) M(b)^(-1)
det C_ab = det N(a) / det N(b).
```

This defines an example representation, not the selected implementation.
Its inverse solves, adjoint order, nested stopping tolerances, and gauge
chain rule require their own review. In particular, approximate inner solves
must not silently make the outer CG matvec variable or non-Hermitian.
Other determinant-equivalent formulations may be preferable; each must prove
HPD and account for transformations before using the rational scaffold.
Standard multi-mass MDWF boundary masses are not automatically additive
shifts of one normal operator; the current shifted-CG interface is not a
solver for every mass in a Hasenbusch ladder.

## 5. Heatbath, action, and force roles

Once a factor has an approved HPD representative `C` with weight `det(C)^p`,
the exact Gaussian construction is:

```text
xi = independent complex Gaussian noise with density exp(-xi^dagger xi)
phi = C^(p/2) xi
S_factor = phi^dagger C^(-p) phi.
```

These equations fix the exponent orientation. A heatbath rational
approximates `x^(p/2)` and an action rational approximates `x^(-p)` over the
factor's measured spectral interval. For exact functions at the generating
gauge field, `S_factor(phi) = ||xi||^2`; practical approximations must bound
and test the corresponding discrepancy. Each factor gets independently
generated noise, kept fixed during gauge differentiation and a trajectory.

The force rational approximates the action kernel for molecular dynamics;
it is not obtained by blindly substituting the scalar derivative
`-p x^(-p-1)` into the existing force workspace. For

```text
r_MD(C) = c0 I + sum_j alpha_j (C + sigma_j I)^(-1)
chi_j = (C + sigma_j I)^(-1) phi
delta S_MD = -sum_j alpha_j chi_j^dagger (delta C) chi_j.
```

The derivative of composite `C_ab` includes both numerator and denominator
operators and their inverse/adjoint placements. The existing
`eta = M chi` single-normal-operator force path is not automatically that
composite derivative. A distinct MD approximation may be lower accuracy than
the acceptance action, but the difference, rational errors, solve residuals,
and reversibility must be controlled explicitly. This note chooses neither
the momentum-update sign nor an additional factor of two/link multiplication.

Record role, factor identity, exponent, interval, error, shifts, residues,
degree, and solver tolerance with every coefficient set. Check positivity
and heatbath/action consistency; do not reuse HISQ determinant powers by
label. No coefficient generation or production noise generator is added here.

For equal masses the exact representative is `C = I`: the determinant ratio
is one, the heatbath is `phi = xi`, the action is the gauge-independent
constant `||phi||^2`, and the gauge derivative is zero. The action itself is
not zero unless this constant is explicitly subtracted by convention.

## 6. DSDR remains a separate 4D factor

The standard DSDR weight is

```text
A = D_W(-M5)^dagger D_W(-M5)
W_DSDR = det(A + epsilon_f^2 I) / det(A + epsilon_b^2 I).
```

This is not a fifth-dimensional PV operator or another MDWF boundary mass.
The standard construction uses a Wilson kernel; a clover-modified variant,
its normalization, and its physics must not be inferred from the MDWF
kernel. Do not choose numerical `M5`, `epsilon_f`, `epsilon_b`, gauge-action
parameters, or an integrator timescale in this patch.

Because both factors are functions of the same `A`, their shifts commute.
This does not make distinct MDWF mass normals commute. DSDR will need
separate 4D action/heatbath/force, spectral, MPI, and disabled-path tests.

Reference for the DSDR determinant orientation:
[RBC/UKQCD, section II.B, equation (3)](https://link.aps.org/accepted/10.1103/PhysRevD.86.094503).

## 7. Staged implementation and acceptance gates

1. Review the physical operator mapping: distinguish kernel height from wall
   mass; define `b5/c5`, projector/wall signs, normalization, spacetime phases,
   and same-kernel PV. Compare an independent small-volume/reference operator
   and adjoint, first at `c_sw = 0`, then with clover. Preserve all existing
   scaffold regressions rather than reinterpreting their parameters.
2. Add a test-only determinant-factor descriptor and mass-ladder validator.
   Test one ratio, several intermediate masses, endpoints, exponent handling,
   identity factors, and invalid metadata on scalar/diagonal mocks. These
   tests do not claim a physical PV implementation.
3. Select an HPD ratio representation and test small dense *noncommuting*
   matrices: Hermiticity, positivity, determinant identity, adjoint order,
   and numerical solves. Matrix products need not telescope even when their
   determinants do.
4. Add factor-specific heatbath/action on fixed gauge fields, matching
   coefficient roles to the chosen representation and spectrum. Validate
   physical/PV cancellation against an independent determinant or effective
   4D reference and rerun the `c_sw = 0` baseline.
5. Derive numerator/PV and every intermediate factor's gauge derivative;
   validate fixed-pseudofermion selected-link and all-link finite differences
   before defining additive force storage or momentum-update conventions.
6. Implement device/MPI ownership and validate identical global fields,
   rank-boundary matrices, separate component contractions, and controlled
   residuals across decompositions. A halo refresh is not reverse summation.
7. Implement DSDR as its own reviewed 4D determinant factor and force; require
   exact disabled/equal-shift behavior and finite-difference/MPI tests.
8. Only then connect an MDWF-specific action set to a trajectory driver with
   reviewed force signs and factor timescales. Require reversibility, step-size
   scaling of delta-H, acceptance diagnostics, and restart/RNG coverage.

The original documentation patch introduced no build target or executable;
the isolated metadata follow-up adds `mdwfDeterminantFactorizationMockTest`.
For subsequent code patches request explicit cluster build/run logs for each
focused target; local inspection is not compile, runtime, or physics evidence.

## 8. Decisions still requiring physics review

- Flavor content and separate sector exponents (for example, whether the
  intended theory is `2+1`; that example is not a chosen default).
- Exact 5D operator convention and independently checked mapping of existing
  kernel/fifth coefficients to `M5`, `mf`, and `b5/c5`.
- Production `Ls`, `c_sw`, spacetime boundary conditions, physical masses,
  PV endpoint mapping, and per-sector Hasenbusch ladders.
- Odd-flavor positivity/sign treatment and the HPD ratio representation.
- Spectral intervals/error budgets and heatbath/action/MD coefficient sets.
- Standard Wilson DSDR identity, regulator parameters, and gauge action.
- Additive force contract, momentum sign/normalization, output halos, MPI
  ownership, and integrator scheduling (separate later review gates).

Do not mark these decisions or PV/Hasenbusch/DSDR/RHMC as implemented or
validated when only the metadata or mock tests have passed. This review is
the dependency contract for those future patches, not authorization to wire
the current test-only matrices into production HMC/RHMC.
