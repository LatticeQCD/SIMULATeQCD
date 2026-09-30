# Even/odd preconditioning of the Möbius clover MDWF operator

First performance item after the working MDWF RHMC (user decision: the solver
is the main cost). Correctness first: every stage is validated against the
unpreconditioned scaffold before it is used or benchmarked.

## 1. Block structure

`M = D_W Din + Shift` with `D_W = A - (1/2) Hop` (`MDWFMobiusMapping.h`):

- `A = mass + clover`, site-local: two Hermitian 6x6 chiral blocks per site,
  exactly the `fmunu_upper/lower` fields that `preCalcFmunu` computes
  (`mass = 4 - M5` after commit `f4acf89`).
- `Hop` (`DiracWilsonEvenOdd2`, templated on output/input layout) connects only
  sites of opposite parity.
- `Din = (b5, c5, c5, -mf c5, -mf c5)` and `Shift = (1, -1, -1, mf, mf)` act
  in `s` and on the chiral projectors `P_+` (upper 6 components, coupled to
  `s-1`) and `P_-` (lower, coupled to `s+1`).

Hence

```text
M_ee = A Din + Shift (even),  M_oo = A Din + Shift (odd),
M_eo = Hop_eo Din,            M_oe = Hop_oe Din,
M_xx^+ = Din^+ A + Shift^+,   M_eo^+ = Din^+ g5 Hop_oe g5,   M_oe^+ = Din^+ g5 Hop_eo g5.
```

## 2. Diagonal-block inverse

Per site and chirality, `Din = b5 + c5 K`, `Shift = 1 - K` with `K` the
one-directional `s` shift with boundary factor `-mf` (`K^Ls = -mf`), so

```text
M_xx = P + Q K,   P = b5 A + 1,   Q = c5 A - 1   (Hermitian, commuting, s-independent)
(P + Q K)^-1 = (1 + R K)^-1 P^-1,   R = P^-1 Q,
```

and `(1 + R K) x = z` is one O(Ls) sweep with `W = (1 + (-1)^Ls mf R^Ls)^-1`
at the boundary (derivation in `MDWFMobiusEvenOdd.h`). Storage: `P^-1`, `R`,
`W` per site and chirality (six packed Hermitian 6x6), recomputed with the
clover field in `refresh()`. `M_xx^+ = P + Q K^+` uses the same matrices with
the sweep directions exchanged. For `c_sw = 0` the matrices are multiples of
the identity (the scalar LDU of Grid's `CayleyFermion5D::MooeeInv`).

## 3. Schur complement and solves (stage E1)

`Mhat = M_ee - M_eo M_oo^-1 M_oe` on even sites, `det M = det M_oo det Mhat`,
and `M^+` has Schur complement `Mhat^+`. `MDWFMobiusEvenOddSolver`:

- `M x = b`: `bhat = b_e - M_eo M_oo^-1 b_o`, CG on `Mhat^+ Mhat`, back-substitute `x_o`.
- `M^+ y = b`: `bhat = b_e - M_oe^+ M_oo^-+ b_o`, CG on `Mhat^+ Mhat` for `w`, `y_e = Mhat w`.
- `M^+ M x = b`: the two solves in sequence.

The CG is the existing `MDWFCoupledCG` through `MDWFCoupledSolverAdapter`
(layout-agnostic). Even/odd split and merge use MDWF-local functors that keep
the stack index; `SpinorfieldAll`'s conversion (`returnSpinor`) reads its
source through a stackless `gSite` and must not be used for `Ls`-stack fields.

Validation: `mdwfMobiusEvenOddTest` (block reconstruction of `M` and `M^+`,
block inverse, Schur adjoint identity, solves against the unpreconditioned CG
with true residuals, iterations, and times).

Cost per CG iteration: one `Mhat` is two half-volume hops, one diagonal block,
one block inverse, i.e. about one unpreconditioned `M`, on half-size vectors.
The gain is the iteration ratio. For `M^+ M` (two solves) the gain is roughly
half of that, so the RHMC should move to even-site pseudofermions (E2).

## 4. Next stages

- **E2 — even/odd pseudofermion actions.** Pseudofermions on even sites with
  `Mhat^+ Mhat` for both the Pauli-Villars two-flavour and the one-flavour RHMC
  actions (the multishift acts on `Mhat^+ Mhat`). `det M_oo` depends on the
  gauge field through the clover term (for `c_sw = 0` it is constant and
  drops out): `log det M_oo = sum_{x odd, chi} [Ls log det P - log det W]`,
  whose force needs `G_chi(x) = sum_s [(P + Q K)^-1 Din]_ss` (a 6x6 per site)
  contracted with the clover derivative. The Pauli-Villars ratio does not
  cancel it (it depends on the mass). Validate with the existing ladder
  (heatbath, energy identity, cancellation, trajectory tests, `<exp(-dH)>`).
- **E3 — performance.** Cache the clover field per gauge configuration (the
  unpreconditioned scaffold recomputes `preCalcFmunu` on every application),
  fuse `Din`/hop/diagonal passes, mixed precision, and a GPU force; then the
  Grid comparison (`BENCHMARK_PROTOCOL.md`).
