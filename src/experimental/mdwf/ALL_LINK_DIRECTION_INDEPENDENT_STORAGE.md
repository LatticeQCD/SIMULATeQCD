# MDWF Single-Rank All-Link Direction-Independent Storage Review

This note defines the storage scaffold after the validated selected-link raw
Wilson/clover projection and additivity checks.  Its single-rank `c_sw = 0`
Wilson gate and the test-only nonzero-`c_sw` split Wilson/clover extension are
cluster-validated.  Neither gate names the result `ipdot`, chooses the HMC
sign, defines MPI ownership, updates momenta, or edits RHMC/HMC, gauge force,
HISQ force, `SiteComm`, `GIndexer`, or global memory layouts.

## Validated input

For one selected link in the left-variation convention,

```text
U(epsilon) = exp(epsilon H) U,
dS(H) = Re tr(H K),
```

the current tests establish at roundoff:

```text
TA(B_W + B_C)
  = TA(B_W) + TA(B_C)
  = K_W + K_C
  = K_total.
```

The earlier single-rank all-link random-direction tests establish correct
coverage of all `4 * volume` bulk links for one deterministic direction per
link.  Combining those results justifies a storage scaffold, but it does not
yet justify a production force field or a distributed ownership rule.

## Scope and representation

The first implementation remains single rank and test-only.  It must reject
the run unless

```text
local_volume == global_volume.
```

Use full `R18` matrices.  Raw contraction matrices are not unitary and must
not be stored through a compressed unitary representation.

The scaffold should keep separate host-side bulk buffers:

```text
raw_wilson[4 * local_volume]
raw_clover[4 * local_volume]
```

and a caller-supplied gauge-field-like `R18` destination for the projected
total.  Separate raw buffers keep the Wilson/clover split inspectable and
prevent projection from being applied before all contributions are present.

The linear bulk link index is exactly

```text
bulk_link = site.isite * 4 + mu,
0 <= site.isite < local_volume,
0 <= mu < 4.
```

No halo index, periodic image, traversal counter, or global coordinate hash
may be used as the destination index.

## Bulk-only coverage contract

Every storage and inspection loop must use:

```text
for site_index in [0, local_volume)
  for mu in [0, 4)
```

The implementation must verify:

- the buffer size is exactly `4 * local_volume`;
- exactly `4 * local_volume` bulk links are visited;
- every bulk link is finalized exactly once;
- no duplicate or missing bulk index exists;
- every stored matrix is finite;
- projected matrices are anti-Hermitian and traceless within tolerance.

The output contract says nothing about destination halos.  The overwrite
operation must not call `updateAll()`.  A later reader may explicitly refresh
halos, but that refresh is assignment into halo storage and is not force
accumulation.

Input gauge and spinor halos must already be valid before stencil evaluation.
That input requirement does not make output halos part of this scaffold.

## Explicit zero and overwrite semantics

The first public test-facing operation should be named to expose its behavior,
for example:

```text
overwriteMDWFAllLinkContractionMatrices(destination, inputs)
```

The exact C++ name may differ, but `overwrite` must remain explicit.

For each evaluation it must:

1. allocate or resize both raw bulk buffers;
2. set every raw Wilson and clover matrix to exact zero;
3. accumulate contributions only into those zeroed buffers;
4. form `raw_total = raw_wilson + raw_clover` per bulk link;
5. apply `SU3::TA()` exactly once to `raw_total`;
6. write every destination bulk link exactly once;
7. never read the previous destination value.

The destination must therefore be independent of its initial contents.
Calling the overwrite operation twice with identical inputs must reproduce
the same bulk matrices.

The operation must not expose an ambiguous method named only `accumulate` or
`computeForce`.  It returns test-local contraction representatives, not a
production force or `ipdot`.

## Internal addition semantics

Linearity requires internal addition before projection.  The implementation
must make the following logical operations visible:

```text
zeroRawBuffers()
addWilsonContribution(link, rational_term, fifth_slice, matrix)
addCloverContribution(link, rational_term, fifth_slice, path_occurrence,
                      matrix)
finalizeProjectedTotalOverwrite(destination)
```

They need not become public functions, but the phases must remain distinct and
reviewable.

Required rules:

- Wilson contributions add over all rational terms and all `Ls` slices.
- Clover contributions add over all rational terms, all `Ls` slices, all
  affected `(mu, nu)` blocks, all four clover leaves, and every occurrence of
  the owned physical link in a leaf.
- Wilson and clover raw matrices remain separate until finalization.
- Mass and fifth-direction terms add nothing because they are gauge
  independent.
- No per-slice, per-term, per-path, or per-component `TA()` is allowed.
- Finalization projects the raw total once per bulk link.

For the first single-rank implementation, correctness and explicit counting
take priority over GPU parallelization or abstraction.

## Nonzero-sentinel overwrite test

The overwrite contract must be tested with a deterministic, finite, nonzero
sentinel on every destination bulk link.  The sentinel should contain
Hermitian, trace, and anti-Hermitian components so an accidental add cannot
hide behind `TA()`.

Run the same evaluation in three states:

```text
A: destination starts at exact zero
B: destination starts at the nonzero per-link sentinel
C: destination contains the completed result from B
```

Then require:

```text
bulk(A) == bulk(B)
bulk(B) == bulk(C after a second overwrite)
```

within a roundoff-level matrix tolerance.  Also require that the sentinel and
the completed result differ on at least one bulk link, so the test cannot pass
with a no-op writer.

Inspect bulk links only.  Sentinel values in output halos are deliberately
ignored because halos are outside the overwrite contract.

A future API that adds MDWF matrices to an already populated caller field
must use a separately named operation such as

```text
addMDWFAllLinkContractionMatrices(destination, inputs)
```

and a different sentinel test requiring `result = sentinel + contribution`.
That additive caller-facing API is not part of the first scaffold.

## Validation ladder

Keep one conceptual change per patch:

1. `c_sw = 0` Wilson-only all-link raw storage and projected overwrite.
2. Validate all `4 * volume` bulk links, zero/sentinel/repeated-overwrite
   equivalence, and contraction with the existing deterministic direction
   field.
3. Rerun the selected-link raw Wilson projection and `c_sw = 0` all-link
   random-direction regressions.
4. Add nonzero-`c_sw` clover raw storage while preserving separate Wilson and
   clover diagnostics.
5. Validate projected additivity and the nonzero-clover all-link centered
   finite difference.
6. Only then review MPI owner-computes storage.

For each all-link test, report:

- local/global volume and inspected/finalized link counts;
- missing and duplicate link counts;
- invalid matrix count;
- maximum anti-Hermitian and trace violations;
- zero-versus-sentinel overwrite difference;
- first-versus-second overwrite difference;
- maximum stored Wilson, clover, raw-total, and projected-total norms;
- deterministic-direction Wilson, clover, total, and stored contractions;
- finite-difference absolute/relative differences and solver residuals.

## Deferred boundaries

This storage review does not decide:

- conversion of the contraction matrix to `ipdot`;
- the HMC force sign or momentum-update normalization;
- a caller-facing additive force API;
- output halo refresh requirements for later consumers;
- MPI ownership, reverse accumulation, or rank-boundary routing;
- RHMC/HMC integration or performance optimization.

The existing HISQ `TA(2 U F)` expression remains out of scope.  The MDWF
left-oriented raw matrices already contain their derived link placement and
must not acquire another link multiplication or factor of two during storage.
