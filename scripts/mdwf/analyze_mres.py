#!/usr/bin/env python3
"""
Residual-mass analysis of mdwfResidualMass output (src/testing/main_mdwfResidualMass.cpp).

Input: the per-solve component files <configuration>.dat.c00 ... c11 (format v2) of one variant directory.
Each holds, for one source spin-colour component a, the raw slice sums along every direction mu = x, y, z, t,

    C_PP,a(mu, n)  = sum_{sites with x_mu = n} |q_a(x)|^2        (q = exported 4D solution)
    C_J5q,a(mu, n) = sum_{sites with x_mu = n} |p_a(x)|^2        (p = midpoint field, Grid ContractJ5q)

plus the CG iterations, residue and time of that solve; the first line is the parameter key, which includes
the source position.

Analysis, per configuration with all 12 components:
  1. C(mu, n) = sum_a C_a(mu, n).
  2. Fold about the source along the chosen direction(s): C(d) = [C(s + d) + C(s - d)] / 2, d = 0 ... L/2;
     direction 'spatial' sums the folded x, y and z correlators (equal extents), 'x' | 'y' | 'z' | 't' one.
  3. m_res(config) = mean over d in [plateau_min, plateau_max] of C_J5q(d) / C_PP(d).
Per beta (read from 'b<4 digits>' in the configuration name):
  - HotQCD-style estimate: mean over the plateau of <C_J5q(d)> / <C_PP(d)>, correlators averaged over the
    configurations first, with a delete-one jackknife error over configurations;
  - the same ratio R(d) = <C_J5q(d)> / <C_PP(d)> fitted to a constant over the plateau with uncorrelated
    weights 1/sigma(d)^2 (sigma(d): jackknife error of R(d)), the fit repeated on every jackknife sample with
    the same weights for its error;
  - mean of the per-configuration values, with its jackknife error.
With a second variant: per-configuration ratios m_res(B) / m_res(A) over the common configurations, their
jackknife mean and error, and the ratio of total solve times.

Usage:
  analyze_mres.py VARIANT [VARIANT2] [--runs-dir DIR] [--plateau 4 14] [--direction spatial|x|y|z|t]
                  [--profile]
  analyze_mres.py --selftest

VARIANT is a directory, or a tag under --runs-dir (default: $MDWF_RUNS_DIR, else ./runs/prod).
--profile prints m_res(d) of every configuration (every second d). Default plateau d = 4..14 (HotQCD).
"""
import argparse
import math
import os
import re
import sys
import tempfile
from collections import defaultdict

NAME = re.compile(r'^(.+)\.dat\.c(\d\d)$')
BETA = re.compile(r'b(\d{4})')
SOURCE = re.compile(r' source (-?\d+) (-?\d+) (-?\d+) (-?\d+) ')
DIRECTIONS = {'spatial': [0, 1, 2], 'x': [0], 'y': [1], 'z': [2], 't': [3]}


def read_component(path):
    """(iterations, seconds, source, {mu: (pp list, j5q list)}) of one component file."""
    lines = open(path).read().split('\n')
    m = SOURCE.search(lines[0] + ' ')
    if not m:
        raise ValueError(f'{path}: no source position in the key line')
    source = tuple(int(v) for v in m.groups())
    _, iterations, _, seconds = lines[1].split()
    i = 2
    corr = {}
    for _ in range(4):
        mu, n = map(int, lines[i].split())
        i += 1
        pp, j5 = [], []
        for _ in range(n):
            _, p, j = lines[i].split()
            i += 1
            pp.append(float(p))
            j5.append(float(j))
        corr[mu] = (pp, j5)
    return int(iterations), float(seconds), source, corr


def load(directory):
    """{configuration: (beta, source, {mu: (pp, j5q)}, iterations, seconds)} for complete ones; count incomplete."""
    parts = defaultdict(dict)
    for f in os.listdir(directory):
        m = NAME.match(f)
        if m:
            parts[m.group(1)][int(m.group(2))] = os.path.join(directory, f)
    configs = {}
    for name, files in sorted(parts.items()):
        if sorted(files) != list(range(12)):
            continue
        total, iterations, seconds, source = {}, 0, 0.0, None
        for a in range(12):
            it, sec, src, corr = read_component(files[a])
            if source is not None and src != source:
                raise ValueError(f'{name}: components with different sources')
            source = src
            iterations += it
            seconds += sec
            for mu, (pp, j5) in corr.items():
                tp, tj = total.setdefault(mu, ([0.0] * len(pp), [0.0] * len(pp)))
                for n in range(len(pp)):
                    tp[n] += pp[n]
                    tj[n] += j5[n]
        b = BETA.search(name)
        configs[name] = (int(b.group(1)) / 1000.0 if b else 0.0, source, total, iterations, seconds)
    return configs, len(parts) - len(configs)


def folded(total, source, direction):
    """Folded (pp, j5q) for d = 0 ... L/2 along the direction(s), about the source position."""
    mus = DIRECTIONS[direction]
    length = len(total[mus[0]][0])
    if any(len(total[mu][0]) != length for mu in mus):
        raise ValueError(f'direction {direction}: unequal extents')
    pp = [0.0] * (length // 2 + 1)
    j5 = [0.0] * (length // 2 + 1)
    for mu in mus:
        tp, tj = total[mu]
        s = source[mu]
        for d in range(length // 2 + 1):
            pp[d] += 0.5 * (tp[(s + d) % length] + tp[(s - d) % length])
            j5[d] += 0.5 * (tj[(s + d) % length] + tj[(s - d) % length])
    return pp, j5


def plateau_mean(pp, j5, lo, hi):
    if hi >= len(pp):
        raise ValueError(f'plateau end {hi} beyond L/2 = {len(pp) - 1}')
    return sum(j5[d] / pp[d] for d in range(lo, hi + 1)) / (hi - lo + 1)


def jackknife(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float('nan')
    jk = [(n * mean - v) / (n - 1) for v in values]
    return mean, math.sqrt((n - 1) / n * sum((j - mean) ** 2 for j in jk))


def averaged_estimate(corrs, lo, hi):
    """Jackknife of the plateau mean of <C_J5q(d)> / <C_PP(d)> (correlators averaged over configurations)."""
    def est(sel):
        pp = [sum(corrs[i][0][d] for i in sel) for d in range(len(corrs[0][0]))]
        j5 = [sum(corrs[i][1][d] for i in sel) for d in range(len(corrs[0][0]))]
        return plateau_mean(pp, j5, lo, hi)
    n = len(corrs)
    full = est(range(n))
    if n < 2:
        return full, float('nan')
    jk = [est([i for i in range(n) if i != k]) for k in range(n)]
    mean = sum(jk) / n
    return full, math.sqrt((n - 1) / n * sum((j - mean) ** 2 for j in jk))


def fitted_estimate(corrs, lo, hi):
    """Uncorrelated constant fit of R(d) = <C_J5q(d)>/<C_PP(d)> over [lo, hi]; (fit, jackknife error, chi2/dof)."""
    n = len(corrs)
    nd = len(corrs[0][0])

    def ratios(sel):
        pp = [sum(corrs[i][0][d] for i in sel) for d in range(nd)]
        j5 = [sum(corrs[i][1][d] for i in sel) for d in range(nd)]
        return [j5[d] / pp[d] for d in range(lo, hi + 1)]

    full = ratios(range(n))
    if n < 2:
        return sum(full) / len(full), float('nan'), float('nan')
    samples = [ratios([i for i in range(n) if i != k]) for k in range(n)]
    sigma = []
    for j in range(len(full)):
        m = sum(sm[j] for sm in samples) / n
        sigma.append(math.sqrt((n - 1) / n * sum((sm[j] - m) ** 2 for sm in samples)))
    if min(sigma) <= 0.0:   # noiseless data (e.g. the selftest): plain mean
        sigma = [1.0] * len(full)
    w = [1.0 / x ** 2 for x in sigma]

    def fit(r):
        return sum(wi * ri for wi, ri in zip(w, r)) / sum(w)

    c = fit(full)
    jk = [fit(sm) for sm in samples]
    mean = sum(jk) / n
    err = math.sqrt((n - 1) / n * sum((j - mean) ** 2 for j in jk))
    dof = len(full) - 1
    chi2 = sum(((ri - c) / si) ** 2 for ri, si in zip(full, sigma)) / dof if dof > 0 else float('nan')
    return c, err, chi2


def analyze(directory, label, direction, lo, hi, profile, out=sys.stdout):
    configs, incomplete = load(directory)
    print(f'== {label} ({directory}): {len(configs)} complete configurations, {incomplete} incomplete; '
          f'direction {direction}, plateau d = {lo}..{hi}', file=out)
    per, corrs, by_beta = {}, {}, defaultdict(list)
    for name, (beta, source, total, iterations, seconds) in configs.items():
        pp, j5 = folded(total, source, direction)
        corrs[name] = (pp, j5)
        per[name] = (beta, plateau_mean(pp, j5, lo, hi), iterations, seconds)
        by_beta[beta].append(name)
        print(f'  {name}  m_res {per[name][1]:.4e}  {iterations:6d} CG its  {seconds / 3600:.2f} h', file=out)
        if profile:
            print('      m_res(d), d = 0, 2, ...: '
                  + ' '.join(f'{j5[d] / pp[d]:.1e}' for d in range(0, len(pp), 2)), file=out)
    for beta, names in sorted(by_beta.items()):
        avg, avg_err = averaged_estimate([corrs[n] for n in names], lo, hi)
        fit, fit_err, chi2 = fitted_estimate([corrs[n] for n in names], lo, hi)
        mean, err = jackknife([per[n][1] for n in names])
        hours = sum(per[n][3] for n in names) / len(names) / 3600
        print(f'  beta {beta:.3f}: m_res = {avg:.4e} +- {avg_err:.1e} (configuration-averaged correlators, '
              f'jackknife), {mean:.4e} +- {err:.1e} (mean of per-configuration values); {len(names)} '
              f'configurations, {hours:.2f} h of solves per configuration', file=out)
        print(f'  beta {beta:.3f}: m_res = {fit:.4e} +- {fit_err:.1e} (constant fit of the averaged ratio over '
              f'd = {lo}..{hi}, weights 1/sigma^2, chi2/dof (uncorrelated) {chi2:.2f})', file=out)
    return per


def paired(label_a, per_a, label_b, per_b, out=sys.stdout):
    common = sorted(set(per_a) & set(per_b))
    print(f'== paired {label_b} / {label_a}: {len(common)} common configurations', file=out)
    by_beta = defaultdict(list)
    for n in common:
        by_beta[per_a[n][0]].append(n)
    for beta, names in sorted(by_beta.items()):
        ratio, err = jackknife([per_b[n][1] / per_a[n][1] for n in names])
        cost = sum(per_b[n][3] for n in names) / sum(per_a[n][3] for n in names)
        print(f'  beta {beta:.3f}: m_res ratio = {ratio:.3f} +- {err:.3f} ({len(names)} configurations), '
              f'solve-time ratio {cost:.2f}', file=out)


def write_component(path, source, a, pp_dirs, j5_dirs, iterations=100, seconds=1.0):
    with open(path, 'w') as f:
        f.write(f'# mdwfResidualMass component v2: lattice test source {" ".join(map(str, source))} gauge test\n')
        f.write(f'{a} {iterations} 1e-10 {seconds}\n')
        for mu in range(4):
            f.write(f'{mu} {len(pp_dirs[mu])}\n')
            for n in range(len(pp_dirs[mu])):
                f.write(f'{n} {pp_dirs[mu][n]!r} {j5_dirs[mu][n]!r}\n')


def selftest():
    """Synthetic data with known answers: folding about an off-origin source, both estimators, jackknife."""
    extents = [8, 8, 8, 4]
    ok = True

    def check(what, got, expected, tol=1e-12):
        nonlocal ok
        good = abs(got - expected) <= tol * max(1.0, abs(expected))
        ok = ok and good
        print(f'  {"PASS" if good else "FAIL"}  {what}: {got!r} (expected {expected!r})')

    with tempfile.TemporaryDirectory() as tmp:
        # Two configurations, source at (3, 1, 0, 2); C_PP(n) = exp(-|dist|), C_J5q = r_c * C_PP with r_c = 1e-4, 3e-4,
        # split unevenly over the 12 components (the sum is what counts).
        source = (3, 1, 0, 2)
        ratios = {'b1801_cfgA': 1e-4, 'b1801_cfgB': 3e-4}
        for name, r in ratios.items():
            for a in range(12):
                w = (a + 1) / 78.0
                pp, j5 = [], []
                for mu in range(4):
                    L = extents[mu]
                    dist = [min((n - source[mu]) % L, (source[mu] - n) % L) for n in range(L)]
                    pp.append([w * math.exp(-x) * (1.0 + 0.1 * mu) for x in dist])
                    j5.append([r * v for v in pp[-1]])
                write_component(os.path.join(tmp, f'{name}.dat.c{a:02d}'), source, a, pp, j5)
        # An incomplete one must be ignored.
        write_component(os.path.join(tmp, 'b1801_cfgC.dat.c00'), source, 0, [[1.0] * L for L in extents],
                        [[1.0] * L for L in extents])
        configs, incomplete = load(tmp)
        check('complete configurations', len(configs), 2)
        check('incomplete configurations', incomplete, 1)
        for name, r in ratios.items():
            beta, src, total, _, _ = configs[name]
            pp, j5 = folded(total, src, 'spatial')
            check(f'{name} folded C_PP(0) (sum over x, y, z at the source)', pp[0], 1.0 + 1.1 + 1.2)
            check(f'{name} m_res (exact ratio)', plateau_mean(pp, j5, 1, 3), r)
        corrs = [folded(configs[n][2], configs[n][1], 'spatial') for n in sorted(configs)]
        avg, avg_err = averaged_estimate(corrs, 1, 3)
        check('averaged-correlator estimate (equal C_PP, so the mean ratio)', avg, 2e-4)
        check('its jackknife error (two configurations: |r_A - r_B| / 2)', avg_err, 1e-4)
        fit, fit_err, _ = fitted_estimate(corrs, 1, 3)
        check('constant fit of the averaged ratio (d-independent ratio: the same value)', fit, 2e-4)
        mean, err = jackknife([1.0, 2.0, 3.0, 4.0])
        check('jackknife mean of 1..4', mean, 2.5)
        check('jackknife error of 1..4 (= standard error)', err, math.sqrt(5.0 / 12.0))
    print('selftest', 'passed' if ok else 'FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('variant', nargs='?')
    ap.add_argument('variant2', nargs='?')
    ap.add_argument('--runs-dir', default=os.environ.get('MDWF_RUNS_DIR', os.path.join('runs', 'prod')))
    ap.add_argument('--plateau', nargs=2, type=int, default=[4, 14])
    ap.add_argument('--direction', default='spatial', choices=sorted(DIRECTIONS))
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.variant:
        ap.error('VARIANT is required (or --selftest)')
    lo, hi = args.plateau

    def resolve(v):
        d = v if os.path.isdir(v) else os.path.join(args.runs_dir, v)
        if not os.path.isdir(d):
            sys.exit(f'no directory {v} (nor {d})')
        return d

    per_a = analyze(resolve(args.variant), args.variant, args.direction, lo, hi, args.profile)
    if args.variant2:
        per_b = analyze(resolve(args.variant2), args.variant2, args.direction, lo, hi, args.profile)
        paired(args.variant, per_a, args.variant2, per_b)


if __name__ == '__main__':
    main()
