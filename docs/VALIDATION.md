# Validation record

Date: 2026-09-24. Version: 0.1.0 (unreleased).

## Numerical accuracy

The suite checks single-SNP, rank-one and equal-weight probabilities against
analytic distributions and 60-digit mpmath references. Independent angular and
beta-direction integrals check unequal spectra, including probabilities close
to one, scale invariance, deep tails, monotonicity and method transitions.

The near-one tests exposed and now prevent cancellation in the old survival
integral: for eigenvalues `[1, 0.4, 0.1]` and `q=1e-6`, the correct log survival
is approximately `-1.3298058e-9`. The small-statistic series evaluates the lower
tail directly before taking `log1p(-CDF)`. Gamma survival uses the same stable
complement principle. This matters for negative Z even when decimal P rounds
to one.

`benchmarks/numerical_accuracy.py` independently checked 35 two-weight cases,
with ratios `{0.001, 0.01, 0.1, 0.5, 0.9}` and statistics
`{0.1, 1, 2, 5, 10, 30, 100}`. Maximum relative error was 1.3592e-12.
Additional beta-direction tests cover multiplicities `(1,1)`, `(2,3)`, `(1,9)`,
`(10,20)`. These are measured checks, not universal error bounds; numerical
fallbacks remain explicitly labeled in `PMETHOD`.

### Independent resolution of small-gene discrepancies

Slurm job 2680538 reran three real AD genes with official MAGMA v1.10 and
fastmagma. An independent oracle integrates radial chi-square survival over
sphere directions, without using either program's characteristic-function
solver. Gauss-Legendre orders 64, 128, 256 and 512 were checked against adaptive
angular integration. All reference calls in these genes are complete.

| Gene | SNPs | Independent P | fastmagma P | Official MAGMA P |
|---|---:|---:|---:|---:|
| 56632 | 3 | 0.4416564273703677 | 0.4416564273703657 | 0.44157 |
| 56733 | 3 | 0.0003204600674659268 | 0.0003204600674659 | 0.00031782 |
| 56734 | 2 | 0.0010524892233385215 | 0.0010524892233385 | 0.0010445 |

The 512-point and adaptive answers agree within 4.5e-16 absolute. These cases
resolve the largest remaining AD discrepancies: fastmagma agrees with the
independent probabilities; the official numerical integration is less accurate.
Changing fastmagma to reproduce these official values would introduce error.

## Official executable boundary checks

`benchmarks/validate_boundaries.py` constructs PLINK fixtures and runs both CLIs.
Job 2680538 covered 28 cases: both programs accepted the same 24 valid cases and
rejected the same four invalid cases (negative/NaN P, NaN/infinite N). Exit-code
numbers differ; rejection behavior agrees.

Cases include complete calls, overlapping/disjoint missingness at 5% and 25%,
6% and 26% missing calls, all-missing/monomorphic SNPs, P=0, P=1, tiny P,
duplicate rows with valid/missing/invalid values in different orders, and sample
sizes around the rounded-N > 50 cutoff. All accepted cases have identical gene
sets, NSNPS, NPARAM and N. Maximum absolute P difference is 4.5762e-6.

The 100-sample, 60-SNP correlated-missingness case exercises two statistical
blocks. Its untruncated corrected LD includes 576 entries above one, reaching
1.26, so it detects accidental clipping of Brown covariance estimates. Only
the within-block eigenvalue matrices are clipped, as in the reference model.
The summary-statistic missing-call cutoff is 25%; MAGMA's generic 5% QC default
does not apply to its reduced-QC `--pval` path.

Extreme-P agreement must not be interpreted as relative numerical equivalence:
the official integral can floor probabilities near 3.9e-16 while fastmagma
preserves much smaller tails. See [algorithm scope](ALGORITHM.md).

## Engineering and package checks

The current suite has 234 tests. Coverage includes bounded genotype caching,
missing-call LD and non-positive-semidefinite spectra, block boundaries and
Brown limits, chunk-independent input ordering, trait-specific SNP masks,
reuse of prepared spectra, input validation, resource guards, atomic output
publication, concurrent chromosome locks, checksums and model-revision-safe
merges. The whole-gene mode retains its separate historical input policy.

All 234 tests passed locally on Python 3.12 and against installed wheels on
Python 3.10 (minimum dependencies), 3.11, 3.13 and 3.14; the installed source
distribution also passed all 234 tests on Python 3.12. Installed-package tests
run outside the source checkout. Ruff and strict Twine checks pass. The minimum dependency
set is NumPy 1.24.0, pandas 2.0.0, SciPy 1.10.0, bed-reader 1.0.0 and
threadpoolctl 3.1.0. Windows checks are configured in CI; no hosted CI success
or Windows execution is claimed here.

Wheel and source archives exclude reference data, private results, logs and
local configuration. No TestPyPI/PyPI upload or namespace ownership has been
verified.

## Complete chromosome-22 comparison

The reference is the existing 1000G EUR PLINK panel (489 samples), with 10 kb
gene annotations and supplied official MAGMA v1.10 CAD/AD outputs. All matched
official P values exceed 1e-8. These runs validate this panel and chromosome;
they do not establish calibration under reference mismatch or new ancestries.

The earlier whole-gene implementation had maximum absolute Z differences of
0.547092 (CAD) and 0.280224 (AD). Adding automatic blocks removed those large
differences and all four CAD nominal P=0.05 disagreements. The final wheel run
(job 2680539), including input and missing-call compatibility, gave:

| Metric | CAD | AD |
|---|---:|---:|
| Matched genes | 1,229 | 1,100 |
| Maximum absolute Z difference | 0.0000568631 | 0.0022933280 |
| Median absolute Z difference | 0.0000043243 | 0.0000038172 |
| Exact NPARAM / NSNPS / N agreement | 100% | 100% |
| Top-20 overlap | 20/20 | 20/20 |
| P=0.05 threshold crossings | 0 | 0 |

The three small-gene oracles above explain the leading AD residuals, including
the maximum at gene 56734. Decimal printing in official output also limits
exact comparisons. Historical investigation is retained in
[the algorithm audit](EQUIVALENCE.md).

## Performance measurements

The final run took 142.858 seconds including imports and peaked at
420,832 KiB RSS (410.97 MiB), using two allocated CPUs and a 3 GiB memory request.
It prepared 2,589 block spectra; Brown aggregation was used for 144 CAD and
five AD genes. One AD gene used the near-one `small_q_series` method. No block used a saddlepoint or LTZ fallback. These are individual
observations, not a controlled speedup comparison with MAGMA.

An earlier extracted-script run peaked at 2,473,888 KiB and took 114.340 s;
scientific changes between versions prevent interpreting this as a comparison
at identical behavior. The synthetic smaller-Gram benchmark checks a separate
150-by-1,500 complete matrix: three repetitions gave median eigensolver times
0.18793 s (SNP Gram) and 0.000952 s (sample Gram), with spectra agreeing within
1e-10. This kernel speed ratio is not an end-to-end speedup.

## Reproduction

```bash
python -m pip install -e '.[dev]'
python -m pytest
python benchmarks/numerical_accuracy.py --out results/numerical_accuracy.json
python benchmarks/benchmark.py --out results/benchmark.json
python benchmarks/validate_boundaries.py --magma /path/to/magma \
  --bfile /path/to/reference.22 --annot /path/to/genes.annot \
  --ad-pval /path/to/ad.pval --out-dir results/boundary_validation
python benchmarks/validate_block_compatibility.py \
  --fast-dir /path/to/fastmagma/output --official-dir /path/to/official/output \
  --out-dir results/comparison
```

Local evidence is under `results/boundary_validation/`, `results/final_validation/`
and `results/algorithm_diagnostic/`. Results and datasets are not distributed.
Private run manifests record input paths, sizes and mtimes; published output
files are checked by SHA-256. Obsolete diagnostic scripts are retained only
beside their historical evidence; active reusable validators live in `benchmarks/`.
