# Statistical definition and numerical implementation

The default `--model magma` reconstructs MAGMA v1.10's summary-statistic
SNP-wise mean gene test, including input QC, missing-call LD adjustment,
balanced SNP blocks, and Brown aggregation. `--model whole` retains the
unpartitioned quadratic-form test. Numerical probability evaluation is
independent of MAGMA. See the [equivalence investigation](EQUIVALENCE.md) for
controlled comparisons and the [validation record](VALIDATION.md) for evidence.

## Input selection

Intersect annotation SNP IDs, PLINK reference IDs, and usable summary-statistic
IDs separately for each gene and trait. SNP order follows the reference panel.
The test uses unsigned P values; it does not infer strand or allele effects.

Default `magma` mode uses the following rules:

- Require at least 50 reference individuals. Keep polymorphic reference SNPs
  with at most 25% missing calls and adjusted sample variance above `1e-8`.
  The variance is `sum((observed - observed_mean)**2) / observed_count * n/(n-1)`.
  The 25% threshold follows MAGMA's **summary-statistic** path; its 5% threshold
  for other analysis paths does not apply here.
- Accept finite SNP P values in `[0, 1]`, then clip to `[1e-50, 1-1e-5]`.
  Literal `NA` and `-1` mark missing P; other invalid P values raise an error.
- Round positive SNP N to the nearest integer, with halves rounded upward,
  then retain only `N > 50`. Thus 50.49 is excluded and 50.5 becomes 51.
  The floating-point implementation uses `floor(abs(N) + 0.5)` with the
  original sign, matching the investigated MAGMA binary at half boundaries.
  Literal `NA` marks missing N; nonfinite values or rounded values outside
  `[0, 2**31-1]` raise an error when the row reaches N validation.
- Process each matched ID in input order. If its first row is excluded for
  missing P/N or low N, later rows cannot restore it. If its first row is
  accepted, a repeat removes it before the repeated row's values are parsed.
  These decisions are independent of parsing chunk size. Duplicate QC counts
  distinguish repeated accepted IDs from IDs already excluded by their first row.

`whole` mode instead requires two reference individuals, finite `0 < P <= 1`
and `N >= 50`, and excludes all duplicate IDs. It removes reference SNPs with
fewer than two observed calls or zero variation, without a missing-rate cutoff.
It does not clip accepted P values or round N.

## SNP statistic and LD

For each included SNP j, compute
`q_j = (-ndtri_exp(log(P_j) - log(2)))**2`, equivalent to
`chi2.isf(P_j, 1)`. In `magma` mode P here is the clipped input probability.
The block or whole-gene statistic is the sum of its SNP statistics.

Mean-impute reference genotypes, center each column, and divide by its L2 norm
to obtain G. With complete data, `R = G.T @ G` has unit diagonal. In `whole`
mode this also defines LD for missing data. In `magma` mode, adjust each
cross-product by `sqrt(n_i*n_j)/n_ij`, where n_i and n_j count observed calls
and n_ij counts jointly observed calls. Centering still uses each SNP's own
observed mean. This correction can produce correlations outside [-1, 1] and
a matrix that is not positive semidefinite.

For block eigendecomposition, clip individual correlations to [-1, 1] and
retain positive eigenvalues above the block cutoff below. Brown's variance
and covariance moments use the **unclipped** corrected correlations instead.
With no missing calls, the smaller of `G.T @ G` and `G @ G.T` gives the same
nonzero eigenvalues; missing-call correction requires the SNP-space matrix.
The `whole` spectrum discards only numerical-null eigenvalues below
`4 * eps * max(G.shape) * max(lambda_max, 1)`.

For a positive semidefinite R, `Z ~ N(0, R)` implies the quadratic-form null
`sum(lambda_j * chi2_1_j)`. Missing-call correction and spectral truncation
form part of the implemented approximation. None of these calculations proves
calibration under arbitrary reference mismatch or variable GWAS cohort overlap.

## Statistical blocks and aggregation

In default `magma` mode, the maximum statistical block size is
`min(floor(reference_sample_size / 2), 1000)`. Split a gene into the smallest
number of balanced, contiguous blocks satisfying this limit, in reference SNP
order after trait-specific selection. Earlier blocks receive remainder SNPs.
This partition is independent of the `--block-snps` genotype I/O cache setting.

Each block uses the SNP statistic and correlation spectrum described above,
retaining eigenvalues above `1e-4 * sum(positive_eigenvalues) / block_SNP_count`.
Compute each block's P value with the numerical routines below. One-block genes
return that result directly. Multi-block genes combine `-2 * sum(log(P_block))`
using Brown's correlation-adjusted chi-square approximation. Block correlations
are estimated from squared cross-block LD, with MAGMA's deterministic SNP
subsampling and size adjustment; the resulting block correlations are clipped
to [0, 1]. For B blocks and correlations rho, the combining distribution has mean `2*B` and variance
`4*B + 2*sum(rho*(3.25 + 0.75*rho))` over distinct block pairs.
No full gene-by-gene or full SNP-by-SNP correlation matrix is retained.

The combination uses log probabilities to preserve deep tails. A Brown result
is an approximation even when each block's tail probability is computed by
accurate integration; its method label records the block numerical methods.
`whole` mode performs one unpartitioned test with numerical-null truncation only.

## Probability evaluation

1. A single-SNP gene returns its accepted SNP P (after clipping in `magma` mode).
2. Zero statistic gives P=1 and Z=-infinity. Rank-one and exactly equal spectra
   use the scaled chi-square distribution; incomplete-gamma evaluation preserves log probabilities near either endpoint.
3. When `q <= 0.2*min(lambda)`, evaluate the lower-tail Gaussian integral
   over an ellipsoid with a rapidly convergent Taylor series, then use `log1p`
   to obtain log survival. This avoids cancellation for probabilities near one
   and is labeled `small_q_series`.
4. Otherwise, invert the exponentially tilted Laplace transform.
   For any `0 < t < 1/(2*max(lambda))`, write
   `K(s) = -0.5*sum(log(1-2*lambda*s))`. Then

   ```text
   P(Q >= q) = exp(K(t)-t*q) / pi
               * integral_0^infinity Re[exp(K(t+iu)-K(t)-iu*q)/(t+iu)] du.
   ```

   The exponential factor is carried in log space. Above the mean, choose the
   saddlepoint tilt `K'(t)=q`; at/below the mean, use `t=0.1/sum(lambda)`.
   Rescale frequency by `sqrt(K''(t))`, split the real integrand into sine and
   cosine components, and use QUADPACK's weighted Fourier integrator. This
   avoids subtracting two values near one half in the upper tail. Accept only
   finite positive integrals with summed estimated relative error <= 1e-7.
   QUADPACK's estimates are diagnostics, not rigorous bounds. If the cached
   callbacks fail this check, retry the original single-component callbacks
   at the same tilt and tolerances before trying another integration route.
5. If tilted integration fails its criterion, try ordinary Imhof integration
   in the bulk. Accept only when `1e-6 < P < 1` and estimated relative error
   is below 1e-4. A positive-tilt Chernoff bound below 1e-6 skips this ordinary
   cancellation-prone integral.
6. If neither integration route succeeds, use the Lugannani-Rice survival
   approximation `SF(w) + phi(w) * (1/u - 1/w)`, where
   `w=sign(t)*sqrt(2*(t*q-K(t)))` and `u=t*sqrt(K''(t))`.
   The mean limit includes skewness. Positive-w evaluation uses `erfcx`
   in log space; negative-w evaluation avoids erfcx overflow.
7. If the saddlepoint fails too, use LTZ moment matching, explicitly labeled
   `ltz_fallback`. Invalid/nonfinite probabilities raise an error; they are
   never replaced by a small P value.

For one-block and `whole` results, `PMETHOD` identifies each route:
`exact1snp`, `exact_zero`, `exact_rank1`, `exact_equal`, `small_q_series`,
`tilted_imhof`, `imhof`, `saddlepoint`, or `ltz_fallback`. Saddlepoint and LTZ
are approximations. Stability at extreme magnitudes does not guarantee accuracy,
monotonicity at every method-switch boundary, or correct ranking across spectra.
The numerical test suite compares two-weight spectra to independent
polar-coordinate integration, repeated-weight spectra to a beta-direction
integral, and exact cases to high-precision incomplete gamma.
Multi-block results use `brown:` followed by their component numerical methods,
so an approximate block tail remains visible in the combined result's label.

## Execution and memory

The quadrature callbacks execute in C; SciPy still controls integration with the
same formulas, tolerances, error checks, and fallback rules above. The sine and
cosine callbacks share a bounded cache of 4,096 nodes, keyed by the exact double
representation. Revisited nodes reuse both components; up to eight bounded
probes resolve hash collisions before replacement and recomputation. There is
no interpolation or change to QUADPACK's node selection
or error estimates. The approximately 96 KiB cache lives only for that integral.

Plain whitespace GWAS tables pass through a native reference-SNP filter, then
bounded string-array chunks and native decimal conversion through CPython's
float parser. Pandas is imported only for general table syntax or output
merging. Both paths feed one input-QC implementation, preserving source row
counts, duplicate order, missing-value rules, and sample-size rounding. Native
filtering uses byte-key membership checks before parsing matched numeric fields.

Gene annotations are streamed one gene at a time during analysis. Numerical
integration is imported after input loading so initialization does not overlap
with temporary parsing buffers.

The BIM file is streamed into one reference-order SNP lookup shared by
annotation mapping and trait input. Unused PLINK metadata arrays are not
materialized. Genotype normalization modifies only newly owned read buffers;
the public normalization helper still copies its input. Complete blocks retain
no missing-call mask, and gene assembly avoids a final copy when all selected
SNPs pass QC. Genes contained in one cached block select their retained columns
directly into an independent result array.

For complete genotypes, BLAS SYRK forms only the lower Gram triangle in Fortran
layout. LAPACK's `dsyevr` computes its eigenvalues, using the same solver as the
previous high-level symmetric eigensolver. Optimal workspace sizes are cached
for at most 256 matrix dimensions; matrix work buffers are not retained. Missing
calls still use the full adjusted SNP correlation matrix before eigendecomposition.
The default parsing chunk is 32,768 retained rows and the genotype cache is
8 MiB; neither changes the statistical gene blocks.

## Output interpretation

- `LOG10P`: log10(P), not minus log10(P).
- `P`: exponentiation of the log probability, with natural float64 underflow.
  There is no artificial cutoff at 1e-300.
- `ZSTAT`: `-ndtri_exp(log(P))`, avoiding manual asymptotic inversion.
- `NPARAM`: in `magma` mode, each block uses MAGMA's concentration-based
  parameter count and the multi-block result uses the correlation-adjusted
  aggregate, bounded below by the largest block count. In `whole` mode it is
  rounded effective rank `(sum(lambda))**2 / sum(lambda**2)`.
- `N`: in `magma` mode, round the mean of the already-rounded included SNP
  sample sizes, with halves upward. In `whole` mode, retain their arithmetic
  mean. This field does not model varying cohort overlap.
- `NSNPS`: number of included polymorphic SNPs, before spectral rank reduction.

Reference-panel ancestry, gene annotation windows, sample overlap, and GWAS QC
remain the analyst's responsibility. Gene tests do not account for multiple
hypothesis testing; output P values are unadjusted.

## References

- MAGMA: de Leeuw et al. (2015), DOI: 10.1371/journal.pcbi.1004219.
- Imhof (1961), DOI: 10.1093/biomet/48.3-4.419.
- Lugannani and Rice (1980), DOI: 10.2307/1426607.
- The survival-function sign is also explicit in the
  [JRSS B saddlepoint treatment](https://academic.oup.com/jrsssb/article/64/1/31/7098295).
- [SciPy ndtri_exp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri_exp.html).
