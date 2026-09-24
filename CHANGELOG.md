# Changelog

## 0.1.0 — unreleased

- Add default MAGMA-style within-gene SNP blocking, block-correlation estimation,
  Brown aggregation and NPARAM; retain the previous test as `--model whole`.
- Match the investigated MAGMA summary-statistic QC: P clipping, ordered
  duplicate handling, SNP and gene sample-size rounding, a strict rounded-N
  threshold, and at least 50 reference individuals.
- Apply the 25% missing-call cutoff and joint-observation LD correction; use
  unclipped correlations for Brown moments and clipped values for block spectra.
- Record statistical model revisions and reject merges that mix models or
  revisions, including earlier `magma` outputs with different QC semantics.

- Extract the standalone package from EdgeMap, retaining its license notice.
- Correct whole-model genotype normalization to construct unit-diagonal
  correlations after mean imputation.
- Correct the Lugannani-Rice survival correction sign and the mean-limit handling.
- Evaluate normal inverse tails with ndtri_exp; support exact rank-one and
  equal-weight cases in log space. Remove the artificial 1e-300 display cutoff.
- Add exponentially tilted Fourier inversion for general spectra to evaluate
  deep tails without cancellation, with explicit numerical-error diagnostics.
- Add small-statistic Taylor-series evaluation to preserve probabilities and
  Z scores near P=1, with the `small_q_series` method label.
- Raise on numerical failure instead of manufacturing extremely significant P values.
- Add chunked summary-statistic QC, disk-backed numeric arrays, shared SNP indexing,
  a bounded genotype block cache, dual-Gram eigenvalues, thread controls, and
  explicit per-gene workspace guards.
- Precompute SNP statistics once per trait and reuse identical trait spectra.
- Validate inputs, preserve string gene IDs, reject incomplete merges by default,
  stage outputs, and record completion manifests and QC diagnostics.
- Separate numerical methods, genotype handling, input parsing, orchestration,
  and CLI. Replace conflicting historical algorithm prose.
- Add analytic, independent-reference, integration, resource, and packaging tests;
  add cross-platform CI and reproducible synthetic benchmarks.
- Validate 28 input/missingness boundary cases against official MAGMA and
  resolve three real low-SNP discrepancies using independent angular integrals.

Original extracted core SHA-256: `8436b112a6176bad336bbe012689bcf9d532cfefec71ca6e1ee21b79010e5cc0`.
