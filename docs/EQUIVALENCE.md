# Algorithm equivalence investigation

Date: 2026-09-24. Controlled comparisons informed the default implementation;
independent integrals distinguish statistical-model differences from numerical errors.

## Conclusion

The large discrepancies in the original implementation are explained by a missing step in MAGMA-py's
reproduction of **default MAGMA v1.10**: automatic within-gene SNP blocking,
followed by combination of block P values. They are not evidence that MAGMA's
matrix calculations are incorrect. MAGMA-py computes the whole-gene quadratic
form accurately in the examined cases, but that is a different statistical test.
Independent angular integrals also resolve three remaining low-SNP discrepancies
in favor of MAGMA-py's probabilities.

Sources were the [official v1.10 source distribution and binaries](https://cncr.nl/research/magma/),
the manual, and controlled runs on the same reference panel, annotation and GWAS
inputs. Older v1.0 source was not used to infer current behavior. No MAGMA source
or binaries have been copied into this package.

## Default blocking explains the large differences

For reference sample size n, default block size is min(floor(n/2), 1000).
Genes above this size are divided into approximately equal contiguous SNP
blocks. Each block receives its own SNPwise-mean test. Block probabilities are
then combined by a correlation-adjusted Fisher/Brown calculation, using
estimated block correlations. This is not algebraically equal to evaluating
one weighted chi-square tail for all SNPs jointly.

Here n=489, so the limit is 244 SNPs. For example, 286 SNPs become 143+143;
262 become 131+131; 520 become 174+173+173. The distinction matches the observed
size dependence of the discrepancies.

Relevant v1.10 source locations are `parse.cpp:136–137`,
`genemodelengine_compound.cpp:34–69,113–144`, and
`engineutils.cpp:169–184`. These are pointers into the separately downloaded
official source, not files distributed with MAGMA-py.

| Trait / gene | SNPs | MAGMA-py whole gene | MAGMA default | MAGMA blocking disabled |
|---|---:|---:|---:|---:|
| CAD / 56553 | 286 | 0.97625993 | 0.92434 | 0.97626 |
| CAD / 55819 | 347 | 0.0457177444 | 0.063059 | 0.045718 |
| CAD / 55836 | 256 | 0.0810181556 | 0.049113 | 0.081019 |
| CAD / 56458 | 520 | 0.0558654499 | 0.037617 | 0.055866 |
| AD / 56913 | 262 | 0.687226543 | 0.5823 | 0.68723 |
| AD / 56512 | 525 | 0.688617253 | 0.74432 | 0.68863 |

Use the retained CSV for full precision; official text output is rounded.
The disabled-block run is a diagnostic intervention, not the official default.

Independent blockwise reproduction (retained with the local diagnostic evidence)
gave CAD56553 P=0.9243360148 and NPARAM=8, and AD56913 P=0.5822991232 and
NPARAM=43, reproducing the rounded official default results. This two-way check
both removes the difference by disabling blocking and recreates it by adding
blocking to an independent calculation.

## Evidence and controls

- Twelve trait–gene cases were selected, including large discrepancies,
  nominal-threshold crossings and agreement controls.
- All selected genotype calls were nonmissing. Official normalized genotype
  dumps match the reference columns up to allele orientation and text rounding;
  all matched-column absolute correlations exceeded 0.9999999999.
- Whole-gene spectra derived from official dumps yield MAGMA-py's probabilities.
- Independent direct weighted-chi-square simulations used 200,000 draws per case.
  They support the whole-gene calculation; they do not test MAGMA's blockwise null.
- Independent Boost Imhof calculations on the same spectra agree for the large
  discrepancies. Official static and dynamic binaries, and an unchanged-source
  build, all reproduce the default outputs.
- A debugger captured a 143-by-143 correlation matrix while processing the
  286-SNP gene, confirming actual blockwise execution.
- For the controlled no-block run, GDB set `block_prop=0` and `block_max=100000`
  at `PartitionedEngine::init_gene`. Source code was unchanged. Setting zero
  through the ordinary debug parameter parser does **not** achieve this: zero
  is treated as an unset value and the default fraction is restored.

Slurm jobs: 2680391 (diagnostics and fresh default runs), 2680398 (genotype
dumps), 2680400 (dynamic binary), 2680401 (unchanged-source build), 2680402
(genotype matching), 2680404 (Eigen controls), 2680405 (block capture), and
2680407 (controlled blocking intervention). Job 2680406 was the ineffective
zero-parameter attempt and must not be treated as the no-block experiment.

Local evidence is retained under `results/algorithm_diagnostic/`, including
`model_comparison.csv`, `compare/genotype_comparison.csv`, independent block
results, low-rank references and reproduction scripts. Research outputs are
excluded from release archives.

## Independently resolved low-SNP discrepancies

AD56734 has two SNPs; AD56733 and AD56632 have three. These genes do not trigger
blocking, have no missing reference calls, and use SNP P values inside MAGMA's
clipping bounds. For a standard Gaussian vector, its squared radius is
chi-square distributed independently of its uniform direction. Integrating the
radial survival function over direction therefore gives an independent,
nonoscillatory reference probability.

The [validation record](VALIDATION.md#independent-resolution-of-small-gene-discrepancies)
contains the three probabilities and official comparisons.

For each case, Gauss-Legendre orders 64, 128, 256, and 512 agree with adaptive
angular integration. The maximum absolute MAGMA-py error against that reference
is approximately `2e-15`; official text rounding is too small to explain the
differences shown. AD56632 has a numerical-null eigenvalue; retaining that tiny
positive value in the independent integral gives the same result as removing it.

AD56733 additionally isolates the quadrature mechanism: an independent Boost
Imhof integral at default recursion depth 15 gives `0.000317818216645838`,
reproducing the official result, while depth 25 gives `0.000320459746894541`.
Float32 eigenvalues from the official genotype dump give `0.0003204601052`,
excluding ordinary spectrum rounding as the explanation. MAGMA's acceptance
check does not use the returned quadrature error estimate. This identifies a
numerical limitation in the examined case, not a general claim about MAGMA's
reliability. MAGMA-py keeps the independently supported probabilities rather
than reproducing these errors.

The reusable oracle is [benchmarks/validate_boundaries.py](../benchmarks/validate_boundaries.py).
Local raw results are in `results/boundary_validation/real_lowrank.json`.
Historical diagnostic scripts are retained locally under
`results/algorithm_diagnostic/reproduction/code/` and are excluded from release
archives.

## QC and missing-data boundaries

The default model now follows the investigated summary-statistic input rules:
P clipping, ordered duplicate exclusion, integer sample-size rounding, a strict
rounded-N threshold, and reference SNP filtering at **25% missing calls**.
Missing-call LD uses SNP-wise centering and joint-observation rescaling. Block
eigenvalues use clipped correlations; Brown moments use the unclipped values.
The [algorithm specification](ALGORITHM.md) defines these rules precisely.

Twenty-eight constructed cases were run against the official binary. Both
programs accepted 24 cases and rejected four invalid-input cases. Accepted gene
sets and `NSNPS`, `NPARAM`, and `N` matched throughout, including overlapping and
disjoint missing calls at 25%, exclusion at 26%, and a 60-SNP two-block example
whose corrected correlations exceed one. The largest absolute P difference was
`4.58e-6`. Relative agreement is not claimed: some extreme synthetic probabilities
expose an official numerical floor, whereas MAGMA-py retains log-space tails.
See the [validation record](VALIDATION.md) for the complete test scope.

## Release interpretation

The default `magma` model implements balanced SNP blocking, sampled block
correlations, Brown aggregation, spectrum truncation, MAGMA-style NPARAM, and
the investigated input and missing-data rules. `whole` preserves the original
unpartitioned calculation and its own input policy. Prepared spectra are reused
across traits sharing SNP sets. Log probabilities and numerical method labels
are retained through aggregation.

Completion metadata records both model and revision. Current `magma` outputs
use revision 2; merging rejects mixed models or revisions. Outputs without a
model are interpreted as `whole`, and older model-tagged outputs without a
revision are interpreted as revision 1.

This establishes agreement on the tested workflows and explains the examined
discrepancies. It does not establish bit-identical numerical behavior, every
MAGMA option, null calibration under reference mismatch or varying cohort
overlap, or compatibility with downstream MAGMA gene-set procedures.
