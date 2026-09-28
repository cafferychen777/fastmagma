# fastmagma

<p align="center">
  <img src="docs/assets/fastmagma-logo.png" alt="fastmagma logo" width="600">
</p>

**Gene-level association testing across multiple GWAS traits.**

A Python implementation of the SNP-wise mean gene test used in
[MAGMA](https://doi.org/10.1371/journal.pcbi.1004219), built for analyzing many
traits against a shared LD reference. Give fastmagma GWAS P values, a PLINK
reference panel, and SNP-to-gene annotations; it returns gene P values, Z scores,
and a combined gene-by-trait matrix.

## Highlights

| Feature | What it provides |
|---|---|
| **Shared LD computation** | Traits with the same SNP set reuse each gene's eigendecomposition. |
| **MAGMA-style blocking** | Default within-gene SNP blocks and correlation-adjusted P-value aggregation. |
| **Controlled memory use** | Chunked input, disk-backed trait arrays, and a bounded genotype cache. |
| **Accurate block tails** | Log-space integration preserves extreme tails; Brown aggregation and numerical fallbacks are labeled. |
| **CPU and cluster support** | Explicit thread limits and independent chromosome jobs, with a Slurm example. |
| **Traceable results** | Per-run QC, numerical-method counts, input records, and output checksums. |

## Quick start

Install from the source checkout with **Python 3.10 or newer** and a C compiler
(GCC/Clang on Linux, Xcode Command Line Tools on macOS, or MSVC Build Tools on
Windows). Source installs compile two small extensions; installing a compatible
platform wheel does not require a compiler:

```bash
python -m pip install .
```

Analyze chromosome 22 for two traits:

```bash
fastmagma run \
  --chr 22 \
  --bfile-prefix /data/reference. \
  --annot /data/genes.annot \
  --pval-dir /data/gwas \
  --traits cad,ad \
  --out-dir results \
  --threads 4
```

Then combine the results into gene tables and a gene-by-trait Z-score matrix:

```bash
fastmagma merge --traits cad,ad --chrs 22 --out-dir results
```

For a genome-wide analysis, run chromosomes 1–22 and merge with `--chrs 1-22`.
See the [Slurm array example](examples/fastmagma_array.sbatch) for parallel jobs.
`python -m fastmagma` also works; use `fastmagma --help` for available commands.

The package runs on CPU with NumPy, SciPy, pandas, bed-reader, and threadpoolctl.
No MAGMA executable or EdgeMap installation is required. The first standalone
release is being prepared; the installation command above uses the local checkout.

## Inputs

| Input | Format | Example |
|---|---|---|
| LD reference | PLINK `.bed`, `.bim`, `.fam` per chromosome | `reference.22.bed` |
| Gene annotation | Tab-separated gene ID, location, and SNP IDs | `genes.annot` |
| GWAS summary statistics | One whitespace-separated `SNP P N` file per trait | `cad.pval`, `ad.pval` |

`--bfile-prefix /data/reference.` selects `/data/reference.22.bed` for chromosome
22, together with its `.bim` and `.fam` files. Reference SNP IDs must be unique.
Autosomes 1–22 are supported. The default model requires at least 50 reference
individuals.

Annotation rows use tabs between fields:

```text
GENE_A<TAB>22:100000:120000<TAB>rs1<TAB>rs2
GENE_B<TAB>22:200000:220000<TAB>rs3<TAB>rs4
```

Replace `<TAB>` with a literal tab. Gene windows are defined by the supplied
annotation; gene IDs are preserved as strings. Lines beginning with `#` are
comments.

Each GWAS file has a header and one row per SNP:

```text
SNP    P       N
rs1    0.001   120000
rs2    0.05    118500
```

By default, fastmagma follows MAGMA v1.10's summary-statistic QC: accepted P
values are clipped to `[1e-50, 1-1e-5]`, SNP sample sizes are rounded and must
exceed 50, and reference SNPs with more than 25% missing calls are excluded.
Duplicate IDs are excluded according to input order; missing-call LD includes
joint-observation correction. See the [input and LD rules](docs/ALGORITHM.md)
for exact behavior, including the separate `whole` policy. QC counts are saved
in the chromosome manifest. Reference panels, annotations, and GWAS datasets
are supplied by the user.

## Outputs

| File | Contents |
|---|---|
| `<trait>.chr<chr>.fastmagma.tsv` | Per-chromosome gene results |
| `<trait>.genes.out` | Results merged across requested chromosomes |
| `zstat_matrix.tsv` | Gene-by-trait Z scores |
| `chr<chr>.manifest.json` | Input records, QC, methods, timing, resource settings, and output checksums |
| `merge.manifest.json` | Chromosomes and traits included in the merge |

Gene tables contain:

| Columns | Meaning |
|---|---|
| `GENE`, `CHR`, `START`, `STOP` | Gene identifier and annotated interval |
| `NSNPS` | Number of included polymorphic SNPs |
| `NPARAM` | MAGMA-style block/aggregation parameter count; effective rank in `whole` mode |
| `N` | Rounded mean of rounded SNP sample sizes; unrounded mean in `whole` mode |
| `P`, `LOG10P`, `ZSTAT` | Unadjusted gene P value, log10(P), and corresponding normal Z score |
| `PMETHOD` | Numerical method used for the gene P value |

Use `LOG10P` for very small probabilities: the decimal `P` column can underflow
to zero. `LOG10P` is **log10(P)**, so more negative values indicate stronger
association.

The default `--model magma` uses MAGMA v1.10's balanced SNP blocks and Brown
aggregation. Use `--model whole` for the original unpartitioned quadratic-form
test. The model and its revision are recorded in completion manifests; merging rejects
mixed models or revisions. Statistical blocks are separate from the `--block-snps` I/O cache size.

Merging checks completion manifests and output checksums. Missing chromosomes
are errors unless `--allow-missing` is explicit. Existing results require
`--overwrite`; chromosome-level locks prevent conflicting writes while allowing
independent chromosomes to run together.

## Performance and memory

fastmagma avoids repeating the same LD work across traits. It also uses the
smaller of the sample-space and SNP-space Gram matrices for complete-data
correlation spectra; missing-call correction uses the SNP-space matrix.

Native numerical callbacks, streaming input and annotations, and reduced matrix
overhead made five chromosome-22 workloads **5.0–9.2×** faster than the original
implementation. In a direct rerun against the previous optimized release in the
same allocation, single CAD improved from **5.11 to 4.76 seconds** and
**154 to 138 MiB**; AD improved from **2.38 to 2.27 seconds** and
**134 to 117 MiB**. All five workloads improved in both median runtime and peak RSS.

With one CPU/thread and three repetitions, paired **MAGMA v1.10** (`--genes-only`)
still ran the single traits faster: **4.20 seconds** for CAD and **1.32 seconds**
for AD, using about **24 and 16 MiB**. From the original full GWAS input files,
the CAD/AD pair took **8.46 seconds** with fastmagma versus **11.37 seconds** with
MAGMA. This workload-specific advantage does not imply a general speedup.

The statistical model and requested integration accuracy are unchanged. Checks
retained gene sets, SNP/parameter/sample counts, and numerical method labels;
maximum absolute Z change versus the original implementation was **1.9e-9**.
See the [runtime report](docs/PERFORMANCE.md) for all repetitions, same-allocation
controls, remaining bottlenecks, and numerical checks.

In a synthetic benchmark with **150 reference samples and 1,500 SNPs**, the
smaller-matrix calculation took **0.95 ms**, compared with **188 ms** for the
full SNP matrix. Both returned matching nonzero eigenvalues. These are median
kernel timings over three repetitions on macOS ARM64 with one BLAS thread;
they do not represent an end-to-end or MAGMA speedup.

Reproduce the benchmark:

```bash
python benchmarks/benchmark.py --out results/benchmark.json
```

<details>
<summary><strong>Resource options and HPC guidance</strong></summary>

| Option | Default | Purpose |
|---|---:|---|
| `--threads` | `1` | Limit BLAS and genotype-reader threads |
| `--chunk-rows` | `32768` | Reference-matched GWAS rows parsed per chunk; all rows for general-parser fallback |
| `--cache-mb` | `8` | Genotype cache capacity in MiB; `0` disables caching |
| `--workspace-mb` | `512` | Numerical-array workspace estimate allowed per gene, in MiB |
| `--block-snps` | `256` | SNPs read per reference block |
| `--max-gene-snps` | `100000` | Maximum SNP count before a gene triggers an error |
| `--temp-dir` | System temporary directory | Location of disk-backed trait arrays |

Scratch arrays require `24 × traits × relevant reference SNPs` bytes. The
workspace setting bounds an array-size estimate, **not total process RSS**;
allow additional memory for parsing, Python objects, library workspaces, and
OS file caching.

For many simultaneous chromosome jobs, use chromosome-partitioned GWAS files
to reduce repeated input scans. Each job reads each supplied trait file once.
Choose a temporary directory with enough disk space, preferably node-local
scratch when available.

Temporary arrays are cleaned up on completion and handled errors. After an
abrupt termination, verify that the process has stopped before removing stale
`.fastmagma.*.lock` files or temporary directories.

</details>

## Validation and relationship to MAGMA

Tests cover analytic chi-square distributions, high-precision extreme tails,
independent reference integrals, missing genotypes, duplicate SNPs, resource
limits, and installed command-line workflows. On 35 deterministic two-weight
cases, the maximum relative P-value error against independent integration was
**1.4 × 10⁻¹²**. This describes those cases, not a universal error bound.

The default block model was compared with MAGMA v1.10 across chromosome 22:

| Trait | Matched genes | Maximum absolute Z difference | NPARAM agreement |
|---|---:|---:|---:|
| Coronary artery disease | 1,229 | 0.000057 | 100% |
| Alzheimer's disease | 1,100 | 0.00229 | 100% |

SNP counts and sample sizes matched for every gene, both top-20 sets matched,
and no results crossed P=0.05 between implementations. The largest remaining
AD difference is a two-SNP gene. Independent angular integration confirms
fastmagma's probabilities for that gene and two additional low-SNP discrepancies.

The earlier large discrepancies came from MAGMA's default within-gene SNP
blocking and block-P-value aggregation, absent from the original implementation.
The default `magma` model now includes that workflow; `whole` preserves the
original calculation. See the [algorithm-equivalence investigation](docs/EQUIVALENCE.md)
for controlled comparisons and a separately verified official integration error.

fastmagma implements gene-level SNP-wise mean testing. It is an independent
implementation, with its own numerical integration and explicitly labeled
saddlepoint/LTZ fallbacks. The default `NPARAM` follows the blockwise MAGMA
formula, and input QC, sample-size rounding, and missing-call LD follow the
investigated summary-statistic workflow. Twenty-eight official-binary boundary
comparisons agreed on accepted/rejected cases and accepted gene metadata.
Floating-point calculations and probability solvers remain independent; full
bit-identical equivalence is not claimed. MAGMA gene-set analysis is outside
the package's scope.

See the [algorithm specification](docs/ALGORITHM.md) for the statistical model,
QC rules, and numerical methods, and the [validation report](docs/VALIDATION.md)
for tested environments and empirical comparisons.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
ruff check src tests scripts benchmarks
python -m build
python -m twine check --strict dist/*
```

The source checkout includes utilities for
[comparison with MAGMA](scripts/validate.py),
[building an annotated gene matrix](scripts/build_gene_z_matrix.py), and
[inspecting LDL gene ranks](scripts/ldl_rank_check.py).

## Documentation and attribution

- [Algorithm and numerical methods](docs/ALGORITHM.md)
- [Validation and benchmarks](docs/VALIDATION.md)
- [Release procedure](docs/RELEASING.md)
- [Changelog](CHANGELOG.md)
- [Source provenance](docs/EXTRACTION.md)

The SNP-wise gene-analysis framework originates from
[de Leeuw et al., *PLOS Computational Biology* (2015)](https://doi.org/10.1371/journal.pcbi.1004219).
fastmagma was developed from the gene-analysis code in EdgeMap.

**Maintainer:** Chen Yang · `cafferychen777@tamu.edu`

**License:** [MIT](LICENSE), with the source project's copyright notice retained.
