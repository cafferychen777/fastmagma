# Runtime comparison with MAGMA

The current implementation (commit `6b43874`) improves both runtime and peak
memory against the previous optimized release, R2, when rerun in the same
compute allocation. Prefiltered single CAD fell from **5.11 to 4.76 s** and
**154 to 138 MiB**; AD fell from **2.38 to 2.27 s** and **134 to 117 MiB**.
These are approximately **7% and 5% less elapsed time**, respectively, with
**10% and 13% less peak RSS**. All five workloads improved in both median
runtime and peak memory, including the full original GWAS inputs.

MAGMA remained faster and smaller for the single traits: **4.20 s / 24 MiB**
for CAD and **1.32 s / 16 MiB** for AD. The different-SNP pair was close to
parity. For the full original GWAS pair, fastmagma took **8.46 s** versus
MAGMA's **11.37 s**, a **1.34×** speed ratio on this workload. These results
do not establish a general speed advantage over MAGMA.

## Same-allocation comparison

Job `2695497` ran current fastmagma, previous R2 fastmagma, and MAGMA on node
`c03` with the same one-CPU allocation, inputs, thread settings, cache warming,
and fresh-process measurement. Previous/current execution order alternated
across three repetitions for every workload. Wall seconds below are medians,
with observed ranges in brackets.

| Workload | Previous R2 fastmagma | Current fastmagma | Paired MAGMA |
| --- | ---: | ---: | ---: |
| Single CAD | 5.11 [5.10–5.13] | 4.76 [4.74–4.79] | 4.20 [4.20–4.24] |
| Single AD | 2.38 [2.36–2.45] | 2.27 [2.23–2.28] | 1.32 [1.32–1.34] |
| Shared SNP pair | 2.94 [2.92–2.96] | 2.81 [2.80–2.82] | 2.65 [2.64–2.67] |
| Different SNP pair | 6.02 [5.95–6.03] | 5.59 [5.57–5.63] | 5.53 [5.52–5.61] |
| Full GWAS pair | 8.86 [8.80–9.22] | 8.46 [8.43–8.56] | 11.37 [11.29–11.95] |

Peak RSS in MiB is the median of each repetition's process peak, with observed
ranges. A serial multi-trait run uses its largest process peak, not their sum.

| Workload | Previous R2 fastmagma | Current fastmagma | Paired MAGMA |
| --- | ---: | ---: | ---: |
| Single CAD | 153.64 [153.64–157.10] | 138.16 [134.89–139.52] | 24.05 [23.00–24.58] |
| Single AD | 134.03 [131.48–134.35] | 116.76 [115.77–119.62] | 15.77 [15.42–16.12] |
| Shared SNP pair | 138.00 [137.50–140.40] | 122.94 [122.62–123.03] | 15.97 [15.86–16.34] |
| Different SNP pair | 173.98 [168.77–177.94] | 155.13 [153.51–160.48] | 23.93 [22.95–25.32] |
| Full GWAS pair | 171.85 [171.76–174.62] | 158.06 [155.66–158.38] | 25.23 [23.54–25.32] |

No repetition is discarded. The full-input control is included specifically to
check that single-trait and memory optimizations do not hide a genome-wide
input-scanning regression: its median improved from **8.86 to 8.46 s**.
Earlier kernel-only savings did not reduce end-to-end RSS; streaming annotations
and delaying analysis imports until input loading finished produced the memory
reduction reported here.

The shared pair took **2.81 s** batched versus **4.53 s** in two independent
fastmagma processes: a **1.61×** within-program speedup, or **37.9% less elapsed
time**. Eigendecompositions fell from **2,206 to 1,103** in every repetition.
This demonstrates reuse, but its timing benefit also includes avoided startup,
imports, and input work.

Relative to the original implementation in historical job `2695039`, current
median runtimes are **4.95–9.17×** faster across the five workloads. That baseline
ran on the same node in a separate allocation; these historical ratios are not
concurrent paired estimates. The tables above use the more recent R2 release
as their directly remeasured control.

## Bottlenecks and changes

The original filtered-CAD profile (job `2695103`) spent **21.974 s** of **29.202 s**
in `tilted_logsf`, with approximately **1.87 million** Python integrand callbacks.
All eigendecompositions together took only **1.032 s**. Native callbacks and
bounded exact-node reuse removed much of that overhead. Cache collisions cause
a lookup or recomputation; values are never interpolated and error criteria
are not relaxed.

The current CAD profile took **5.734 s**, including **1.252 s** in `tilted_logsf`,
**1.389 s** in correlation spectra, **0.878 s** in genotype reads, and **0.697 s**
in imports. The preceding R2 profile took **6.347 s**, including **1.363 s** in
`tilted_logsf` and **1.503 s** in correlation spectra. These separate profiles
include instrumentation overhead and nested calls; their times must not be
added or substituted for the paired uninstrumented measurements.

Plain input tables use bounded chunks without importing pandas; unusual CSV
syntax retains the general parser. Reference filtering preserves original row
counts and QC rules. Numeric conversion writes directly into its destination
array in C. A single reference lookup avoids duplicate metadata arrays.
Gene annotations stream through validation and analysis instead of retaining
all SNP strings. Analysis imports occur after input loading so their memory
does not overlap the parser's peak temporary allocations.

Genotype normalization reuses the owned decode buffer; complete blocks need
no missing mask, and the single-cache-block path removes an intermediate copy.
The eigenvalue calculation forms only the required Gram triangle and calls the
same LAPACK solver directly with cached workspace sizes. Defaults remain
32,768 retained input rows per chunk and an 8 MiB genotype cache. The statistical
model, SNP selection, integration tolerances, and fallback criteria are unchanged.

An independent 35-case two-weight integration check had maximum relative
probability error **1.36e-12**. Across all five real workloads, current and
original fastmagma outputs retained identical gene sets, NSNPS, NPARAM, N, and
PMETHOD. The largest absolute Z change was **1.88e-9** and the largest absolute
P change **6.75e-10**. If shared-node quadrature misses its error criterion, the
same integral is retried uncached at the same tolerances before the existing
numerical fallbacks. These checks support close numerical agreement on the
tested cases, not a universal error bound.

## Workloads

All workloads use the same 489-sample 1000 Genomes EUR chromosome-22 PLINK
reference and the same chromosome-22 gene annotation with 10 kb windows. CAD and
AD retain their own observed P values and sample sizes. The reference contains
141,123 SNPs and the chromosome annotation contains 1,263 genes. CAD and AD
match 128,088 and 17,060 reference SNP IDs, respectively; the shared input
contains 16,922 SNP IDs.

| Workload | Traits | P-value input |
| --- | --- | --- |
| Single CAD | CAD | Original rows restricted to reference SNP IDs |
| Single AD | AD | Original rows restricted to reference SNP IDs |
| Shared SNP pair | CAD and AD | Both restricted to the intersection of their reference SNP IDs |
| Different SNP pair | CAD and AD | Each retains its own reference SNP IDs |
| Full GWAS pair | CAD and AD | Original genome-wide P-value files |

Filtering preserves row order, duplicate rows, and missing values. Preparation
is outside timing for both programs. The first four workloads measure analysis
from already restricted input files; they do not include the cost of creating
those files. The full GWAS workload includes each program's scanning and parsing
of the original genome-wide inputs, while its gene analysis still covers only
chromosome 22.

The shared SNP pair additionally compares batched fastmagma with two independent,
serial fastmagma invocations using exactly the same shared input files. The
reported eigendecomposition counts show whether LD preparation is reused.
The runtime difference also includes savings from process startup, imports,
input handling, and other shared work; it cannot all be attributed to LD reuse.

## Measurement

- Arseven Slurm job `2695497`, node `c03`: one allocated CPU and a 3 GiB memory
  limit. R2 and current fastmagma were both run within this allocation for all
  five workloads, alongside MAGMA.
- Recorded CPU affinity was logical CPUs `2,130`. One thread is configured for
  fastmagma and the BLAS/OpenMP environment of both programs. MAGMA uses one
  process per trait.
- Three paired repetitions per workload. MAGMA and fastmagma run in alternating
  order; the shared-input serial control rotates through the three positions.
  Benchmark processes run serially within the allocation.
- All command inputs are read immediately before each timed process to warm the
  filesystem cache. There is no untimed analysis warmup. These measurements are
  not cold-storage benchmarks.
- Each timed invocation is a fresh process. Timing includes startup, library
  imports, input parsing, statistical computation, and writing gene results.
- MAGMA uses `--genes-only` to suppress gene–gene correlation computation, which
  fastmagma does not provide. Both programs use their default SNP-wise mean
  gene-analysis model. Neither annotation generation nor cross-chromosome
  merging is included.
- A fresh, standard-library-only Python wrapper uses `os.wait4` to obtain the
  target process's user CPU time, system CPU time, and peak resident memory;
  `time.perf_counter` measures elapsed time. The wrapper avoids inheriting the
  benchmark driver's scientific-library memory footprint into the measured
  process. Linux peak RSS values are reported in KiB.
- For serial multi-trait invocations, wall and CPU times are summed; peak RSS is
  the maximum process peak, not the sum. Summary timings use the median of the
  three repetitions and retain their range.

The wrapper's own startup, cache warming, input preparation, and output comparison
are outside timing. The allocation does not reserve the entire node, so the
measurements do not establish performance on an otherwise idle machine.

## Output checks

All repetitions matched gene IDs, SNP counts, retained parameter counts, and
sample sizes exactly. The full and different-SNP inputs produced 1,229 CAD and
1,100 AD gene results; shared SNP inputs produced 1,097 results per trait. Every
comparison retained the same top 20 genes and had no crossings of P = 0.05.

| Input / trait | Maximum absolute Z difference | Top-20 overlap | P = 0.05 crossings |
| --- | ---: | ---: | ---: |
| Original SNP sets, CAD | 0.0000568631 | 20 / 20 | 0 |
| Original SNP sets, AD | 0.00229333 | 20 / 20 | 0 |
| Shared SNP set, CAD | 0.0128991 | 20 / 20 | 0 |
| Shared SNP set, AD | 0.00229333 | 20 / 20 | 0 |

The original SNP-set rows apply to the single-trait, different-SNP pair, and full
GWAS workloads. Shared batched and serial comparisons gave the same agreement
statistics. Statistical compatibility does not imply bitwise equality of probability
calculations. The independent checks in [VALIDATION.md](VALIDATION.md) explain
previously identified small-gene differences. The larger shared-input CAD
difference has now also been independently resolved: for gene **56659**, angular
integration and inverse Laplace inversion agree on **P = 5.49110465704e-5**.
fastmagma agrees to relative error **5.33e-15**, whereas MAGMA reports
**5.7882e-5**, approximately **5.41% higher**.

A separate oracle run (`2695177`) examined this CAD case and four sparse-input
LDL cases: genes **55839, 56534, 56629, and 56846**. De Hoog and Talbot inverse
Laplace methods agreed at 50- and 80-digit precision; independent angular
integration also agreed for the four rank-two/rank-three cases. Across all five,
fastmagma's maximum relative P error was **1.26e-13**. For example, LDL gene
56534 has oracle P **1.52664018918e-16**, while MAGMA reports **5e-10**. These
results identify numerical probability error in the official outputs for these
specific cases. They do not diagnose every discrepancy in the larger
multi-trait experiments. The reproducible check is
[`validate_tail_probabilities.py`](../benchmarks/validate_tail_probabilities.py).

## Extended multi-trait experiments

The completed sparse experiment (job `2695160`, node `c03`) used **R1** and
427 common usable SNPs across CAD, LDL, T2D, BMI, TG, SCZ, SBP, and DBP. Groups
of 1, 2, 4, and 8 traits use prefixes of this list on the same fixed SNP set,
retaining each trait's observed P/N values. Preparation is outside timing.
These measurements must not be combined with current timings as one scaling curve.

| Traits | R1 fastmagma, seconds | MAGMA, seconds | R1 fastmagma peak MiB | MAGMA peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.78 [1.78–1.79] | 0.25 [0.25–0.25] | 212.19 [211.27–213.24] | 14.35 [13.79–14.62] |
| 2 | 2.10 [1.96–2.41] | 59.26 [58.51–59.79] | 215.57 [214.91–216.51] | 54.56 [53.21–55.04] |
| 4 | 2.23 [2.21–2.63] | 67.01 [66.99–69.11] | 222.59 [222.46–226.20] | 53.36 [53.09–53.92] |
| 8 | 2.75 [2.71–2.79] | 160.33 [158.68–160.33] | 240.19 [238.35–242.14] | 53.02 [52.79–53.96] |

Eight batched traits took **2.75 s** versus **14.78 s** for serial R1 fastmagma
(**5.38×**), with eigendecompositions reduced from **3,040 to 380**. However,
the large advantage over MAGMA is strongly influenced by numerical probability
cost: the sparse LDL run alone took about 59 seconds. This is not a measure of
pure LD-reuse benefit, and the single CAD trait was substantially faster in MAGMA.

All repetitions matched gene sets, NSNPS, NPARAM, and N, with no P = 0.05
crossings. Probability agreement was less uniform: the largest absolute Z
difference across eight traits was **7.028**, and the lowest top-20 overlap was
**16/20**. The five independently checked cases above establish official numerical
errors in those cases only; other extended-experiment discrepancies remain
undiagnosed. These sparse inputs do not represent dense LD workloads. A denser
common-SNP experiment did not complete its required repetitions and is excluded
from these tables; no aggregate timing is claimed for it.

## Reproduction

Run the benchmark on a compute allocation with one CPU, the installed fastmagma
package and dependencies, and the official MAGMA executable. The script imports
its existing output-comparison helper from the same `benchmarks` directory.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python benchmarks/compare_runtime.py \
  --magma /path/to/magma \
  --bfile /path/to/1000G.EUR.QC.22 \
  --annot /path/to/annot_w10.genes.annot \
  --pval-dir /path/to/pval \
  --out-dir /path/to/runtime-results \
  --threads 1 --repetitions 3
```

The P-value directory must contain `cad.pval` and `ad.pval`, each with `SNP`, `P`,
and `N` columns. Use an output directory dedicated to this benchmark. The script
writes process logs, per-process measurements, gene results, and incremental
`results.json`. Re-running with the same arguments resumes completed measurements
whose output files still exist. Use a fresh directory when changing software or
input files.

For the multi-trait experiment, use `compare_multitrait_runtime.py` with the same
arguments and an explicit `--traits` list containing eight unique identifiers.
Use a separate output directory for each trait set and software build. Inspect
the recorded common usable SNP count before interpreting scaling.

These workloads cover one chromosome and one thread. They do not establish
whole-genome runtime, cold-cache throughput, or scaling across thread counts.

To reproduce a selected unblocked-gene tail check, install the optional test
dependency `mpmath` and provide a JSON list identifying exact input/output files:

```json
[{
  "trait": "cad", "gene": "56659",
  "annotation": "/path/to/chr22.genes.annot",
  "pval": "/path/to/shared/cad.pval",
  "fast": "/path/to/cad.chr22.fastmagma.tsv",
  "official": "/path/to/cad.genes.out"
}]
```

```bash
python benchmarks/validate_tail_probabilities.py \
  --bfile /path/to/1000G.EUR.QC.22 \
  --cases /path/to/cases.json --out /path/to/tail-oracle.json
```

Use the exact SNP-filtered P-value files for the reported analysis. The checker
reconstructs LD independently and rejects unexpected QC or blocked genes rather
than silently evaluating a different test.
