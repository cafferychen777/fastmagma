# Runtime comparison with MAGMA

Two optimization rounds removed Python quadrature callbacks, redundant parsing
and indexing, unnecessary genotype copies, and repeated evaluation at identical
integration nodes. The current implementation (R2) reduced runtime by
**4.52–8.47×** against the original fastmagma baseline across five chromosome-22
workloads, without changing the statistical model or requested integration accuracy.

The second round directly improved prefiltered single-trait analysis: CAD fell
from **7.08 to 5.22 s**, with peak RSS from **257 to 155 MiB**; AD fell from
**3.37 to 2.43 s**, with RSS from **216 to 136 MiB**. These are approximately
**26–28% less time and 37–40% less peak memory** than R1. MAGMA remained faster
and smaller for these single-trait workloads: **4.28 s / 25 MiB** for CAD and
**1.34 s / 16 MiB** for AD.

For the full original GWAS input pair, R2 took **8.86 s** versus paired MAGMA's
**11.35 s**, a **1.28×** speed ratio on this workload. This does not establish
a universal speed advantage. All four prefiltered workloads remained slower
than MAGMA, and fastmagma retained higher process memory throughout.

R2 results are from job `2695477` at production commit `831f66a`; R1 results
are from job `2695169` and match production commit `403f765`. The separate extended multi-trait results below
use R1, not R2.

## Results

Wall seconds are medians over three repetitions, with observed ranges in
brackets. Baseline, R1, and R2 were separate jobs on the same node; MAGMA was
remeasured alongside R2. Historical before/after ratios are not concurrent
paired estimates.

| Workload | Original fastmagma | R1 fastmagma | Current R2 fastmagma | Paired MAGMA |
| --- | ---: | ---: | ---: | ---: |
| Single CAD | 23.58 [23.36–23.71] | 7.08 [7.00–7.18] | 5.22 [5.17–5.42] | 4.28 [4.26–4.31] |
| Single AD | 14.70 [14.58–14.97] | 3.37 [3.30–3.39] | 2.43 [2.40–2.44] | 1.34 [1.33–1.36] |
| Shared SNP pair | 25.78 [25.37–26.42] | 4.06 [4.02–4.07] | 3.04 [3.04–3.05] | 2.68 [2.68–2.68] |
| Different SNP pair | 35.99 [34.95–36.97] | 7.80 [7.71–7.86] | 6.14 [6.01–6.15] | 5.56 [5.55–5.63] |
| Full GWAS pair | 57.28 [56.77–58.04] | 10.94 [10.84–10.94] | 8.86 [8.80–8.89] | 11.35 [11.32–11.41] |

Peak RSS in MiB is the median of each repetition's process peak, with observed
ranges. A serial multi-trait run uses its largest process peak, not their sum.

| Workload | Original fastmagma | R1 fastmagma | Current R2 fastmagma | Paired MAGMA |
| --- | ---: | ---: | ---: | ---: |
| Single CAD | 382.31 [381.62–386.33] | 257.27 [255.71–259.38] | 154.60 [153.62–155.96] | 24.60 [23.60–25.39] |
| Single AD | 354.16 [353.27–355.27] | 216.10 [214.98–220.20] | 135.52 [132.00–136.04] | 16.09 [15.12–16.43] |
| Shared SNP pair | 359.23 [358.84–360.75] | 226.30 [225.52–227.37] | 140.32 [138.86–141.46] | 15.29 [15.00–16.55] |
| Different SNP pair | 391.28 [390.36–394.86] | 262.50 [259.67–263.97] | 177.71 [177.36–178.95] | 23.88 [23.82–25.32] |
| Full GWAS pair | 411.21 [409.89–415.25] | 262.20 [259.67–262.75] | 174.33 [172.33–175.07] | 24.45 [24.41–25.10] |

The R2 shared pair took **3.04 s** batched versus **4.85 s** in two independent
fastmagma processes: a **1.59×** within-program speedup, or **37.3% less elapsed
time**. Eigendecompositions fell from **2,206 to 1,103** in every repetition.
This demonstrates reuse, but the timing benefit also includes avoided startup,
imports, and input work.

## Bottlenecks and changes

The baseline filtered-CAD profile (job `2695103`) spent **21.974 s** of **29.202 s**
in `tilted_logsf`, with approximately **1.87 million** Python integrand callbacks.
All eigendecompositions together took only **1.032 s**. Optimizing eigenvalues
alone could not address the dominant cost.

R1 moved the unchanged callback formulas into C. R2 additionally caches both
Fourier components at exactly matching quadrature nodes in a bounded shared
cache. Hash collisions cause recomputation; there is no interpolation or
relaxation of error estimates. The R2 CAD profile took **6.347 s**, including
**1.363 s** in `tilted_logsf`, **1.503 s** in correlation spectra, **0.912 s** in
genotype reads, and **0.719 s** in imports. The corresponding R1 tilted-integration
time was **2.207 s**. Profile times include instrumentation overhead and nested
calls; they must not be added or substituted for the uninstrumented table.

Plain input tables now use bounded array chunks without importing pandas;
unusual CSV syntax retains the general parser. Reference filtering preserves
original row counts and the existing QC rules. A single dictionary owns SNP
lookup and ordering, while the BIM reader skips unused metadata. Genotype
normalization reuses the owned decode buffer; complete blocks need no missing
mask, and retained gene matrices avoid redundant copies. Default parser chunks
remain 32,768 retained rows and the genotype cache remains 8 MiB. These execution
changes do not alter statistical SNP blocks or selection rules.

An independent 35-case two-weight integration check had maximum relative
probability error **1.36e-12**. Across all five real workloads, R2 and original
fastmagma outputs retained identical gene sets, NSNPS, NPARAM, and N. The largest
absolute Z change was **1.88e-9** and the largest absolute P change **6.75e-10**.
PMETHOD labels also matched for every gene. If shared-node quadrature misses
its error criterion, the same integral is retried uncached at the same tolerances
before considering the existing numerical fallbacks. These checks support close
numerical agreement on the tested cases, not a universal error bound.

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

- Arseven Slurm jobs `2695039` (baseline), `2695169` (R1), and `2695477` (R2), node `c03`:
  one allocated CPU and a 3 GiB memory limit per job.
- Recorded CPU affinity was logical CPUs `21,149` for the baseline, `25,153`
  for R1, and `3,131` for R2
  exposed by the allocation. One thread is configured for fastmagma and the
  BLAS/OpenMP environment of both programs.
  MAGMA is invoked as a single process per trait.
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
These measurements must not be combined with R2 timings as one scaling curve.

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
