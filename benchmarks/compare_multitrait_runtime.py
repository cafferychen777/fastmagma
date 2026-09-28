"""Measure 1/2/4/8 real traits on one common reference-SNP set.

All groups are prefixes of the same ordered trait list and use the intersection
of SNPs passing the existing MAGMA input QC in all eight traits. Original P and
N values are retained. Filtering and QC preparation are outside timing. This
controlled sharing experiment complements the different-mask/full-input cases
in compare_runtime.py; it is not a substitute for those end-to-end workloads.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from compare_runtime import filter_pvalues, run_tool, save, summarize

TRAITS = ["cad", "ldl", "t2d", "bmi", "tg", "scz", "sbp", "dbp"]


def prepare(args):
    import numpy as np

    from magma_py._input import filter_pval
    from magma_py.io import TraitStore

    traits = args.traits
    directory = args.out_dir / "inputs"
    filtered, shared = directory / "filtered", directory / "shared"
    filtered.mkdir(parents=True, exist_ok=True)
    shared.mkdir(exist_ok=True)
    snps = [line.split()[1] for line in Path(str(args.bfile) + ".bim").read_text().splitlines()]
    annotation = directory / "chr22.genes.annot"
    genes = 0
    with args.annot.open() as reader, annotation.open("w") as writer:
        for line in reader:
            fields = line.split()
            if line.startswith("#"):
                writer.write(line)
            elif len(fields) >= 2 and fields[1].split(":")[0] == "22":
                writer.write(line)
                genes += 1
    if not genes:
        raise ValueError("No chromosome-22 genes in annotation")
    source_rows = {}
    for trait in traits:
        source, destination = args.pval_dir / f"{trait}.pval", filtered / f"{trait}.pval"
        source_rows[trait] = filter_pval(source, snps, destination)
        if source_rows[trait] is None:
            raise ValueError(f"Expected plain whitespace benchmark input: {source}")
    store = TraitStore(directory, traits, snps, model="magma")
    try:
        store.load(filtered, 100000)
        usable = np.isfinite(store.values[:, 0, :]).all(axis=0)
        common = {snp for snp, keep in zip(snps, usable) if keep}
        qc = store.qc
    finally:
        store.close()
        store.path.unlink()
    if not common:
        raise ValueError("No common usable reference SNPs across the eight traits")
    for trait in traits:
        present, rows = filter_pvalues(filtered / f"{trait}.pval", shared / f"{trait}.pval", common)
        if present != common or rows != len(common):
            raise ValueError(f"Common inputs are not unique and complete for {trait}")
    return annotation, dict(
        annotation_genes=genes,
        reference_snps=len(snps),
        source_rows=source_rows,
        reference_qc=qc,
        common_usable_snps=len(common),
        traits=traits,
        policy="Intersect MAGMA-QC usable SNP IDs across all eight traits; retain original P/N rows",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("magma", "bfile", "annot", "pval-dir", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--traits",
        default=",".join(TRAITS),
        help="Exactly eight unique comma-separated trait identifiers",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1 or not str(args.bfile).endswith("22"):
        parser.error("Positive counts and a chromosome-22 bfile prefix are required")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(args.threads)
    from magma_py.io import parse_traits

    try:
        args.traits = parse_traits(args.traits)
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.traits) != 8:
        parser.error("Exactly eight unique traits are required")
    from validate_block_compatibility import compare

    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotation, inputs = prepare(args)
    result_path = args.out_dir / "results.json"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if result_path.exists():
        results = json.loads(result_path.read_text())
        if results["config"] != config or results["inputs"] != inputs:
            raise ValueError("Resume inputs or arguments differ from the saved benchmark")
    else:
        results = dict(
            config=config,
            inputs=inputs,
            host=platform.node(),
            platform=platform.platform(),
            python=sys.version,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            cache_policy="Read all command inputs before each timed fresh CLI process",
            records={},
        )
    save(result_path, results)
    for count in (1, 2, 4, 8):
        scenario, traits = f"shared_{count}", args.traits[:count]
        for repetition in range(1, args.repetitions + 1):
            record = results["records"].setdefault(scenario, {}).setdefault(str(repetition), {})
            order = ["magma", "magma_py"] if repetition % 2 else ["magma_py", "magma"]
            if count == 8:
                order.insert((repetition - 1) % 3, "fast_serial")
            for tool in order:
                if tool in record and all(
                    Path(p).exists() for p in record[tool]["outputs"].values()
                ):
                    continue
                print(f"Running {scenario} repetition {repetition}: {tool}", flush=True)
                record[tool] = run_tool(
                    args,
                    tool,
                    scenario,
                    traits,
                    args.out_dir / "inputs/shared",
                    annotation,
                    repetition,
                )
                save(result_path, results)
            record["comparisons"] = {}
            for tool in ("magma_py", "fast_serial"):
                if tool not in record:
                    continue
                record["comparisons"][tool] = {}
                for trait in traits:
                    _, agreement = compare(
                        record[tool]["outputs"][trait], record["magma"]["outputs"][trait]
                    )
                    record["comparisons"][tool][trait] = agreement
                    if (
                        agreement["only_fast"]
                        or agreement["only_official"]
                        or any(
                            agreement[key] != 1.0
                            for key in (
                                "nparam_match_fraction",
                                "nsnps_match_fraction",
                                "n_match_fraction",
                            )
                        )
                    ):
                        save(result_path, results)
                        raise RuntimeError(f"Unequal workload: {scenario}, {trait}, {tool}")
            results["summary"] = summarize(results["records"])
            save(result_path, results)
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
