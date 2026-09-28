"""Measure paired MAGMA/MAGMA-py runtime on one serial compute allocation.

Input preparation and explicit filesystem-cache warming are outside timing.
Each fresh CLI process includes imports, input parsing, analysis, and output.
Official multi-trait runs are serial; their wall/CPU times are summed and their
peak RSS is the maximum, not the sum. Shared-input MAGMA-py also runs serially
as a within-program control for the benefit of batching and LD reuse.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def filter_pvalues(source, destination, snps):
    """Keep original rows and order, including duplicates and missing values."""
    present, rows = set(), 0
    with source.open() as reader, destination.open("w") as writer:
        header = next(reader)
        column = header.split().index("SNP")
        writer.write(header)
        for line in reader:
            fields = line.split()
            if fields and fields[column] in snps:
                writer.write(line)
                present.add(fields[column])
                rows += 1
    return present, rows


def prepare(args):
    directory = args.out_dir / "inputs"
    different, shared = directory / "different", directory / "shared"
    different.mkdir(parents=True, exist_ok=True)
    shared.mkdir(exist_ok=True)
    snps = {line.split()[1] for line in Path(str(args.bfile) + ".bim").read_text().splitlines()}
    annotation = directory / "chr22.genes.annot"
    annotation_genes = 0
    with args.annot.open() as reader, annotation.open("w") as writer:
        for line in reader:
            fields = line.split()
            if line.startswith("#"):
                writer.write(line)
            elif len(fields) >= 2 and fields[1].split(":")[0] == "22":
                writer.write(line)
                annotation_genes += 1
    if not annotation_genes:
        raise ValueError("No chromosome-22 genes in annotation")
    selected, counts = {}, {}
    for trait in ("cad", "ad"):
        selected[trait], counts[trait] = filter_pvalues(
            args.pval_dir / f"{trait}.pval", different / f"{trait}.pval", snps
        )
    common = selected["cad"] & selected["ad"]
    for trait in selected:
        filter_pvalues(different / f"{trait}.pval", shared / f"{trait}.pval", common)
    return annotation, dict(
        annotation_genes=annotation_genes,
        reference_snps=len(snps),
        different_rows=counts,
        different_unique_snps={t: len(s) for t, s in selected.items()},
        shared_unique_snps=len(common),
    )


def warm(paths):
    for path in dict.fromkeys(paths):
        with path.open("rb") as reader:
            while reader.read(8 * 1024 * 1024):
                pass


def measure_child(command, timing, console):
    """Measure one child from a fresh stdlib-only Python process (Linux KiB RSS)."""
    with console.open("w") as log:
        started = time.perf_counter()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(child.pid, 0)
        elapsed = time.perf_counter() - started
        child.returncode = os.waitstatus_to_exitcode(status)
    save(
        timing,
        dict(
            wall_seconds=elapsed,
            user_seconds=usage.ru_utime,
            system_seconds=usage.ru_stime,
            max_rss_kib=usage.ru_maxrss / 1024 if sys.platform == "darwin" else usage.ru_maxrss,
            returncode=child.returncode,
        ),
    )
    return child.returncode


def timed(command, prefix, inputs):
    warm(inputs)
    timing = prefix.with_suffix(".time.json")
    command = list(map(str, command))
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--measure",
            str(timing),
            str(prefix.with_suffix(".console.log")),
            json.dumps(command),
        ]
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {prefix}.console.log")
    measurement = json.loads(timing.read_text())
    measurement["command"] = command
    return measurement


def run_tool(args, tool, scenario, traits, pval_dir, annotation, repetition):
    directory = args.out_dir / scenario / f"repeat{repetition}" / tool
    directory.mkdir(parents=True, exist_ok=True)
    common_inputs = [Path(str(args.bfile) + suffix) for suffix in (".bed", ".bim", ".fam")]
    common_inputs.append(annotation)
    measurements, outputs, resources = [], {}, []
    groups = [traits] if tool == "magma_py" else [[trait] for trait in traits]
    for index, group in enumerate(groups):
        output = directory / f"part{index}"
        inputs = common_inputs + [pval_dir / f"{trait}.pval" for trait in group]
        if tool == "magma":
            command = [
                args.magma,
                "--bfile",
                args.bfile,
                "--gene-annot",
                annotation,
                "--pval",
                inputs[-1],
                "use=SNP,P",
                "ncol=N",
                "--genes-only",
                "--out",
                output,
            ]
            outputs[group[0]] = str(output) + ".genes.out"
        else:
            command = [
                sys.executable,
                "-m",
                "magma_py",
                "run",
                "--chr",
                "22",
                "--bfile-prefix",
                str(args.bfile)[:-2],
                "--annot",
                annotation,
                "--pval-dir",
                pval_dir,
                "--traits",
                ",".join(group),
                "--out-dir",
                output,
                "--threads",
                str(args.threads),
                "--overwrite",
            ]
            outputs.update({trait: str(output / f"{trait}.chr22.magma_py.tsv") for trait in group})
        measurements.append(timed(command, directory / f"process{index}", inputs))
        if tool != "magma":
            resources.append(json.loads((output / "chr22.manifest.json").read_text())["resources"])
    return dict(
        processes=measurements,
        outputs=outputs,
        resources=resources,
        wall_seconds=sum(m["wall_seconds"] for m in measurements),
        cpu_seconds=sum(m["user_seconds"] + m["system_seconds"] for m in measurements),
        max_rss_kib=max(m["max_rss_kib"] for m in measurements),
        eigendecompositions=sum(r["eigendecompositions"] for r in resources) if resources else None,
    )


def summarize(records):
    summaries = {}
    for scenario, repetitions in records.items():
        tools = set.intersection(*(set(rep) for rep in repetitions.values()))
        summary = {}
        for tool in sorted(tools):
            if tool == "comparisons":
                continue
            values = [rep[tool] for rep in repetitions.values()]
            summary[tool] = {
                metric: dict(
                    median=statistics.median(v[metric] for v in values),
                    min=min(v[metric] for v in values),
                    max=max(v[metric] for v in values),
                )
                for metric in ("wall_seconds", "cpu_seconds", "max_rss_kib")
            }
        if "magma" in summary and "magma_py" in summary:
            summary["speedup_median_times"] = (
                summary["magma"]["wall_seconds"]["median"]
                / summary["magma_py"]["wall_seconds"]["median"]
            )
        if "fast_serial" in summary and "magma_py" in summary:
            summary["batching_speedup_median_times"] = (
                summary["fast_serial"]["wall_seconds"]["median"]
                / summary["magma_py"]["wall_seconds"]["median"]
            )
        summaries[scenario] = summary
    return summaries


def main():
    from validate_block_compatibility import compare

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("magma", "bfile", "annot", "pval-dir", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1 or not str(args.bfile).endswith("22"):
        parser.error(
            "Positive thread/repetition counts and a chromosome-22 bfile prefix are required"
        )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotation, inputs = prepare(args)
    result_path = args.out_dir / "results.json"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if result_path.exists():
        results = json.loads(result_path.read_text())
        if results["config"] != config:
            raise ValueError("Resume arguments differ from the saved benchmark")
    else:
        results = dict(
            config=config,
            host=platform.node(),
            platform=platform.platform(),
            python=sys.version,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            inputs=inputs,
            cache_policy="Read all command inputs before each timed process; fresh processes, no untimed analysis warmup",
            records={},
        )
    scenarios = [
        ("single_cad", ["cad"], args.out_dir / "inputs/different"),
        ("single_ad", ["ad"], args.out_dir / "inputs/different"),
        ("pair_shared", ["cad", "ad"], args.out_dir / "inputs/shared"),
        ("pair_different", ["cad", "ad"], args.out_dir / "inputs/different"),
        ("pair_full_gwas", ["cad", "ad"], args.pval_dir),
    ]
    save(result_path, results)
    for scenario, traits, pval_dir in scenarios:
        for repetition in range(1, args.repetitions + 1):
            record = results["records"].setdefault(scenario, {}).setdefault(str(repetition), {})
            order = ["magma", "magma_py"] if repetition % 2 else ["magma_py", "magma"]
            if scenario == "pair_shared":
                order.insert((repetition - 1) % 3, "fast_serial")
            for tool in order:
                if tool in record and all(
                    Path(p).exists() for p in record[tool]["outputs"].values()
                ):
                    continue
                print(f"Running {scenario} repetition {repetition}: {tool}", flush=True)
                record[tool] = run_tool(
                    args, tool, scenario, traits, pval_dir, annotation, repetition
                )
                save(result_path, results)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record[tool].items()
                            if k not in ("processes", "outputs", "resources")
                        }
                    ),
                    flush=True,
                )
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
                            agreement[k] != 1.0
                            for k in (
                                "nparam_match_fraction",
                                "nsnps_match_fraction",
                                "n_match_fraction",
                            )
                        )
                    ):
                        save(result_path, results)
                        raise RuntimeError(
                            f"Unequal workload detected: {scenario}, {trait}, {tool}"
                        )
            results["summary"] = summarize(results["records"])
            save(result_path, results)
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--measure":
        sys.exit(measure_child(json.loads(sys.argv[4]), Path(sys.argv[2]), Path(sys.argv[3])))
    main()
