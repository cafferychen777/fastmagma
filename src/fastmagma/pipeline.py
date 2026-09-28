"""Chromosome orchestration, resource limits, and validated output merging."""

from collections import Counter
import csv
import hashlib
from contextlib import ExitStack, closing, contextmanager
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

from bed_reader import open_bed
import numpy as np
from threadpoolctl import threadpool_limits

from .genotypes import GenotypeReader, workspace_bytes
from .io import (
    COLUMNS,
    TraitStore,
    iter_annotation,
    load_reference_index,
    parse_chromosomes,
    parse_traits,
    read_gene_table,
)

LOG = logging.getLogger(__name__)


@contextmanager
def locked_outputs(directory, chromosomes, merging=False):
    """Lock affected chromosomes; independent chromosome jobs can run together."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    names = [".fastmagma.merge.lock"] if merging else []
    names += [f".fastmagma.chr{c}.lock" for c in sorted(chromosomes)]
    acquired = []
    try:
        for name in names:
            lock = out / name
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"Output is locked: {lock}; check for an active process"
                ) from exc
            acquired.append(lock)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        for lock in reversed(acquired):
            lock.unlink(missing_ok=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_outputs(paths, overwrite):
    if not overwrite:
        existing = [str(p) for p in paths if p.exists()]
        if existing:
            raise FileExistsError(f"Output exists; use --overwrite: {existing[0]}")


def _publish(staging, out_dir, names, manifest_name, metadata):
    """Publish complete individual files; the manifest is the final commit marker."""
    metadata["outputs"] = {name: {"sha256": _sha256(staging / name)} for name in names}
    marker = out_dir / manifest_name
    marker.unlink(missing_ok=True)
    for name in names:
        os.replace(staging / name, out_dir / name)
    (staging / manifest_name).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    os.replace(staging / manifest_name, marker)


def _input_record(path):
    p = Path(path).resolve()
    st = p.stat()
    return dict(path=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def run_chromosome(args):
    started = time.perf_counter()
    model = args.model
    traits = parse_traits(args.traits)
    chrom = parse_chromosomes(str(args.chr))[0]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [f"{t}.chr{chrom}.fastmagma.tsv" for t in traits]
    manifest_name = f"chr{chrom}.manifest.json"
    _check_outputs([out / name for name in [*names, manifest_name]], args.overwrite)
    bfile = f"{args.bfile_prefix}{chrom}"
    input_paths = [f"{bfile}.{ext}" for ext in ("bed", "bim", "fam")]
    input_paths += [args.annot] + [str(Path(args.pval_dir) / f"{t}.pval") for t in traits]
    records = [_input_record(p) for p in input_paths]
    snps = load_reference_index(f"{bfile}.bim", chrom)
    # The shared lookup owns SNP metadata; skip all PLINK property arrays.
    bed = open_bed(
        f"{bfile}.bed",
        count_A1=False,
        sid_count=len(snps),
        num_threads=args.threads,
        properties=dict.fromkeys(
            (
                "fid",
                "iid",
                "father",
                "mother",
                "sex",
                "pheno",
                "sid",
                "chromosome",
                "cm_position",
                "bp_position",
                "allele_1",
                "allele_2",
            )
        ),
    )
    minimum_samples = 50 if model == "magma" else 2
    if bed.iid_count < minimum_samples:
        raise ValueError(
            f"The {model} model requires at least {minimum_samples} reference individuals"
        )
    annotated_reference = False
    reader = GenotypeReader(bed, args.cache_mb, args.block_snps, args.threads)
    counts = {t: 0 for t in traits}
    methods = {t: Counter() for t in traits}
    max_estimate = 0
    eigen_count = 0
    with tempfile.TemporaryDirectory(prefix="fastmagma-", dir=args.temp_dir) as scratch:
        # One reference-order index serves both input validation and genotype reads.
        store = TraitStore(scratch, traits, snps, model=model)
        try:
            store.load(args.pval_dir, args.chunk_rows)
            # Initialize numerical integration after input buffers are released.
            from .blocks import prepare_gene

            available = np.zeros(len(snps), dtype=bool)
            for t in range(len(traits)):
                available |= np.isfinite(store.values[t, 0])
            with tempfile.TemporaryDirectory(prefix=".fastmagma-", dir=out) as stage:
                staging = Path(stage)
                with ExitStack() as stack, threadpool_limits(limits=args.threads):
                    writers = []
                    for name in names:
                        fh = stack.enter_context(open(staging / name, "w", newline=""))
                        writer = csv.writer(fh, delimiter="\t")
                        writer.writerow(COLUMNS)
                        writers.append(writer)
                    genes = stack.enter_context(closing(iter_annotation(args.annot, chrom)))
                    for gene_number, gene in enumerate(genes, 1):
                        if gene_number % 100 == 0:
                            LOG.info("Chromosome %s: gene %s", chrom, gene_number)
                        ids = snps.get_indexer(gene.snps)
                        ids = np.unique(ids[ids >= 0])
                        annotated_reference |= bool(len(ids))
                        # Avoid genotype work for SNPs absent in all traits.
                        ids = ids[available[ids]]
                        if not len(ids):
                            continue
                        if len(ids) > args.max_gene_snps:
                            raise MemoryError(
                                f"Gene {gene.identifier}: {len(ids)} SNPs exceeds --max-gene-snps"
                            )
                        estimate = workspace_bytes(bed.iid_count, len(ids), args.block_snps)
                        estimate += 2 * len(traits) * len(ids)
                        max_estimate = max(max_estimate, estimate)
                        if estimate > args.workspace_mb * 2**20:
                            raise MemoryError(
                                f"Gene {gene.identifier}: estimated numerical workspace "
                                f"{estimate / 2**20:.1f} MiB exceeds --workspace-mb"
                            )
                        g, keep, missing = reader.read_with_missing(ids, model=model)
                        positions = ids[keep]
                        if not len(positions):
                            continue
                        groups = {}
                        for t in range(len(traits)):
                            mask = np.isfinite(store.values[t, 0, positions])
                            if mask.any():
                                groups.setdefault(mask.tobytes(), (mask, []))[1].append(t)
                        for mask, group in groups.values():
                            selected = positions[mask]
                            prepared = prepare_gene(
                                g if mask.all() else g[:, mask],
                                model=model,
                                missing=None if missing is None else missing[:, mask],
                            )
                            eigen_count += prepared.eigendecompositions
                            for t in group:
                                p, n, q = store.values[t][:, selected]
                                result = prepared.test(q, p)
                                mean_n = (
                                    math.floor(float(n.mean()) + 0.5)
                                    if model == "magma"
                                    else float((n / n.max()).mean() * n.max())
                                )
                                writers[t].writerow(
                                    [
                                        gene.identifier,
                                        chrom,
                                        gene.start,
                                        gene.stop,
                                        len(p),
                                        prepared.nparam,
                                        mean_n,
                                        result.zstat,
                                        result.p,
                                        result.log10p,
                                        result.method,
                                    ]
                                )
                                counts[traits[t]] += 1
                                methods[traits[t]][result.method] += 1
                if not annotated_reference:
                    raise ValueError("No annotated SNPs occur in the reference panel")
                empty = [t for t, count in counts.items() if not count]
                if empty:
                    raise ValueError(f"No polymorphic gene results for traits: {','.join(empty)}")
                if records != [_input_record(p) for p in input_paths]:
                    raise RuntimeError(
                        "Input files changed during computation; outputs were not published"
                    )
                metadata = dict(
                    schema=1,
                    version=version("fastmagma"),
                    chromosome=chrom,
                    model=model,
                    model_revision=2 if model == "magma" else 1,
                    traits=traits,
                    genes=counts,
                    methods=methods,
                    qc=store.qc,
                    inputs=records,
                    seconds=time.perf_counter() - started,
                    resources=dict(
                        threads=args.threads,
                        cache_mb=args.cache_mb,
                        workspace_mb=args.workspace_mb,
                        chunk_rows=args.chunk_rows,
                        block_snps=args.block_snps,
                        max_gene_snps=args.max_gene_snps,
                        estimated_peak_workspace_bytes=max_estimate,
                        cache_bytes=reader.cache_bytes,
                        block_reads=reader.block_reads,
                        eigendecompositions=eigen_count,
                        trait_store_bytes=store.values.nbytes,
                    ),
                )
                _publish(staging, out, names, manifest_name, metadata)
        finally:
            store.close()
    LOG.info("Chromosome %s complete: %s; %.2fs", chrom, counts, time.perf_counter() - started)


def merge_chromosomes(args):
    import pandas as pd

    traits = parse_traits(args.traits)
    chromosomes = parse_chromosomes(args.chrs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [f"{t}.genes.out" for t in traits] + ["zstat_matrix.tsv"]
    marker = "merge.manifest.json"
    _check_outputs([out / name for name in [*names, marker]], args.overwrite)
    missing = [
        f"{t}.chr{c}.fastmagma.tsv"
        for t in traits
        for c in chromosomes
        if not (out / f"{t}.chr{c}.fastmagma.tsv").is_file()
    ]
    if missing and not args.allow_missing:
        raise FileNotFoundError(f"Missing chromosome outputs: {', '.join(missing)}")
    if missing:
        LOG.warning("Explicitly allowing missing chromosome files: %s", missing)
    zmat = {}
    source_models = set()
    with tempfile.TemporaryDirectory(prefix=".fastmagma-merge-", dir=out) as stage:
        staging = Path(stage)
        for trait in traits:
            parts = []
            for chrom in chromosomes:
                path = out / f"{trait}.chr{chrom}.fastmagma.tsv"
                if not path.exists():
                    continue
                source_marker = out / f"chr{chrom}.manifest.json"
                if not source_marker.is_file():
                    raise ValueError(f"Missing completion manifest: {source_marker}")
                record = json.loads(source_marker.read_text())
                # Outputs created before model selection used the whole-gene test.
                source_model = record.get("model", "whole")
                if source_model not in {"magma", "whole"}:
                    raise ValueError(
                        f"Unknown statistical model in {source_marker}: {source_model}"
                    )
                source_models.add((source_model, record.get("model_revision", 1)))
                if len(source_models) > 1:
                    raise ValueError(
                        "Cannot merge chromosome outputs from different statistical models or revisions"
                    )
                entry = record.get("outputs", {}).get(path.name)
                if not entry or entry.get("sha256") != _sha256(path):
                    raise ValueError(f"Output does not match its completion manifest: {path}")
                df = read_gene_table(path)
                if df.empty or not (df.CHR == chrom).all():
                    raise ValueError(f"Empty output or incorrect chromosome in {path}")
                parts.append(df)
            if not parts:
                raise ValueError(f"No chromosome outputs for {trait}")
            df = pd.concat(parts, ignore_index=True).sort_values(["CHR", "START", "GENE"])
            if df.GENE.duplicated().any():
                raise ValueError(f"Gene IDs occur on multiple chromosomes for {trait}")
            df.to_csv(staging / f"{trait}.genes.out", sep="\t", index=False)
            zmat[trait] = df.set_index("GENE").ZSTAT
        pd.DataFrame(zmat).to_csv(staging / "zstat_matrix.tsv", sep="\t", na_rep="NA")
        model, model_revision = next(iter(source_models))
        _publish(
            staging,
            out,
            names,
            marker,
            dict(
                schema=1,
                version=version("fastmagma"),
                traits=traits,
                chromosomes=chromosomes,
                model=model,
                model_revision=model_revision,
                missing=missing,
            ),
        )
    LOG.info("Merged %s chromosomes for %s traits", len(chromosomes), len(traits))
