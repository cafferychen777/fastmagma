"""Validated input parsing and disk-backed summary-statistic arrays."""

from dataclasses import dataclass
from pathlib import Path
import logging
import re

import numpy as np
import pandas as pd
from scipy.special import ndtri_exp

LOG = logging.getLogger(__name__)

COLUMNS = [
    "GENE",
    "CHR",
    "START",
    "STOP",
    "NSNPS",
    "NPARAM",
    "N",
    "ZSTAT",
    "P",
    "LOG10P",
    "PMETHOD",
]


def parse_traits(value):
    traits = [x.strip() for x in value.split(",")]
    if not traits or len(set(traits)) != len(traits):
        raise ValueError("Traits must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) for x in traits):
        raise ValueError("Traits must be nonempty filename-safe identifiers")
    return traits


def parse_chromosomes(value):
    try:
        if "-" in value:
            lo, hi = map(int, value.split("-"))
            result = list(range(lo, hi + 1))
        else:
            result = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise ValueError("Chromosomes must be a range or comma-separated integers") from exc
    if not result or len(result) != len(set(result)) or any(c < 1 or c > 22 for c in result):
        raise ValueError("Chromosomes must be unique autosomes from 1 to 22")
    return sorted(result)


@dataclass(frozen=True)
class Gene:
    identifier: str
    chrom: int
    start: int
    stop: int
    snps: tuple[str, ...]


def load_annotation(path, chrom):
    genes, seen = [], set()
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            try:
                gene, location, *snps = fields
                c, start, stop = map(int, location.split(":"))
                if not gene or start < 0 or stop <= start or any(not s for s in snps):
                    raise ValueError()
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Malformed annotation at {path}:{number}") from exc
            if c != chrom:
                continue
            if gene in seen:
                raise ValueError(f"Duplicate gene ID {gene!r} on chromosome {chrom}")
            seen.add(gene)
            genes.append(Gene(gene, c, start, stop, tuple(dict.fromkeys(snps))))
    if not genes:
        raise ValueError(f"No genes annotated on chromosome {chrom}")
    return genes


class TraitStore:
    """Three float64 arrays per trait: P, N, and precomputed chi-square values.

    Input parsing is chunked. SNP identity is indexed once across all traits.
    ``whole`` excludes duplicate rows before QC. ``magma`` follows MAGMA 1.10
    input order, P truncation, and integer sample-size rules.
    """

    def __init__(self, directory, traits, snps, model="whole"):
        if model not in {"whole", "magma"}:
            raise ValueError("Model must be magma or whole")
        self.model = model
        self.traits = traits
        self.index = pd.Index(snps)
        if not len(self.index) or not self.index.is_unique:
            raise ValueError("Reference SNP identifiers must be nonempty and unique")
        self.path = Path(directory) / "traits.f64"
        self.values = np.memmap(
            self.path, mode="w+", dtype="float64", shape=(len(traits), 3, len(snps))
        )
        self.qc = {}

    def load(self, pval_dir, chunk_rows):
        for t, trait in enumerate(self.traits):
            values = self.values[t]
            values[:] = np.nan
            seen = np.zeros(len(self.index), dtype=np.uint8)
            rows, matched, invalid = 0, 0, 0
            path = Path(pval_dir) / f"{trait}.pval"
            chunks = pd.read_csv(
                path,
                sep=r"\s+",
                usecols=["SNP", "P", "N"],
                dtype=str,
                keep_default_na=False,
                chunksize=chunk_rows,
            )
            with chunks:
                for chunk in chunks:
                    rows += len(chunk)
                    position = self.index.get_indexer(chunk["SNP"])
                    present = position >= 0
                    position = position[present]
                    matched += len(position)
                    if not len(position):
                        continue
                    unique, first, counts = np.unique(
                        position, return_index=True, return_counts=True
                    )
                    selected = chunk.loc[present, ["P", "N"]]
                    if self.model == "magma":
                        # States: unread, accepted, duplicated, excluded. An excluded
                        # ID stays excluded; a repeat of an accepted ID is dropped
                        # before parsing that repeat's P or N.
                        unread = seen[unique] == 0
                        repeated = unique[seen[unique] == 1]
                        seen[repeated] = 2
                        ids = unique[unread]
                        selected = selected.iloc[first[unread]]
                        p, n, good = self._parse_rows(selected, path)
                        invalid += int((~good).sum())
                        seen[ids] = np.where(good, 1, 3)
                        seen[ids[good & (counts[unread] > 1)]] = 2
                    else:
                        p, n, good = self._parse_rows(selected, path)
                        invalid += int((~good).sum())
                        seen[unique] = np.minimum(2, seen[unique].astype(np.int64) + counts)
                        ids = position
                    good &= seen[ids] == 1
                    values[0, ids[good]], values[1, ids[good]] = p[good], n[good]
                    values[:, unique[seen[unique] == 2]] = np.nan
            usable = np.isfinite(values[0])
            ids = np.flatnonzero(usable)
            # Inverse normal in log space preserves subnormal positive input P.
            for start in range(0, len(ids), chunk_rows):
                selected = ids[start : start + chunk_rows]
                z = -ndtri_exp(np.log(values[0, selected]) - np.log(2.0))
                values[2, selected] = z * z
            self.qc[trait] = dict(
                rows=rows,
                matched=matched,
                invalid_rows=invalid,
                duplicate_snps=int((seen == 2).sum()),
                usable=int(usable.sum()),
            )
            LOG.info(
                "Loaded %s: %s usable SNPs, %s duplicate IDs",
                trait,
                int(usable.sum()),
                int((seen == 2).sum()),
            )
            if not usable.any():
                raise ValueError(f"No usable reference SNPs for trait {trait}")
        self.values.flush()

    def _parse_rows(self, rows, path):
        """Parse only rows that reach validation under the selected input model."""
        p = pd.to_numeric(rows["P"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(rows["N"], errors="coerce").to_numpy(dtype=float)
        if self.model == "whole":
            good = np.isfinite(p) & np.isfinite(n) & (p > 0) & (p <= 1) & (n >= 50)
            return p, n, good

        missing_p = rows["P"].isin(["NA", "-1"]).to_numpy()
        invalid_p = ~missing_p & (~np.isfinite(p) | (p < 0) | (p > 1))
        if invalid_p.any():
            raise ValueError(f"Invalid SNP P in {path}: {rows.loc[invalid_p, 'P'].iloc[0]!r}")
        # C++ round rounds halfway cases away from zero. Sample sizes are positive.
        n = np.copysign(np.floor(np.abs(n) + 0.5), n)
        missing_n = rows["N"].eq("NA").to_numpy()
        invalid_n = (
            ~missing_p & ~missing_n & (~np.isfinite(n) | (n < 0) | (n > np.iinfo(np.int32).max))
        )
        if invalid_n.any():
            raise ValueError(f"Invalid SNP N in {path}: {rows.loc[invalid_n, 'N'].iloc[0]!r}")
        # MAGMA 1.10 applies a strict lower bound after rounding, so N=50 is dropped.
        good = ~missing_p & ~missing_n & (n > 50)
        return np.clip(p, 1e-50, 1 - 1e-5), n, good

    def close(self):
        self.values.flush()
        self.values._mmap.close()


def read_gene_table(path):
    df = pd.read_csv(path, sep=r"\s+", dtype={"GENE": str}, keep_default_na=False)
    if not set(COLUMNS).issubset(df):
        raise ValueError(f"Missing output columns in {path}")
    if df.GENE.duplicated().any() or (df.GENE == "").any():
        raise ValueError(f"Duplicate or empty gene ID in {path}")
    for name in ["CHR", "START", "STOP", "NSNPS", "NPARAM", "N", "ZSTAT", "P", "LOG10P"]:
        df[name] = pd.to_numeric(df[name], errors="raise")
    # Z=-inf is valid at P=1; all other nonfinite values are invalid.
    for name in ["CHR", "START", "STOP", "NSNPS", "NPARAM", "N", "P", "LOG10P"]:
        if not np.isfinite(df[name]).all():
            raise ValueError(f"Nonfinite {name} in {path}")
    valid_z = np.isfinite(df.ZSTAT) | (np.isneginf(df.ZSTAT) & (df.LOG10P == 0))
    if not valid_z.all() or (df.LOG10P > 0).any() or ((df.P < 0) | (df.P > 1)).any():
        raise ValueError(f"Invalid gene probabilities in {path}")
    for name in ["CHR", "START", "STOP", "NSNPS", "NPARAM"]:
        if (df[name] != np.floor(df[name])).any():
            raise ValueError(f"Noninteger {name} in {path}")
    if (
        (df.CHR < 1)
        | (df.CHR > 22)
        | (df.START < 0)
        | (df.STOP < df.START)
        | (df.NSNPS < 1)
        | (df.NPARAM < 1)
        | (df.NPARAM > df.NSNPS)
        | (df.N < 50)
    ).any():
        raise ValueError(f"Invalid gene metadata in {path}")
    expected_p = np.exp(df.LOG10P.to_numpy() * np.log(10.0))
    expected_z = -ndtri_exp(df.LOG10P.to_numpy() * np.log(10.0))
    if not np.allclose(df.P, expected_p, rtol=1e-9, atol=np.nextafter(0.0, 1.0)) or not np.allclose(
        df.ZSTAT, expected_z, rtol=1e-9, atol=1e-10
    ):
        raise ValueError(f"Inconsistent P, LOG10P, or ZSTAT in {path}")
    return df[COLUMNS]
