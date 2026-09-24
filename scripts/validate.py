#!/usr/bin/env python
"""Compare supplied fastmagma outputs with official MAGMA gene tables.

This describes agreement; it does not establish calibration or declare a pass.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from fastmagma.io import parse_chromosomes, parse_traits, read_gene_table


def compare(fast, magma, trait, chrom, threshold):
    if magma.GENE.duplicated().any():
        raise ValueError("Official MAGMA output contains duplicate gene IDs")
    matched = fast.merge(magma, on="GENE", suffixes=("_fast", "_magma"), validate="one_to_one")
    finite = np.isfinite(matched.ZSTAT_fast) & np.isfinite(matched.ZSTAT_magma)
    bulk = matched[finite & (matched.P_magma > threshold)]
    delta = (bulk.ZSTAT_fast - bulk.ZSTAT_magma).abs()
    enough = len(bulk) > 1 and bulk.ZSTAT_fast.nunique() > 1 and bulk.ZSTAT_magma.nunique() > 1
    # Use each complete table for top lists; unmatched top genes count as disagreements.
    top_fast = set(fast.nlargest(20, "ZSTAT").GENE)
    top_magma = set(magma.nlargest(20, "ZSTAT").GENE)
    return dict(
        trait=trait,
        chrom=chrom,
        n_fast=len(fast),
        n_magma=len(magma),
        n_matched=len(matched),
        n_bulk=len(bulk),
        magma_p_threshold=threshold,
        pearson=float(pearsonr(bulk.ZSTAT_fast, bulk.ZSTAT_magma)[0]) if enough else np.nan,
        spearman=float(spearmanr(bulk.ZSTAT_fast, bulk.ZSTAT_magma)[0]) if enough else np.nan,
        median_abs_dz=float(delta.median()),
        max_abs_dz=float(delta.max()),
        nsnps_equal_pct=float((matched.NSNPS_fast == matched.NSNPS_magma).mean() * 100),
        nparam_equal_pct=float((matched.NPARAM_fast == matched.NPARAM_magma).mean() * 100),
        max_abs_n_difference=float((matched.N_fast - matched.N_magma).abs().max()),
        top20_overlap=len(top_fast & top_magma),
        top_fast_count=len(top_fast),
        top_magma_count=len(top_magma),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits", required=True)
    parser.add_argument("--chrs", required=True)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--magma-dir", type=Path, required=True)
    parser.add_argument("--min-magma-p", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, default=Path("validation_summary.csv"))
    args = parser.parse_args()
    if not 0 <= args.min_magma_p < 1:
        parser.error("--min-magma-p must be in [0, 1)")
    rows = []
    for trait in parse_traits(args.traits):
        for chrom in parse_chromosomes(args.chrs):
            fast = read_gene_table(args.fast_dir / f"{trait}.chr{chrom}.fastmagma.tsv")
            path = args.magma_dir / trait / f"{trait}.batch{chrom}_chr.genes.out"
            magma = pd.read_csv(path, sep=r"\s+", comment="#", dtype={"GENE": str})
            rows.append(compare(fast, magma, trait, chrom, args.min_magma_p))
    summary = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
