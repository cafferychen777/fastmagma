"""Compare chromosome-22 block-model outputs with canonical MAGMA tables."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compare(fast_path, official_path):
    fast = pd.read_csv(fast_path, sep=r"\s+", dtype={"GENE": str})
    official = pd.read_csv(official_path, sep=r"\s+", comment="#", dtype={"GENE": str})
    if fast.GENE.duplicated().any() or official.GENE.duplicated().any():
        raise ValueError("Gene IDs must be unique in each comparison table")
    joined = fast.merge(official, on="GENE", suffixes=("_fast", "_official"), validate="one_to_one")
    if joined.empty:
        raise ValueError("Comparison tables have no matched genes")
    joined["delta_z"] = joined.ZSTAT_fast - joined.ZSTAT_official
    joined["abs_delta_z"] = np.abs(joined.delta_z)
    joined["nominal_crossing"] = (joined.P_fast < 0.05) != (joined.P_official < 0.05)
    summary = dict(
        fast_genes=len(fast),
        official_genes=len(official),
        matched_genes=len(joined),
        only_fast=sorted(set(fast.GENE) - set(official.GENE)),
        only_official=sorted(set(official.GENE) - set(fast.GENE)),
        max_abs_delta_z=float(joined.abs_delta_z.max()),
        median_abs_delta_z=float(joined.abs_delta_z.median()),
        p99_abs_delta_z=float(joined.abs_delta_z.quantile(0.99)),
        spearman_z=float(spearmanr(joined.ZSTAT_fast, joined.ZSTAT_official).statistic),
        top20_overlap=len(
            set(joined.nsmallest(20, "P_fast").GENE) & set(joined.nsmallest(20, "P_official").GENE)
        ),
        nparam_match_fraction=float((joined.NPARAM_fast == joined.NPARAM_official).mean()),
        nsnps_match_fraction=float((joined.NSNPS_fast == joined.NSNPS_official).mean()),
        n_match_fraction=float((joined.N_fast == joined.N_official).mean()),
        nominal_crossings=int(joined.nominal_crossing.sum()),
    )
    return joined.sort_values("abs_delta_z", ascending=False), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--traits", default="cad,ad")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for trait in args.traits.split(","):
        rows, summary = compare(
            args.fast_dir / f"{trait}.chr22.fastmagma.tsv",
            args.official_dir / trait / f"{trait}.batch22_chr.genes.out",
        )
        rows.to_csv(args.out_dir / f"{trait}.comparison.csv", index=False)
        (args.out_dir / f"{trait}.summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        summaries[trait] = summary
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
