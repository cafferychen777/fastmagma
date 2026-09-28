#!/usr/bin/env python
"""Join merged gene outputs to a user-supplied mapping.

The map must contain unique string gene_loc_id values plus symbol, chr, start,
and stop. Symbols need not be unique. No gene mapping is bundled.
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--genes-dir", required=True, help="dir with <trait>.genes.out (magma-py merge output)"
    )
    ap.add_argument(
        "--traits", required=True, help="comma-separated trait list, in desired column order"
    )
    ap.add_argument(
        "--map",
        required=True,
        help="gene_id_symbol_map.tsv (gene_loc_id symbol chr start stop)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from magma_py.io import parse_traits, read_gene_table

    traits = parse_traits(args.traits)
    gmap = pd.read_csv(args.map, sep="\t", dtype={"gene_loc_id": str}, keep_default_na=False)
    if gmap["gene_loc_id"].duplicated().any():
        raise ValueError("Gene mapping IDs must be unique")
    base = gmap[["gene_loc_id", "symbol", "chr", "start", "stop"]].copy()

    zcols = {}
    nsnps_cols = {}
    for tr in traits:
        fp = os.path.join(args.genes_dir, f"{tr}.genes.out")
        df = read_gene_table(fp)
        zcols[tr] = df.set_index("GENE")["ZSTAT"]
        nsnps_cols[tr] = df.set_index("GENE")["NSNPS"]
        print(f"[build] {tr}: {len(df)} genes")

    out = base.set_index("gene_loc_id")
    nsnps_df = pd.DataFrame(index=out.index)
    for tr in traits:
        out[tr] = zcols[tr].reindex(out.index)
        nsnps_df[tr] = nsnps_cols[tr].reindex(out.index)

    # Minimum SNP count across traits with a result for this gene.
    out["nsnps_min"] = nsnps_df.min(axis=1, skipna=True)

    # Drop genes with no result in any requested trait.
    has_any = out[traits].notna().any(axis=1)
    out = out[has_any]

    ordered_cols = ["symbol", "chr", "start", "stop", "nsnps_min"] + traits
    out = out[ordered_cols]
    out.index.name = "gene_loc_id"
    out = out.reset_index().sort_values(["chr", "start"])

    out.to_csv(args.out, sep="\t", index=False, na_rep="NA", float_format="%.6g")
    print(f"[build] wrote {args.out}: {out.shape[0]} genes x {len(traits)} traits")


if __name__ == "__main__":
    main()
