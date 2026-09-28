#!/usr/bin/env python
"""Describe LDL target ranks; this is not a calibration or protein-coding test.

Usage: python scripts/ldl_rank_check.py --genes-out ldl.genes.out --map genes.tsv
"""

import argparse

import pandas as pd
from magma_py.io import read_gene_table

TARGET_SYMBOLS = ["LDLR", "APOB", "PCSK9", "HMGCR", "APOE", "SORT1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes-out", required=True, help="MAGMA-py merged ldl.genes.out")
    ap.add_argument("--map", required=True, help="gene_id_symbol_map.tsv")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    df = read_gene_table(args.genes_out)
    gmap = pd.read_csv(args.map, sep="\t", dtype={"gene_loc_id": str}, keep_default_na=False)[
        ["gene_loc_id", "symbol"]
    ]
    df = df.merge(gmap, left_on="GENE", right_on="gene_loc_id", how="left", validate="one_to_one")

    df_all = df.sort_values("ZSTAT", ascending=False).reset_index(drop=True)
    df_all["rank_all"] = df_all.index + 1

    df_filt = df[df["NSNPS"] >= 3].sort_values("ZSTAT", ascending=False).reset_index(drop=True)
    df_filt["rank_filtered"] = df_filt.index + 1

    print(f"Total genes: {len(df)}, with NSNPS>=3: {len(df_filt)}\n")

    print(
        f"{'symbol':<8} {'GENE':>7} {'NSNPS':>6} {'ZSTAT':>9} {'LOG10P':>10} "
        f"{'method':<12} {'rank(all)':>10} {'rank(NSNPS>=3)':>15}"
    )
    for sym in TARGET_SYMBOLS:
        row = df_all[df_all["symbol"] == sym]
        if row.empty:
            print(f"{sym:<8} NOT FOUND")
            continue
        r = row.iloc[0]
        rf = df_filt[df_filt["symbol"] == sym]
        rf_rank = int(rf.iloc[0]["rank_filtered"]) if len(rf) else "excluded(NSNPS<3)"
        print(
            f"{sym:<8} {r['GENE']:>7} {int(r['NSNPS']):>6} {r['ZSTAT']:>9.3f} "
            f"{r['LOG10P']:>10.2f} {r['PMETHOD']:<12} {int(r['rank_all']):>10} {str(rf_rank):>15}"
        )

    print(f"\nTop {args.top} genes by ZSTAT, NSNPS>=3 filter applied:")
    cols = ["rank_filtered", "GENE", "symbol", "NSNPS", "ZSTAT", "LOG10P", "PMETHOD"]
    print(df_filt[cols].head(args.top).to_string(index=False))

    targets_in_top = [s for s in TARGET_SYMBOLS[:5] if s in set(df_filt.head(args.top)["symbol"])]
    print(
        f"Of {TARGET_SYMBOLS[:5]}, in filtered top {args.top}: {targets_in_top} "
        f"({len(targets_in_top)}/5)"
    )


if __name__ == "__main__":
    main()
