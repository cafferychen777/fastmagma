"""Check selected gene tails by independent high-precision Laplace inversion.

The JSON case list supplies trait, gene, annotation, pval, fast, and official
for each case. Annotation/P-value paths identify the exact analysis inputs;
fast/official paths identify their output tables. All cases use --bfile.
Requires the optional test dependency mpmath. Run on a compute allocation.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path

from bed_reader import open_bed
import mpmath as mp
import numpy as np
import pandas as pd
from scipy import special
from threadpoolctl import threadpool_limits

from validate_boundaries import sphere_probability


@lru_cache(maxsize=None)
def table(path, key):
    return pd.read_csv(path, sep=r"\s+", comment="#", dtype={key: str}).set_index(key)


@lru_cache(maxsize=None)
def annotation(path):
    return {
        fields[0]: fields[2:]
        for line in Path(path).read_text().splitlines()
        if not line.startswith("#") and (fields := line.split())
    }


def check_case(case, bed, lookup):
    gene = str(case["gene"])
    pvals = table(case["pval"], "SNP")
    if not pvals.index.is_unique:
        raise ValueError("The independent oracle requires already deduplicated SNP inputs")
    snps = sorted(
        set(annotation(case["annotation"])[gene]) & set(lookup) & set(pvals.index),
        key=lookup.get,
    )
    fast = table(case["fast"], "GENE").loc[gene]
    official = table(case["official"], "GENE").loc[gene]
    raw = bed.read(index=np.s_[:, [lookup[s] for s in snps]], dtype="float64")
    observed = np.isfinite(raw)
    centered = np.where(observed, raw - np.nanmean(raw, axis=0), 0)
    norms = np.linalg.norm(centered, axis=0)
    # Fail on unexpected QC instead of silently evaluating a different model.
    if not ((observed.sum(axis=0) >= 0.75 * len(raw)) & (norms > 0)).all():
        raise ValueError(f"{gene}: genotype QC needs separate investigation")
    if len(snps) != int(fast.NSNPS) or len(snps) != int(official.NSNPS):
        raise ValueError(f"{gene}: SNP counts differ")
    if len(snps) > min(len(raw) // 2, 1000):
        raise ValueError("This oracle checks unblocked genes only")
    g = centered / norms
    counts = observed.sum(axis=0)
    joint = observed.astype(float).T @ observed.astype(float)
    if np.any(joint == 0):
        raise ValueError("Every SNP pair must have joint observations")
    # Independent reconstruction of the summary-statistic LD model. Clip each
    # correlation as MAGMA does; do not project the matrix onto the PSD cone.
    corr = np.clip((g.T @ g) * np.sqrt(counts[:, None] * counts[None, :]) / joint, -1, 1)
    raw_eigenvalues = np.linalg.eigvalsh(corr)
    cutoff = 1e-4 * raw_eigenvalues[raw_eigenvalues > 0].sum() / len(snps)
    lam = raw_eigenvalues[raw_eigenvalues > cutoff]
    p = pvals.loc[snps, "P"].to_numpy(float)
    n = pvals.loc[snps, "N"].to_numpy(float)
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Oracle inputs require finite P values in [0, 1]")
    if not (np.isfinite(n).all() and (np.floor(n + 0.5) > 50).all()):
        raise ValueError("Oracle inputs require sample sizes passing MAGMA QC")
    p = np.clip(p, 1e-50, 1 - 1e-5)
    q = float(np.square(special.ndtri(p / 2)).sum())
    values = {}
    for precision in (50, 80):
        with mp.workdps(precision):
            weights = [mp.mpf(float(x)) for x in lam]

            def transform(s):
                # Laplace transform of the CDF of sum(lambda_i * chi2_1).
                return mp.fprod((1 + 2 * x * s) ** (-mp.mpf(".5")) for x in weights) / s

            for method in ("dehoog", "talbot"):
                survival = 1 - mp.invertlaplace(transform, mp.mpf(q), method=method)
                values[f"{method}_{precision}"] = mp.nstr(survival, precision - 5)
    oracle = float(values["dehoog_80"])
    if not 0 < oracle < 1:
        raise ArithmeticError("Oracle probability is outside its supported floating-point range")
    convergence = max(abs(float(value) / oracle - 1) for value in values.values())
    if convergence > 1e-10:
        raise ArithmeticError("Independent inverse Laplace calculations did not agree")
    angular = sphere_probability(q, lam) if len(lam) in (2, 3) else None
    # The inherited adaptive integral uses an absolute tolerance. For tiny
    # tails, inspect relative convergence of its high-order Legendre grids.
    angular_relative_error = (
        abs(angular["legendre"]["512"] / oracle - 1) if angular is not None else None
    )
    return dict(
        trait=case["trait"],
        gene=gene,
        snps=snps,
        q=q,
        raw_eigenvalues=raw_eigenvalues.tolist(),
        eigenvalues=lam.tolist(),
        cutoff=float(cutoff),
        missing_counts=(~observed).sum(axis=0).tolist(),
        oracle_p=oracle,
        inverse_laplace=values,
        inverse_laplace_relative_spread=convergence,
        angular=angular,
        angular_512_relative_error=angular_relative_error,
        fast_p=float(fast.P),
        official_p=float(official.P),
        fast_relative_error=abs(float(fast.P) / oracle - 1),
        official_relative_error=abs(float(official.P) / oracle - 1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfile", required=True, help="PLINK reference prefix")
    parser.add_argument(
        "--cases", type=Path, required=True, help="JSON list of explicit input cases"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        bed = open_bed(args.bfile + ".bed", count_A1=False, num_threads=1)
        lookup = {s: i for i, s in enumerate(bed.sid)}
        records = []
        for case in cases:
            records.append(check_case(case, bed, lookup))
            args.out.write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
