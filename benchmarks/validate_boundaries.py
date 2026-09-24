"""Compare MAGMA boundary behavior and independently integrate low-rank real genes.

Run this script on a compute allocation with the official MAGMA executable.
The independent oracle integrates radial chi-square survival over directions;
it does not use either program's characteristic-function probability solver.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from bed_reader import open_bed, to_bed
import numpy as np
import pandas as pd
from scipy import integrate, special, stats
from threadpoolctl import threadpool_limits


def sphere_probability(q, eigenvalues):
    """Return independent rank-two/three angular-integral convergence records."""
    lam = np.asarray(eigenvalues, dtype=float)
    if len(lam) not in (2, 3) or np.any(lam <= 0):
        raise ValueError("The angular oracle requires rank two or three")
    values = {}
    for order in (64, 128, 256, 512):
        x, w = special.roots_legendre(order)
        phi = (x + 1) * np.pi / 4
        variance = lam[0] * np.cos(phi) ** 2 + lam[1] * np.sin(phi) ** 2
        if len(lam) == 2:
            value = np.dot(w, np.exp(-q / (2 * variance))) / 2
        else:
            u = (x + 1) / 2
            variance = (1 - u[:, None] ** 2) * variance + lam[2] * u[:, None] ** 2
            value = np.einsum("i,ij,j", w, special.gammaincc(1.5, q / (2 * variance)), w) / 4
        values[str(order)] = float(value)
    if len(lam) == 2:
        value, error = integrate.quad(
            lambda phi: np.exp(-q / (2 * (lam[0] * np.cos(phi) ** 2 + lam[1] * np.sin(phi) ** 2))),
            0,
            np.pi / 2,
            epsabs=1e-13,
            epsrel=1e-12,
        )
    else:
        value, error = integrate.dblquad(
            lambda u, phi: special.gammaincc(
                1.5,
                q
                / (
                    2
                    * (
                        (1 - u * u) * (lam[0] * np.cos(phi) ** 2 + lam[1] * np.sin(phi) ** 2)
                        + lam[2] * u * u
                    )
                ),
            ),
            0,
            np.pi / 2,
            lambda _: 0,
            lambda _: 1,
            epsabs=1e-13,
            epsrel=1e-12,
        )
    value, error = float(value * 2 / np.pi), float(error * 2 / np.pi)
    return dict(
        legendre=values,
        adaptive_p=value,
        adaptive_error_estimate=error,
        agreement_512=abs(values["512"] - value),
    )


def execute(command, log):
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300
    )
    log.write_text(result.stdout)
    return dict(returncode=result.returncode, command=list(map(str, command)), log=str(log))


def read_table(path):
    if not path.exists():
        return []
    return pd.read_csv(path, sep=r"\s+", comment="#", dtype={"GENE": str}).to_dict("records")


def compare(args, directory, bfile, annot, pval):
    official = execute(
        [
            args.magma,
            "--bfile",
            str(bfile),
            "--pval",
            str(pval),
            "use=SNP,P",
            "ncol=N",
            "--gene-annot",
            str(annot),
            "--out",
            str(directory / "official"),
        ],
        directory / "official.console.log",
    )
    ours = execute(
        [
            sys.executable,
            "-m",
            "fastmagma",
            "run",
            "--chr",
            "22",
            "--bfile-prefix",
            str(bfile)[:-2],
            "--annot",
            str(annot),
            "--pval-dir",
            str(pval.parent),
            "--traits",
            pval.stem,
            "--out-dir",
            str(directory / "fast"),
            "--threads",
            "2",
            "--chunk-rows",
            "2",
        ],
        directory / "fast.console.log",
    )
    official["rows"] = read_table(directory / "official.genes.out")
    ours["rows"] = read_table(directory / "fast" / f"{pval.stem}.chr22.fastmagma.tsv")
    manifest = directory / "fast/chr22.manifest.json"
    if manifest.exists():
        ours["qc"] = json.loads(manifest.read_text())["qc"]
    return dict(official=official, fastmagma=ours)


def synthetic(args):
    rng = np.random.default_rng(39571)
    base = rng.binomial(2, 0.35, (100, 6)).astype(float)
    cases = [
        "complete",
        "missing_5pct_overlap",
        "missing_5pct_disjoint",
        "missing_6pct",
        "missing_25pct_overlap",
        "missing_25pct_disjoint",
        "missing_26pct",
        "block_correlated_missing",
        "all_missing",
        "monomorphic",
        "p_zero",
        "p_one",
        "p_extreme",
        "p_negative",
        "p_nan",
        "duplicate_valid",
        "duplicate_invalid",
        "missing_then_valid",
        "low_n_then_valid",
        "valid_then_missing",
        "valid_then_garbage",
        "n_49_6",
        "n_50",
        "n_50_4",
        "n_50_49",
        "n_50_5",
        "n_nan",
        "n_inf",
    ]
    records = {}
    for name in cases:
        directory = args.out_dir / "synthetic" / name
        directory.mkdir(parents=True, exist_ok=True)
        raw = base.copy()
        p, n = np.array([0.01, 0.05, 0.2, 0.4, 0.7, 0.9]), np.full(6, 1000.0)
        if name == "missing_5pct_overlap":
            raw[:5, :2] = np.nan
        elif name == "missing_5pct_disjoint":
            raw[:5, 0], raw[5:10, 1] = np.nan, np.nan
        elif name == "missing_6pct":
            raw[:6, 0] = np.nan
        elif name == "missing_25pct_overlap":
            raw[:25, :2] = np.nan
        elif name == "missing_25pct_disjoint":
            raw[:25, 0], raw[25:50, 1] = np.nan, np.nan
        elif name == "missing_26pct":
            raw[:26, 0] = np.nan
        elif name == "block_correlated_missing":
            # Two 30-SNP blocks with shared signal and disjoint missing calls.
            # Missing calls occur where genotypes equal their observed mean,
            # making corrected cross-correlations exceed one for some pairs.
            signal = np.tile([0.0, 2.0], 25)
            raw = np.ones((100, 60))
            for column in range(60):
                values = signal.copy()
                # Balanced flips preserve each column's exact mean of one.
                flips = rng.choice(25, size=3, replace=False)
                values[2 * flips] = 2
                values[2 * flips + 1] = 0
                raw[50:, column] = values
                start = 0 if column % 3 else 25
                raw[start : start + 25, column] = np.nan
            p = np.geomspace(0.001, 0.5, 60)
            n = np.full(60, 1000.0)
        elif name == "all_missing":
            raw[:, 0] = np.nan
        elif name == "monomorphic":
            raw[:, 0] = 0
        elif name.startswith("p_"):
            p[0] = {
                "p_zero": 0,
                "p_one": 1,
                "p_extreme": 1e-100,
                "p_negative": -0.1,
                "p_nan": np.nan,
            }[name]
        elif name.startswith("n_"):
            n[0] = {
                "n_49_6": 49.6,
                "n_50": 50,
                "n_50_4": 50.4,
                "n_50_49": 50.49,
                "n_50_5": 50.5,
                "n_nan": np.nan,
                "n_inf": np.inf,
            }[name]
        ids = [f"rs{i}" for i in range(raw.shape[1])]
        to_bed(
            directory / "ref.22.bed",
            raw,
            properties={
                "sid": ids,
                "chromosome": ["22"] * len(ids),
                "bp_position": list(range(1, len(ids) + 1)),
            },
        )
        annotation = directory / "genes.annot"
        annotation.write_text(
            "# window_up = 0\n# window_down = 0\n"
            + f"100\t22:1:{len(ids) + 1}\t"
            + "\t".join(ids)
            + "\n"
            + "".join(f"{101 + i}\t22:{i + 1}:{i + 2}\t{s}\n" for i, s in enumerate(ids))
        )
        table = pd.DataFrame(dict(SNP=ids, P=p, N=n))
        if name.startswith("duplicate_"):
            table = pd.concat(
                [
                    table,
                    pd.DataFrame(
                        dict(SNP=["rs0"], P=[0.02 if name == "duplicate_valid" else -1], N=[1000])
                    ),
                ]
            )
        if name in {
            "missing_then_valid",
            "low_n_then_valid",
            "valid_then_missing",
            "valid_then_garbage",
        }:
            if name == "missing_then_valid":
                table["P"] = table["P"].astype(object)
                table.loc[0, "P"] = "NA"
            elif name == "low_n_then_valid":
                table.loc[0, "N"] = 49.6
            extra_p = {
                "missing_then_valid": 0.02,
                "low_n_then_valid": 0.02,
                "valid_then_missing": "NA",
                "valid_then_garbage": "garbage",
            }[name]
            table = pd.concat([table, pd.DataFrame(dict(SNP=["rs0"], P=[extra_p], N=[1000]))])
        pval = directory / "fixture.pval"
        table.to_csv(pval, sep="\t", index=False, na_rep="NaN")
        records[name] = compare(args, directory, directory / "ref.22", annotation, pval)
        print("Completed synthetic", name, flush=True)
        (args.out_dir / "synthetic.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def real_lowrank(args):
    identifiers = {"56734", "56733", "56632"}
    directory = args.out_dir / "real"
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        line
        for line in Path(args.annot).read_text().splitlines(True)
        if line.startswith("#") or line.split("\t")[0] in identifiers
    ]
    annotation = directory / "genes.annot"
    annotation.write_text("".join(lines))
    gene_snps = {
        line.split("\t")[0]: line.rstrip().split("\t")[2:]
        for line in lines
        if not line.startswith("#")
    }
    if set(gene_snps) != identifiers:
        raise ValueError("Missing requested real gene annotations")
    wanted = set().union(*map(set, gene_snps.values()))
    selected = []
    for chunk in pd.read_csv(args.ad_pval, sep=r"\s+", chunksize=100000, dtype={"SNP": str}):
        selected.append(chunk[chunk.SNP.isin(wanted)])
    table = pd.concat(selected)
    pval = directory / "ad.pval"
    table.to_csv(pval, sep="\t", index=False)
    compared = compare(args, directory, Path(args.bfile), annotation, pval)
    bed = open_bed(str(args.bfile) + ".bed", count_A1=False, num_threads=2)
    lookup = {s: i for i, s in enumerate(bed.sid)}
    by_snp = table.set_index("SNP")
    records = {}
    for gene, snps in gene_snps.items():
        snps = sorted(set(snps) & set(lookup) & set(by_snp.index), key=lookup.get)
        raw = bed.read(index=np.s_[:, [lookup[s] for s in snps]], dtype="float64")
        # Independent direct mean imputation and Pearson-correlation construction.
        means = np.nanmean(raw, axis=0)
        filled = np.where(np.isnan(raw), means[None, :], raw)
        keep = np.std(filled, axis=0) > 0
        filled = filled[:, keep]
        snps = list(np.array(snps)[keep])
        p = by_snp.loc[snps, "P"].to_numpy(float)
        q = float(stats.chi2.isf(p, 1).sum())
        spectrum = np.linalg.eigvalsh(np.corrcoef(filled, rowvar=False))
        oracle = sphere_probability(q, spectrum)
        oracle.update(
            snps=snps,
            q=q,
            eigenvalues=spectrum.tolist(),
            missing_counts=np.isnan(raw).sum(axis=0).tolist(),
        )
        for label in ("official", "fastmagma"):
            row = next((r for r in compared[label]["rows"] if r["GENE"] == gene), None)
            oracle[label] = row
            if row is not None:
                oracle[label + "_absolute_error"] = abs(float(row["P"]) - oracle["adaptive_p"])
        records[gene] = oracle
    result = dict(comparison=compared, independent_oracles=records)
    (args.out_dir / "real_lowrank.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magma", required=True)
    parser.add_argument("--bfile", required=True)
    parser.add_argument("--annot", required=True)
    parser.add_argument("--ad-pval", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        real_lowrank(args)
        synthetic(args)


if __name__ == "__main__":
    main()
