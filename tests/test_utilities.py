"""Source utilities preserve IDs and compare the intended MAGMA subset."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
from scipy.special import ndtri_exp
import numpy as np

from magma_py.io import COLUMNS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_mapping_and_comparison(tmp_path):
    rows = []
    for identifier, p in [("001", 0.1), ("002", 0.01), ("003", 1e-20)]:
        rows.append(
            [identifier, 22, 1, 2, 3, 2, 1000, -ndtri_exp(np.log(p)), p, np.log10(p), "imhof"]
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(tmp_path / "x.genes.out", sep="\t", index=False)
    frame.to_csv(tmp_path / "x.chr22.magma_py.tsv", sep="\t", index=False)
    (tmp_path / "x").mkdir()
    frame.to_csv(tmp_path / "x/x.batch22_chr.genes.out", sep="\t", index=False)
    (tmp_path / "map.tsv").write_text(
        "gene_loc_id\tsymbol\tchr\tstart\tstop\n001\tLDLR\t22\t1\t2\n002\tAPOB\t22\t1\t2\n003\tAPOE\t22\t1\t2\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_gene_z_matrix.py"),
            "--genes-dir",
            str(tmp_path),
            "--traits",
            "x",
            "--map",
            str(tmp_path / "map.tsv"),
            "--out",
            str(tmp_path / "matrix.tsv"),
        ],
        check=True,
    )
    matrix = pd.read_csv(tmp_path / "matrix.tsv", sep="\t", dtype={"gene_loc_id": str})
    assert matrix.gene_loc_id.tolist() == ["001", "002", "003"]
    assert matrix.symbol.tolist() == ["LDLR", "APOB", "APOE"]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate.py"),
            "--traits",
            "x",
            "--chrs",
            "22",
            "--fast-dir",
            str(tmp_path),
            "--magma-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "compare.csv"),
        ],
        check=True,
    )
    report = pd.read_csv(tmp_path / "compare.csv").iloc[0]
    assert report.n_bulk == 2 and report.n_matched == 3 and report.top20_overlap == 3
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "ldl_rank_check.py"),
            "--genes-out",
            str(tmp_path / "x.genes.out"),
            "--map",
            str(tmp_path / "map.tsv"),
        ],
        check=True,
    )
