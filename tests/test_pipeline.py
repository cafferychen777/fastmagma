"""Installed CLI end-to-end tests using small synthetic PLINK files."""

import json
import subprocess
import sys

from bed_reader import to_bed
import numpy as np
import pandas as pd
import pytest

from fastmagma.cli import main


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(9)
    raw = rng.integers(0, 3, (50, 8)).astype(float)
    raw[:, 7] = 0
    raw[:2, 0] = np.nan
    to_bed(
        tmp_path / "ref.22.bed",
        raw,
        properties={"sid": [f"rs{i}" for i in range(8)], "chromosome": ["22"] * 8},
    )
    (tmp_path / "genes.annot").write_text(
        "001\t22:1:2\trs0\nG2\t22:1:8\trs0\trs1\trs2\nG3\t22:2:9\trs1\trs2\trs3\nG4\t22:9:10\trs7\n"
    )
    for trait in ["a", "b"]:
        (tmp_path / f"{trait}.pval").write_text(
            "SNP P N\n" + "".join(f"rs{i} {0.01 * (i + 1)} 1000\n" for i in range(8))
        )
    args = [
        "run",
        "--chr",
        "22",
        "--bfile-prefix",
        str(tmp_path / "ref."),
        "--annot",
        str(tmp_path / "genes.annot"),
        "--pval-dir",
        str(tmp_path),
        "--traits",
        "a,b",
        "--out-dir",
        str(tmp_path / "out"),
        "--chunk-rows",
        "2",
    ]
    return tmp_path, args


def test_cli_run_merge_and_reuse(inputs):
    p, args = inputs
    # A subprocess outside the checkout exercises the installed module.
    subprocess.run([sys.executable, "-m", "fastmagma", *args], cwd=p, check=True)
    a = pd.read_csv(p / "out/a.chr22.fastmagma.tsv", sep="\t", dtype={"GENE": str})
    b = pd.read_csv(p / "out/b.chr22.fastmagma.tsv", sep="\t", dtype={"GENE": str})
    pd.testing.assert_frame_equal(a, b)
    assert a.GENE.tolist() == ["001", "G2", "G3"]
    assert a.P.iloc[0] == pytest.approx(0.01)
    manifest = json.loads((p / "out/chr22.manifest.json").read_text())
    assert manifest["resources"]["eigendecompositions"] == 3
    assert manifest["resources"]["block_reads"] == 1
    assert main(args) == 1
    merge = ["merge", "--traits", "a,b", "--out-dir", str(p / "out"), "--chrs", "21-22"]
    assert main(merge) == 1
    assert not (p / "out/a.genes.out").exists()
    assert main(merge + ["--allow-missing"]) == 0
    merged = pd.read_csv(p / "out/a.genes.out", sep="\t", dtype={"GENE": str})
    assert "001" in merged.GENE.tolist()
    assert json.loads((p / "out/merge.manifest.json").read_text())["missing"]


def test_trait_missingness_and_uncached_equivalence(inputs):
    p, args = inputs
    (p / "b.pval").write_text("SNP P N\nrs0 .01 1000\nrs2 .03 1000\n")
    assert main(args) == 0
    expected = (p / "out/b.chr22.fastmagma.tsv").read_bytes()
    assert main(args + ["--overwrite", "--cache-mb", "0", "--block-snps", "2"]) == 0
    assert expected == (p / "out/b.chr22.fastmagma.tsv").read_bytes()
    b = pd.read_csv(p / "out/b.chr22.fastmagma.tsv", sep="\t")
    assert b.NSNPS.tolist() == [1, 2, 1]


def test_resource_failure_does_not_publish(inputs):
    p, args = inputs
    assert main(args + ["--max-gene-snps", "1"]) == 1
    assert not list((p / "out").glob("*.tsv"))
    assert not list((p / "out").glob("*.json"))
    assert not list((p / "out").glob(".fastmagma-*"))


def test_merge_rejects_wrong_chromosome(inputs):
    p, args = inputs
    assert main(args) == 0
    path = p / "out/a.chr22.fastmagma.tsv"
    df = pd.read_csv(path, sep="\t")
    df["CHR"] = 21
    df.to_csv(path, sep="\t", index=False)
    assert main(["merge", "--traits", "a", "--out-dir", str(p / "out"), "--chrs", "22"]) == 1


def test_version_and_help_without_numpy(tmp_path):
    code = "import sys; from fastmagma.cli import main\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, capture_output=True)


def test_lock_and_manifest_integrity(inputs):
    p, args = inputs
    out = p / "out"
    out.mkdir()
    (out / ".fastmagma.chr22.lock").write_text("active process")
    assert main(args) == 1
    assert (out / ".fastmagma.chr22.lock").read_text() == "active process"
    (out / ".fastmagma.chr22.lock").unlink()
    assert main(args) == 0
    assert not (out / ".fastmagma.chr22.lock").exists()
    (out / "chr22.manifest.json").unlink()
    assert main(["merge", "--traits", "a", "--out-dir", str(out), "--chrs", "22"]) == 1


def test_workspace_failure_and_preserve_prior_output(inputs):
    p, args = inputs
    assert main(args) == 0
    original = (p / "out/a.chr22.fastmagma.tsv").read_bytes()
    assert main(args + ["--overwrite", "--workspace-mb", "1", "--block-snps", "100000"]) == 1
    assert (p / "out/a.chr22.fastmagma.tsv").read_bytes() == original
    assert not (p / "out/.fastmagma.chr22.lock").exists()


def test_whole_model_preserves_p_one(inputs):
    p, args = inputs
    for trait in ["a", "b"]:
        (p / f"{trait}.pval").write_text("SNP P N\n" + "".join(f"rs{i} 1 1000\n" for i in range(8)))
    assert main(args + ["--model", "whole"]) == 0
    assert main(["merge", "--traits", "a,b", "--out-dir", str(p / "out"), "--chrs", "22"]) == 0
    a = pd.read_csv(p / "out/a.genes.out", sep="\t")
    assert np.isneginf(a.ZSTAT).all()
    assert (a.P == 1).all()


def test_magma_input_clipping_and_integer_mean_n(inputs):
    p, args = inputs
    for trait in ["a", "b"]:
        (p / f"{trait}.pval").write_text("SNP P N\nrs0 1 50.5\nrs1 .1 51.5\nrs2 .2 53\n")
    assert main(args) == 0
    a = pd.read_csv(p / "out/a.chr22.fastmagma.tsv", sep="\t", dtype={"GENE": str})
    a = a.set_index("GENE")
    assert a.loc["001", "P"] == pytest.approx(1 - 1e-5)
    assert a.loc["001", "N"] == 51
    assert a.loc["G3", "N"] == 53


def test_independent_chromosome_locks_and_merge_exclusion(tmp_path):
    from fastmagma.pipeline import locked_outputs

    with locked_outputs(tmp_path, [22]):
        with locked_outputs(tmp_path, [21]):
            assert (tmp_path / ".fastmagma.chr21.lock").exists()
        with pytest.raises(RuntimeError, match="locked"):
            with locked_outputs(tmp_path, [21, 22], merging=True):
                pytest.fail("Merge acquired a locked chromosome")
        assert not (tmp_path / ".fastmagma.merge.lock").exists()
        assert not (tmp_path / ".fastmagma.chr21.lock").exists()
        assert (tmp_path / ".fastmagma.chr22.lock").exists()
    assert not list(tmp_path.glob("*.lock"))


def test_large_gene_model_selection_and_io_independence(tmp_path):
    """Model selection changes the test, while I/O and shared traits do not."""
    rng = np.random.default_rng(314)
    raw = rng.integers(0, 3, (50, 60)).astype(float)
    snps = [f"rs{i}" for i in range(raw.shape[1])]
    to_bed(
        tmp_path / "ref.22.bed",
        raw,
        properties={"sid": snps, "chromosome": ["22"] * len(snps)},
    )
    (tmp_path / "genes.annot").write_text("large\t22:1:60\t" + "\t".join(snps) + "\n")
    for trait in ("a", "b"):
        (tmp_path / f"{trait}.pval").write_text(
            "SNP P N\n" + "".join(f"{s} {0.1 + i / 100} 1000\n" for i, s in enumerate(snps))
        )
    args = [
        "run",
        "--chr",
        "22",
        "--bfile-prefix",
        str(tmp_path / "ref."),
        "--annot",
        str(tmp_path / "genes.annot"),
        "--pval-dir",
        str(tmp_path),
        "--traits",
        "a,b",
        "--out-dir",
        str(tmp_path / "out"),
    ]
    assert main(args) == 0
    out = tmp_path / "out"
    default = (out / "a.chr22.fastmagma.tsv").read_bytes()
    assert default == (out / "b.chr22.fastmagma.tsv").read_bytes()
    metadata = json.loads((out / "chr22.manifest.json").read_text())
    assert metadata["model"] == "magma"
    # Three shared statistical blocks, not six separate trait computations.
    assert metadata["resources"]["eigendecompositions"] == 3
    assert (
        main(args + ["--overwrite", "--model", "magma", "--block-snps", "7", "--cache-mb", "0"])
        == 0
    )
    assert default == (out / "a.chr22.fastmagma.tsv").read_bytes()
    assert main(args + ["--overwrite", "--model", "whole"]) == 0
    metadata = json.loads((out / "chr22.manifest.json").read_text())
    assert metadata["model"] == "whole"
    assert metadata["resources"]["eigendecompositions"] == 1
    whole = pd.read_csv(out / "a.chr22.fastmagma.tsv", sep="\t")
    assert (out / "a.chr22.fastmagma.tsv").read_bytes() != default
    from fastmagma.genotypes import correlation_spectrum, normalize_genotypes
    from fastmagma.stats import gene_test
    from scipy.special import ndtri

    p = np.array([0.1 + i / 100 for i in range(len(snps))])
    g, keep = normalize_genotypes(raw)
    reference = gene_test(float(np.square(ndtri(p / 2)).sum()), correlation_spectrum(g[:, keep]))
    assert whole.P.iloc[0] == pytest.approx(reference.p, rel=1e-12)


@pytest.mark.parametrize("second_model", ["whole", None, "magma"])
def test_merge_rejects_mixed_models_and_legacy_outputs(inputs, second_model):
    import hashlib

    p, args = inputs
    assert main(args) == 0
    out = p / "out"
    source = pd.read_csv(out / "a.chr22.fastmagma.tsv", sep="\t", dtype={"GENE": str})
    source["CHR"] = 21
    source["GENE"] = "chr21_" + source.GENE
    second = out / "a.chr21.fastmagma.tsv"
    source.to_csv(second, sep="\t", index=False)
    metadata = {
        "outputs": {second.name: {"sha256": hashlib.sha256(second.read_bytes()).hexdigest()}}
    }
    if second_model is not None:
        metadata["model"] = second_model
    (out / "chr21.manifest.json").write_text(json.dumps(metadata))
    merge = ["merge", "--traits", "a", "--out-dir", str(out), "--chrs", "21-22"]
    assert main(merge) == 1
    assert not (out / "a.genes.out").exists()
    # A legacy completion record represents the historical whole-gene model.
    first_marker = out / "chr22.manifest.json"
    first = json.loads(first_marker.read_text())
    first["model"] = second_model or "whole"
    first["model_revision"] = 1
    first_marker.write_text(json.dumps(first))
    assert main(merge) == 0
    assert json.loads((out / "merge.manifest.json").read_text())["model"] == (
        second_model or "whole"
    )


def test_reference_order_deduplicates_annotation_and_ignores_unneeded_metadata(inputs):
    p, args = inputs
    assert main(args) == 0
    expected = (p / "out/a.chr22.fastmagma.tsv").read_bytes()
    # Unused PLINK metadata need not be parsed into numerical arrays.
    for name, columns in [("ref.22.bim", [2, 3]), ("ref.22.fam", [4])]:
        path = p / name
        lines = []
        for line in path.read_text().splitlines():
            fields = line.split()
            for column in columns:
                fields[column] = "unused"
            lines.append("\t".join(fields))
        path.write_text("\n".join(lines) + "\n")
    annotation = p / "genes.annot"
    annotation.write_text(
        annotation.read_text().replace("rs0\trs1\trs2", "rs2\tmissing\trs0\trs1\trs2")
    )
    assert main(args + ["--overwrite"]) == 0
    assert (p / "out/a.chr22.fastmagma.tsv").read_bytes() == expected


@pytest.mark.parametrize("late_line", ["malformed\n", "001\t22:10:12\trs1\n"])
def test_late_annotation_failure_preserves_published_outputs(inputs, late_line):
    root, args = inputs
    assert main(args) == 0
    output = root / "out"
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    annotation = root / "genes.annot"
    annotation.write_text(annotation.read_text() + late_line)
    assert main(args + ["--overwrite"]) == 1
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert before == after
    assert not list(output.glob(".fastmagma-*"))


@pytest.mark.parametrize(
    "annotation,error",
    [
        ("G1\t1:1:3\trs0\n", "No genes annotated on chromosome 22"),
        ("G1\t22:1:3\tabsent\n", "No annotated SNPs occur in the reference panel"),
    ],
)
def test_streamed_annotation_empty_or_absent_reference(inputs, annotation, error, capsys):
    root, args = inputs
    (root / "genes.annot").write_text(annotation)
    assert main(args) == 1
    assert error in capsys.readouterr().err
    assert not list((root / "out").glob("*.tsv"))
    assert not list((root / "out").glob("*.manifest.json"))
    assert not list((root / "out").glob(".fastmagma-*"))
