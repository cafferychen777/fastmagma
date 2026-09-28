"""Input QC includes duplicate IDs across chunks and nonfinite values."""

import numpy as np
import pytest

from fastmagma.io import TraitStore, load_annotation, parse_chromosomes, parse_traits


def test_qc_duplicates_across_chunks_and_invalid_rows(tmp_path):
    (tmp_path / "a.pval").write_text(
        "SNP P N\nr1 0.1 100\nr2 1e-320 100\nr3 0.2 inf\n"
        "r4 0 100\nr5 bad 100\nr1 0.2 1\nr6 1 50\nr7 0.1 49\n"
    )
    store = TraitStore(tmp_path, ["a"], [f"r{i}" for i in range(1, 8)])
    try:
        store.load(tmp_path, chunk_rows=2)
        v = store.values[0].copy()
        assert np.flatnonzero(np.isfinite(v[0])).tolist() == [1, 5]
        assert v[2, 1] > 1400
        assert v[2, 5] == 0
        assert store.qc["a"]["duplicate_snps"] == 1
    finally:
        store.close()


def test_same_chunk_duplicates_and_gene_ids(tmp_path):
    (tmp_path / "a.pval").write_text("SNP P N\nr1 .1 100\nr1 .2 100\nr2 .1 100\n")
    store = TraitStore(tmp_path, ["a"], ["r1", "r2"])
    try:
        store.load(tmp_path, 100)
        assert np.isnan(store.values[0, 0, 0])
        assert store.values[0, 0, 1] == 0.1
    finally:
        store.close()
    p = tmp_path / "g.annot"
    p.write_text("001\t22:1:2\tr1\tr1\n")
    g = load_annotation(p, 22)
    assert g[0].identifier == "001" and g[0].snps == ("r1",)
    p.write_text("001\t22:1:2\tr1\n001\t22:2:3\tr2\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_annotation(p, 22)


@pytest.mark.parametrize("value", ["", "a,a", "a,", "../a", "a/b"])
def test_invalid_traits(value):
    with pytest.raises(ValueError):
        parse_traits(value)


@pytest.mark.parametrize("value", ["", "22-1", "1,1", "0", "23", "1-2-3"])
def test_invalid_chromosomes(value):
    with pytest.raises(ValueError):
        parse_chromosomes(value)


@pytest.mark.parametrize("chunk_rows", [1, 2, 100])
def test_magma_ordered_duplicates_and_missing_rows(tmp_path, chunk_rows):
    (tmp_path / "a.pval").write_text(
        "SNP P N\n"
        "missing_p NA garbage\nmissing_p .1 100\n"
        "missing_n .1 NA\nmissing_n .2 100\n"
        "small_n .1 50\nsmall_n .2 100\n"
        "duplicate .1 100\nduplicate garbage garbage\n"
        "missing_sentinel -1 100\nmissing_sentinel .1 100\n"
        "valid .2 100\n"
    )
    snps = ["missing_p", "missing_n", "small_n", "duplicate", "missing_sentinel", "valid"]
    store = TraitStore(tmp_path, ["a"], snps, model="magma")
    try:
        store.load(tmp_path, chunk_rows)
        assert np.flatnonzero(np.isfinite(store.values[0, 0])).tolist() == [5]
        assert store.qc["a"]["duplicate_snps"] == 1
        assert store.qc["a"]["invalid_rows"] == 4
    finally:
        store.close()


def test_magma_p_truncation_and_rounded_sample_size(tmp_path):
    (tmp_path / "a.pval").write_text(
        "SNP P N\nzero 0 100\nlow 1e-320 100\none 1 100\n"
        "n49 .1 49.5\nn50 .1 50\nnlow .1 50.499\nhalf .1 50.5\n"
        "nhigh .1 51.5\n"
    )
    snps = ["zero", "low", "one", "n49", "n50", "nlow", "half", "nhigh"]
    store = TraitStore(tmp_path, ["a"], snps, model="magma")
    try:
        store.load(tmp_path, 2)
        values = store.values[0]
        np.testing.assert_array_equal(values[0, :3], [1e-50, 1e-50, 1 - 1e-5])
        assert np.isnan(values[0, 3:6]).all()
        np.testing.assert_array_equal(values[1, 6:], [51, 52])
        assert np.isfinite(values[:, [0, 1, 2, 6, 7]]).all()
    finally:
        store.close()


@pytest.mark.parametrize(
    "p,n,error",
    [
        ("-0.1", "100", "P"),
        ("1.01", "100", "P"),
        ("bad", "100", "P"),
        ("inf", "100", "P"),
        ("nan", "100", "P"),
        (".1", "-1", "N"),
        (".1", "bad", "N"),
        (".1", "inf", "N"),
        (".1", "2147483648", "N"),
    ],
)
def test_magma_invalid_first_row_is_an_error(tmp_path, p, n, error):
    (tmp_path / "a.pval").write_text(f"SNP P N\nr1 {p} {n}\nr1 .1 100\n")
    store = TraitStore(tmp_path, ["a"], ["r1"], model="magma")
    try:
        with pytest.raises(ValueError, match=f"Invalid SNP {error}"):
            store.load(tmp_path, 100)
    finally:
        store.close()


@pytest.mark.parametrize("chunk_rows", [1, 2, 100])
@pytest.mark.parametrize("model", ["whole", "magma"])
def test_reference_filter_preserves_qc(tmp_path, monkeypatch, chunk_rows, model):
    import fastmagma.io as io

    (tmp_path / "a.pval").write_text(
        "EXTRA N SNP P\n"
        "x 100 outside garbage\n"
        "x 100 r1 .1\n"
        "x NA r2 .2\n"
        "x 100 outside2 garbage\n"
        "x 100 r1 .3\n"
        "x 100 r2 .2\n"
        "x 100 r3 .5\n"
    )
    store = TraitStore(tmp_path, ["a"], ["r1", "r2", "r3"], model=model)
    try:
        store.load(tmp_path, chunk_rows)
        actual, qc = store.values.copy(), store.qc.copy()
        monkeypatch.setattr(io, "filter_pval", lambda *args: None)
        store.load(tmp_path, chunk_rows)
        np.testing.assert_array_equal(actual, store.values)
        assert qc == store.qc
    finally:
        store.close()
    assert not list(tmp_path.glob("tmp*.pval"))


@pytest.mark.parametrize(
    "content",
    [
        'SNP P N\n"r1" .1 100\n',
        "SNP P N\nr1 .1\nr2 .2 100\n",
        "\ufeffSNP P N\nr1 .1 100\n",
        "SNP P N\nr1\x00ignored .1 100\n",
    ],
)
def test_reference_filter_delegates_general_tables(tmp_path, content):
    from fastmagma._input import filter_pval

    source, destination = tmp_path / "input", tmp_path / "output"
    source.write_text(content)
    assert filter_pval(source, ["r1"], destination) is None


def test_reference_filter_handles_line_boundaries(tmp_path):
    from fastmagma._input import filter_pval

    source, destination = tmp_path / "input", tmp_path / "output"
    long_id = "x" * 140000
    source.write_bytes(f"\nN\tSNP P\r\n100 outside .3\n100\t{long_id}\t.2\n\n100 r1 .1".encode())
    assert filter_pval(source, ["r1", long_id], destination) == 3
    assert destination.read_text() == f"N\tSNP P\n100\t{long_id}\t.2\n100 r1 .1"


def test_reference_filter_unicode_paths_and_error_cleanup(tmp_path):
    from fastmagma._input import filter_pval

    directory = tmp_path / "统计"
    directory.mkdir()
    source, destination = directory / "输入.pval", directory / "输出.pval"
    source.write_text("SNP P N\nr1 .1 100\n")
    assert filter_pval(source, ["r1"], destination) == 1
    assert destination.read_text() == source.read_text()
    with pytest.raises(OSError):
        filter_pval(source, ["r1"], directory / "missing" / "output")
    source.unlink()  # Open handles would prevent this on Windows.
    with pytest.raises(OSError):
        filter_pval(source, ["r1"], destination)
    destination.unlink()


@pytest.mark.parametrize("separator", ["\v", "\f"])
def test_reference_filter_preserves_unusual_whitespace_qc(tmp_path, monkeypatch, separator):
    import fastmagma.io as io

    (tmp_path / "a.pval").write_text(f"SNP P N\n{separator}\nr1 .2 100\n")
    store = TraitStore(tmp_path, ["a"], ["r1"])
    try:
        store.load(tmp_path, 100)
        values, qc = store.values.copy(), store.qc.copy()
        monkeypatch.setattr(io, "filter_pval", lambda *args: None)
        store.load(tmp_path, 100)
        np.testing.assert_array_equal(values, store.values)
        assert qc == store.qc
        assert qc["a"]["rows"] == 2
    finally:
        store.close()
