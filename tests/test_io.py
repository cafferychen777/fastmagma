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
    source.write_text(content, encoding="utf-8")
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


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "-0",
        "+.1",
        "1.",
        " .1 ",
        "\t.1\t",
        "1e-320",
        "4.9406564584124654e-324",
        "1.7976931348623157e308",
        "1e309",
        "1.e-320",
        "+1e+20",
        "1_0",
        "0x1",
        "nan",
        "NaN",
        "inf",
        "+Inf",
        "-Infinity",
        "NA",
        "",
        "\u00a0.1\u00a0",
    ],
)
def test_numeric_parser_matches_decimal_acceptance(text):
    import pandas as pd
    from fastmagma.io import _numeric

    expected = pd.to_numeric(pd.Series([text]), errors="coerce").to_numpy(dtype=float)
    actual = _numeric(np.asarray([text], dtype=object))
    # pandas versions differ on overflow (NaN versus infinity); both are
    # rejected by QC. Require matching finite acceptance and finite values.
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-14, atol=0)


def test_quoted_numeric_whitespace_preserves_qc(tmp_path):
    (tmp_path / "a.pval").write_text('SNP P N\nr1 " .2 " " 100 "\n')
    store = TraitStore(tmp_path, ["a"], ["r1"], model="magma")
    try:
        store.load(tmp_path, 100)
        assert store.values[0, 0, 0] == 0.2
        assert store.values[0, 1, 0] == 100
    finally:
        store.close()


def test_plain_trait_loading_does_not_import_pandas(tmp_path):
    import subprocess
    import sys

    (tmp_path / "a.pval").write_text("SNP P N\nr1 .2 100\n")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from fastmagma.io import TraitStore; "
            "store = TraitStore(sys.argv[1], ['a'], ['r1']); "
            "store.load(sys.argv[1], 100); store.close(); "
            "assert 'pandas' not in sys.modules",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_snp_index_reuses_single_lookup(tmp_path):
    from fastmagma.io import SnpIndex

    index = SnpIndex(["r2", "r1"])
    np.testing.assert_array_equal(index.get_indexer(["r1", "missing", "r2"]), [1, -1, 0])
    store = TraitStore(tmp_path, ["a"], index)
    try:
        assert store.index is index
    finally:
        store.close()


@pytest.mark.parametrize(
    "content,error",
    [
        ("22\t\t0\t1\tA\tC\n", "nonempty"),
        ("22\tr1\t0\t1\tA\tC\n22\tr1\t0\t2\tA\tC\n", "unique"),
        ("21\tr1\t0\t1\tA\tC\n", "chromosome"),
        ("22\n", "Malformed BIM"),
    ],
)
def test_streamed_reference_index_rejects_invalid_metadata(tmp_path, content, error):
    from fastmagma.io import load_reference_index

    path = tmp_path / "reference.bim"
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        load_reference_index(path, 22)


def test_streamed_reference_index_preserves_order_and_ignores_unused_fields(tmp_path):
    from fastmagma.io import load_reference_index

    path = tmp_path / "reference.bim"
    path.write_text("# comment\n\n22\tr2\tbad\tbad\tA\tC\n22\tr1\t0\t1\tA\tC # comment\n")
    index = load_reference_index(path, 22)
    assert list(index) == ["r2", "r1"]
    np.testing.assert_array_equal(index.get_indexer(["r1", "r2"]), [1, 0])


@pytest.mark.parametrize(
    "value,rounded",
    [
        (np.nextafter(0.5, 0), 1),
        (0.5, 1),
        (np.nextafter(0.5, 1), 1),
        (np.nextafter(-0.5, 0), -1),
        (np.nextafter(50.5, 0), 50),
        (50.5, 51),
        (np.nextafter(50.5, 51), 51),
    ],
)
def test_magma_rounding_adjacent_to_halfway_values(tmp_path, value, rounded):
    store = TraitStore(tmp_path, ["a"], ["r1"], model="magma")
    try:
        if rounded < 0:
            with pytest.raises(ValueError, match="Invalid SNP N"):
                store._parse_rows(
                    {"P": np.array([".1"]), "N": np.array([repr(float(value))])},
                    tmp_path,
                )
            return
        _, n, good = store._parse_rows(
            {"P": np.array([".1"]), "N": np.array([repr(float(value))])},
            tmp_path,
        )
        assert n[0] == rounded
        assert good[0] == (rounded > 50)
    finally:
        store.close()


@pytest.mark.parametrize("value", [-0.5, np.nextafter(-0.5, -1)])
def test_magma_negative_halfway_rounds_away_from_zero(tmp_path, value):
    store = TraitStore(tmp_path, ["a"], ["r1"], model="magma")
    try:
        with pytest.raises(ValueError, match="Invalid SNP N"):
            store._parse_rows(
                {"P": np.array([".1"]), "N": np.array([repr(float(value))])},
                tmp_path,
            )
    finally:
        store.close()


def test_native_numeric_matches_previous_float_conversion():
    import re
    from fastmagma.io import _numeric

    pattern = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
    tokens = [
        "0",
        "-0",
        " .2\t",
        "1e-320",
        "4.9406564584124654e-324",
        "1e309",
        "+Infinity",
        "-INF",
        "1_0",
        "0x1p0",
        "1.2junk",
        "",
        "\x001",
        "1\x00",
        "nan",
        "-NaN",
        "NA",
        "１２",
        "\u00a0.2",
        "-0.49999999999999994",
        "50.49999999999999",
        "50.5",
    ]
    rng = np.random.default_rng(126)
    tokens += [format(value, ".17g") for value in rng.normal(size=1000)]
    expected = []
    for token in tokens:
        token = token.strip(" \t\r\n\v\f")
        expected.append(
            float(token)
            if pattern.fullmatch(token)
            or token.lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
            else np.nan
        )
    expected = np.asarray(expected)
    actual = _numeric(np.asarray(tokens, dtype=object))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint64), expected[valid].view(np.uint64))


@pytest.mark.parametrize(
    "output",
    [
        np.empty(2, dtype=np.float32),
        np.empty((1, 2)),
        np.empty(4)[::2],
        np.empty(1),
        np.empty(3),
    ],
)
def test_native_numeric_rejects_invalid_output_buffers(output):
    from fastmagma._input import parse_numeric

    with pytest.raises(ValueError):
        parse_numeric([".1", ".2"], output)


def test_native_numeric_rejects_readonly_and_nonstring_inputs():
    from fastmagma._input import parse_numeric

    output = np.empty(1)
    output.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        parse_numeric([".1"], output)
    with pytest.raises(TypeError):
        parse_numeric([123], np.empty(1))
    parse_numeric([], np.empty(0))


def test_reference_filter_accepts_mapping_keys_without_changing_rows(tmp_path):
    from fastmagma._input import filter_pval

    source = tmp_path / "input"
    source.write_text("SNP P N\nr2 .2 100\noutside .3 100\nr1 .1 100\nr1 NA bad\n")
    mapping = {"r1": 1, "r2": 0}
    outputs = [tmp_path / "list", tmp_path / "mapping"]
    assert filter_pval(source, list(mapping), outputs[0]) == 4
    assert filter_pval(source, mapping, outputs[1]) == 4
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert mapping == {"r1": 1, "r2": 0}


def test_native_numeric_handles_unaligned_buffers_and_surrogates():
    from fastmagma._input import parse_numeric

    backing = bytearray(17)
    output = np.ndarray((2,), dtype=np.float64, buffer=backing, offset=1)
    assert not output.flags.aligned
    parse_numeric([".2", "\ud800"], output)
    assert output[0] == 0.2
    assert np.isnan(output[1])


def test_annotation_iterator_matches_materialized_order(tmp_path):
    from fastmagma.io import iter_annotation

    path = tmp_path / "genes.annot"
    path.write_text("# comment\nG0\t1:1:2\trs0\nG2\t22:1:3\trs2\trs1\trs2\nG1\t22:2:4\trs1\n")
    assert list(iter_annotation(path, 22)) == load_annotation(path, 22)
    assert [gene.identifier for gene in iter_annotation(path, 22)] == ["G2", "G1"]


def test_annotation_iterator_validates_late_rows_on_consumption(tmp_path):
    from fastmagma.io import iter_annotation

    path = tmp_path / "genes.annot"
    path.write_text("G1\t22:1:3\trs1\nmalformed\n")
    genes = iter_annotation(path, 22)
    assert next(genes).identifier == "G1"
    with pytest.raises(ValueError, match="Malformed annotation"):
        next(genes)
