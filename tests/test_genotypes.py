"""Correlation invariants, dual-Gram equivalence, and cache limits."""

import numpy as np
import pytest
from bed_reader import open_bed, to_bed

from fastmagma.genotypes import GenotypeReader, correlation_spectrum, normalize_genotypes


def test_missing_data_correlation_matches_explicit_pearson():
    raw = np.array(
        [
            [0, 0, np.nan, 1],
            [1, 1, np.nan, 1],
            [2, np.nan, np.nan, 1],
            [0, 2, np.nan, 1],
            [1, 1, np.nan, 1],
            [2, 0, np.nan, 1.0],
        ]
    )
    g, keep = normalize_genotypes(raw)
    assert keep.tolist() == [True, True, False, False]
    used = raw[:, :2].copy()
    used[np.isnan(used)] = np.nanmean(used, axis=0)[np.where(np.isnan(used))[1]]
    assert g[:, keep].T @ g[:, keep] == pytest.approx(np.corrcoef(used.T), abs=1e-14)
    assert np.isfinite(g).all()


@pytest.mark.parametrize("n,k", [(30, 5), (10, 100), (20, 20)])
def test_dual_spectrum_matches_svd(n, k):
    raw = np.random.default_rng(10).integers(0, 3, size=(n, k)).astype(float)
    raw[0, 0] = np.nan
    g, keep = normalize_genotypes(raw)
    g = g[:, keep]
    expected = np.linalg.svd(g, compute_uv=False) ** 2
    expected = np.sort(expected[expected > 1e-10])
    actual = correlation_spectrum(g)
    assert actual == pytest.approx(expected, rel=1e-11, abs=1e-12)
    assert actual.sum() == pytest.approx(g.shape[1])


def test_block_cache_bounded_and_reused(tmp_path):
    raw = np.random.default_rng(4).integers(0, 3, size=(40, 50)).astype(float)
    to_bed(tmp_path / "g.bed", raw)
    reader = GenotypeReader(open_bed(tmp_path / "g.bed"), cache_mb=0.007, block_snps=10)
    a, keep = reader.read([1, 2, 13])
    reads = reader.block_reads
    b, _ = reader.read([1, 2, 13])
    assert a == pytest.approx(b)
    assert reader.block_reads == reads
    reader.read([20, 30, 40])
    assert reader.cache_bytes <= reader.capacity
    uncached = GenotypeReader(reader.bed, cache_mb=0, block_snps=10)
    c, _ = uncached.read([1, 2, 13])
    assert c == pytest.approx(a)
    assert uncached.cache_bytes == 0


def test_magma_missingness_qc_boundary_and_trait_selection(tmp_path):
    raw = np.tile([0.0, 1.0, 2.0, 1.0], (20, 1))
    raw[:, 0] = np.arange(20) % 3
    raw[:, 1] = np.arange(20) % 3
    raw[:, 2] = np.arange(20) % 3
    raw[:5, 0] = np.nan  # Exactly 25%: retained.
    raw[:6, 1] = np.nan  # More than 25%: removed.
    to_bed(tmp_path / "missing.bed", raw)
    reader = GenotypeReader(open_bed(tmp_path / "missing.bed"))
    g, keep, missing = reader.read_with_missing([2, 0, 1, 3])
    assert keep.tolist() == [True, True, False, False]
    assert missing.shape == g.shape == (20, 2)
    assert missing[0].tolist() == [False, True]
    _, whole_keep = reader.read([2, 0, 1, 3])
    assert whole_keep.tolist() == [True, True, True, False]
    _, _, complete = reader.read_with_missing([2])
    assert complete is None
    assert reader.cache_bytes <= reader.capacity


def test_magma_pairwise_ld_uses_individual_means_and_joint_count():
    from fastmagma.genotypes import magma_correlation

    raw = np.array(
        [[0.0, 2.0, 1.0], [1.0, 0.0, 0.0], [2.0, 1.0, 2.0], [0.0, np.nan, 1.0], [np.nan, 2.0, 0.0]]
    )
    g, keep = normalize_genotypes(raw)
    assert keep.all()
    expected = np.eye(3)
    means = np.nanmean(raw, axis=0)
    variances = np.nanvar(raw, axis=0)
    for i in range(3):
        for j in range(i):
            joint = np.isfinite(raw[:, i]) & np.isfinite(raw[:, j])
            covariance = np.mean((raw[joint, i] - means[i]) * (raw[joint, j] - means[j]))
            expected[i, j] = expected[j, i] = np.clip(
                covariance / np.sqrt(variances[i] * variances[j]), -1, 1
            )
    missing = np.isnan(raw)
    assert magma_correlation(g, missing) == pytest.approx(expected, abs=1e-14)
    assert magma_correlation(g[:, :1], missing[:, :1], g[:, 1:], missing[:, 1:]) == pytest.approx(
        expected[:1, 1:], abs=1e-14
    )


@pytest.mark.parametrize("missing_rate", [0, 0.002, 0.25])
def test_normalization_preserves_input_and_matches_explicit_imputation(missing_rate):
    raw = np.random.default_rng(41).integers(0, 3, (489, 37)).astype(float)
    raw[np.random.default_rng(42).random(raw.shape) < missing_rate] = np.nan
    raw[:, -1] = 1
    original = raw.copy()
    g, keep = normalize_genotypes(raw)
    np.testing.assert_array_equal(raw, original)
    means = np.nanmean(raw, axis=0)
    expected = np.where(np.isnan(raw), means, raw) - means
    lengths = np.sqrt(np.einsum("ij,ij->j", expected, expected))
    expected[:, keep] /= lengths[keep]
    np.testing.assert_allclose(g, expected, rtol=0, atol=1e-15)


def test_reader_mixed_blocks_order_duplicates_and_cache_ownership(tmp_path):
    raw = np.random.default_rng(8).integers(0, 3, (20, 30)).astype(float)
    raw[0, 15] = np.nan
    raw[:, 27] = 1
    to_bed(tmp_path / "mixed.bed", raw)
    reader = GenotypeReader(open_bed(tmp_path / "mixed.bed"), block_snps=10)
    indices = [21, 15, 2, 27, 15]
    g, keep, missing = reader.read_with_missing(indices)
    expected, expected_keep = normalize_genotypes(raw[:, indices])
    np.testing.assert_array_equal(keep, expected_keep)
    np.testing.assert_array_equal(g, expected[:, keep])
    np.testing.assert_array_equal(missing, np.isnan(raw[:, indices])[:, keep])
    assert reader.cache[0][2] is None
    assert reader.cache[2][2] is None
    g[:] = 99
    again, _, _ = reader.read_with_missing(indices)
    np.testing.assert_array_equal(again, expected[:, keep])
    assert reader.cache_bytes <= reader.capacity


def test_missing_calls_in_rejected_columns_do_not_leave_a_missing_mask(tmp_path):
    raw = np.random.default_rng(12).integers(0, 3, (20, 4)).astype(float)
    raw[:6, 1] = np.nan
    to_bed(tmp_path / "rejected.bed", raw)
    reader = GenotypeReader(open_bed(tmp_path / "rejected.bed"))
    g, keep, missing = reader.read_with_missing([1, 2])
    assert keep.tolist() == [False, True]
    assert g.shape == (20, 1)
    assert missing is None


@pytest.mark.parametrize("shape", [(489, 1), (489, 30), (80, 244)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_symmetric_gram_layout_matches_full_gram(shape, order):
    from scipy.linalg import eigvalsh

    g = np.array(np.random.default_rng(91).normal(size=shape), order=order)
    g -= g.mean(axis=0)
    g /= np.linalg.norm(g, axis=0)
    gram = g.T @ g if shape[1] <= shape[0] else g @ g.T
    expected = eigvalsh(gram, check_finite=False)
    tolerance = np.finfo(float).eps * max(shape) * max(expected[-1], 1) * 4
    expected = expected[expected > tolerance]
    actual = correlation_spectrum(g)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)


@pytest.mark.parametrize("model", ["magma", "whole"])
def test_single_cached_block_preserves_selection_and_result_ownership(tmp_path, model):
    raw = np.random.default_rng(83).integers(0, 3, (40, 20)).astype(float)
    raw[:12, 8] = np.nan
    raw[:, 4] = 1
    to_bed(tmp_path / "single.bed", raw)
    reader = GenotypeReader(open_bed(tmp_path / "single.bed"), block_snps=32)
    ids = [8, 2, 4, 2, 9]
    g, keep, missing = reader.read_with_missing(ids, model=model)
    expected, whole_keep = normalize_genotypes(raw[:, ids])
    expected_keep = (
        whole_keep & (np.isfinite(raw[:, ids]).sum(axis=0) >= 30)
        if model == "magma"
        else whole_keep
    )
    np.testing.assert_array_equal(keep, expected_keep)
    np.testing.assert_array_equal(g, expected[:, keep])
    if model == "magma":
        assert missing is None
    else:
        np.testing.assert_array_equal(missing, np.isnan(raw[:, ids])[:, keep])
    g[:] = 0
    again, _, _ = reader.read_with_missing(ids, model=model)
    np.testing.assert_array_equal(again, expected[:, keep])
