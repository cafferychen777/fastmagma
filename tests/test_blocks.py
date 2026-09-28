"""Analytical block limits, dependence, spectral cutoffs, and stable tails."""

import math

import numpy as np
import pytest
from scipy.stats import chi2

from magma_py.blocks import prepare_gene
from magma_py.genotypes import normalize_genotypes
from magma_py.stats import effective_parameters, gene_test


def test_independent_blocks_reduce_to_fisher():
    g = np.eye(12)[:, :8]
    prepared = prepare_gene(g)
    q = np.arange(1, 9, dtype=float)
    probabilities = [chi2.sf(q[:4].sum(), 4), chi2.sf(q[4:].sum(), 4)]
    expected = chi2.sf(-2 * np.log(probabilities).sum(), 4)
    assert prepared.block_sizes == (4, 4)
    assert prepared.brown_scale == 1
    assert prepared.brown_df == 4
    assert prepared.nparam == 6  # Each four-SNP block has MAGMA NPARAM 3.
    assert prepared.test(q).p == pytest.approx(expected, rel=1e-13)


def test_identical_blocks_do_not_double_evidence():
    block = np.eye(12)[:, :4]
    prepared = prepare_gene(np.column_stack([block, block]))
    q = np.tile([1.0, 3.0, 2.0, 4.0], 2)
    assert prepared.brown_scale == 2
    assert prepared.brown_df == 2
    assert prepared.nparam == 3
    assert prepared.test(q).p == pytest.approx(chi2.sf(10, 4), rel=1e-13)


@pytest.mark.parametrize(
    "n,k,sizes",
    [
        (10, 5, (5,)),
        (10, 6, (3, 3)),
        (10, 11, (4, 4, 3)),
        (489, 244, (244,)),
        (489, 245, (123, 122)),
    ],
)
def test_balanced_block_boundary(n, k, sizes):
    g, keep = normalize_genotypes(np.random.default_rng(40).normal(size=(n, k)))
    assert keep.all()
    prepared = prepare_gene(g)
    assert prepared.block_sizes == sizes
    assert prepared.eigendecompositions == len(sizes)
    whole = prepare_gene(g, model="whole")
    assert whole.block_sizes == (k,)
    assert whole.eigendecompositions == 1
    assert whole.nparam == effective_parameters(whole.eigenvalues[0])
    q = np.arange(1, k + 1, dtype=float)
    assert whole.test(q) == gene_test(float(q.sum()), whole.eigenvalues[0])


def test_magma_truncates_boundary_eigenvalue_but_whole_retains_it(monkeypatch):
    monkeypatch.setattr(
        "magma_py.blocks.correlation_spectrum", lambda g: np.array([0.0001, 1.9999])
    )
    g = np.eye(6)[:, :2]
    assert prepare_gene(g).eigenvalues[0] == pytest.approx([1.9999])
    assert prepare_gene(g, "whole").eigenvalues[0] == pytest.approx([0.0001, 1.9999])


def test_maximum_block_size_is_one_thousand(monkeypatch):
    # Avoid unnecessary eigendecomposition while exercising the actual partition.
    monkeypatch.setattr("magma_py.blocks.correlation_spectrum", lambda g: np.ones(g.shape[1]))
    g = np.zeros((2002, 1001))
    prepared = prepare_gene(g)
    assert prepared.block_sizes == (501, 500)


def test_sampled_squared_ld_matches_independent_dense_formula():
    rng = np.random.default_rng(413)
    raw = rng.normal(size=(240, 232))
    raw[:, 116:] += raw[:, :116] * 0.7
    g, _ = normalize_genotypes(raw)
    prepared = prepare_gene(g)
    r = np.corrcoef(raw.T)
    left = np.arange(1, 116, 2)
    right = np.arange(116, 232, 2)
    covariance = (r[np.ix_(left, right)] ** 2).sum() * 4
    rho = covariance / math.sqrt((r[:116, :116] ** 2).sum() * (r[116:, 116:] ** 2).sum())
    variance = 8 + 2 * rho * (3.25 + 0.75 * rho)
    assert prepared.brown_scale == pytest.approx(variance / 8, rel=1e-13)
    assert prepared.brown_df == pytest.approx(32 / variance, rel=1e-13)


def test_brown_survives_component_probability_underflow():
    prepared = prepare_gene(np.eye(12)[:, :8])
    result = prepared.test(np.full(8, 2000.0))
    # chi-square(4) has exact log survival -x/2 + log(1+x/2).
    component_logp = -4000 + math.log1p(4000)
    half_statistic = -2 * component_logp
    expected_logp = -half_statistic + math.log1p(half_statistic)
    assert result.log10p == pytest.approx(expected_logp / math.log(10), rel=1e-13)
    assert result.p == 0
    assert np.isfinite(result.zstat)


def test_single_snp_preserves_input_probability():
    prepared = prepare_gene(np.array([[-1.0], [1.0]]) / math.sqrt(2))
    result = prepared.test([3.0], [0.02])
    assert prepared.nparam == 1
    assert result.p == pytest.approx(0.02)
    assert result.method == "exact1snp"


def test_zero_statistic_combination():
    result = prepare_gene(np.eye(12)[:, :8]).test(np.zeros(8))
    assert result.p == 1
    assert result.zstat == -np.inf


@pytest.mark.parametrize(
    "q,p", [([1], None), ([-1, 1], None), ([1, np.nan], None), ([1, 2], [0, 1])]
)
def test_invalid_statistics_fail(q, p):
    with pytest.raises(ValueError):
        prepare_gene(np.eye(6)[:, :2]).test(q, p)


@pytest.mark.parametrize(
    "g,model",
    [
        (np.ones((1, 2)), "magma"),
        (np.empty((4, 0)), "magma"),
        (np.eye(3), "unknown"),
        (np.full((3, 2), np.nan), "magma"),
    ],
)
def test_invalid_preparation_fails(g, model):
    with pytest.raises(ValueError):
        prepare_gene(g, model)


def test_missing_ld_discards_negative_eigenvalues_without_psd_projection():
    from magma_py.genotypes import magma_correlation

    rng = np.random.default_rng(343)
    raw = np.tile(rng.integers(0, 3, 40), (3, 1)).T.astype(float)
    for column in range(3):
        raw[rng.choice(40, 2, replace=False), column] = np.nan
    g, _ = normalize_genotypes(raw)
    missing = np.isnan(raw)
    spectrum = np.linalg.eigvalsh(magma_correlation(g, missing))
    assert spectrum[0] < -1e-5
    expected = spectrum[spectrum > 1e-4 * spectrum[spectrum > 0].sum() / 3]
    prepared = prepare_gene(g, missing=missing)
    assert prepared.eigenvalues[0] == pytest.approx(expected, abs=1e-13)
    q = np.array([2.0, 3.0, 4.0])
    assert prepared.test(q).p == pytest.approx(gene_test(q.sum(), expected).p, rel=1e-12)
    assert prepare_gene(g, "whole", missing=missing).test(q) == prepare_gene(g, "whole").test(q)


def test_missing_block_moments_use_unclipped_ld():
    rng = np.random.default_rng(71)
    raw = np.tile(rng.integers(0, 3, 40), (22, 1)).T.astype(float)
    for column in range(raw.shape[1]):
        raw[rng.choice(40, 2, replace=False), column] = np.nan
    missing = np.isnan(raw)
    g, _ = normalize_genotypes(raw)
    # Independent definition in raw units, retaining values above one in moments.
    means, variance = np.nanmean(raw, axis=0), np.nanvar(raw, axis=0)
    r = np.eye(22)
    for i in range(22):
        for j in range(i):
            joint = ~missing[:, i] & ~missing[:, j]
            r[i, j] = r[j, i] = np.mean(
                (raw[joint, i] - means[i]) * (raw[joint, j] - means[j])
            ) / np.sqrt(variance[i] * variance[j])
    assert np.max(r) > 1
    rho = np.clip(
        (r[:11, 11:] ** 2).sum() / math.sqrt((r[:11, :11] ** 2).sum() * (r[11:, 11:] ** 2).sum()),
        0,
        1,
    )
    expected_scale = (8 + 2 * rho * (3.25 + 0.75 * rho)) / 8
    prepared = prepare_gene(g, missing=missing)
    assert prepared.block_sizes == (11, 11)
    assert prepared.brown_scale == pytest.approx(expected_scale, rel=1e-13)


def test_missing_mask_shape_is_checked():
    with pytest.raises(ValueError, match="Missing-call"):
        prepare_gene(np.eye(5), missing=np.zeros((5, 4), dtype=bool))
