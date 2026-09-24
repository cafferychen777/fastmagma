"""Analytic, independent-quadrature, and failure-path numerical checks."""

import math
import warnings

import mpmath as mp
import numpy as np
import pytest
from scipy import integrate, special

from fastmagma import stats


@pytest.mark.parametrize("k", [1, 2, 5, 20, 100])
@pytest.mark.parametrize("q", [0.0, 0.01, 1, 5, 20, 100, 2000, 20000])
def test_equal_spectrum_high_precision(k, q):
    result = stats.gene_test(q, np.ones(k))
    with mp.workdps(60):
        expected = float(
            mp.log(mp.gammainc(mp.mpf(k) / 2, mp.mpf(q) / 2, mp.inf) / mp.gamma(mp.mpf(k) / 2))
        )
    assert result.log10p * math.log(10) == pytest.approx(expected, abs=2e-10, rel=3e-13)
    if q:
        assert special.log_ndtr(-result.zstat) == pytest.approx(expected, abs=2e-9, rel=3e-12)
    else:
        assert result.zstat == -np.inf


@pytest.mark.parametrize("p", [1.0, 0.5, 0.05, 1e-50, 1e-300, 1e-320, np.nextafter(0.0, 1.0)])
def test_single_snp_full_float_range(p):
    result = stats.gene_test(1, [1], single_snp_p=p)
    assert result.log10p == pytest.approx(math.log10(p), rel=2e-15)
    assert result.p == pytest.approx(p, rel=1e-12, abs=np.nextafter(0.0, 1.0))


def two_weight_reference(q, weights):
    # A 2D standard normal has uniform polar angle and chi2_2 radius squared.
    # Integrating its conditional survival is independent of Imhof/LR.
    a, b = weights
    value, _ = integrate.quad(
        lambda theta: math.exp(-q / (2 * (a * math.cos(theta) ** 2 + b * math.sin(theta) ** 2))),
        0,
        math.pi / 2,
        epsabs=1e-100,
        epsrel=1e-11,
        limit=300,
    )
    return value * 2 / math.pi


@pytest.mark.parametrize("weights", [[1.0, 0.9], [1.0, 0.1], [1.0, 0.001]])
@pytest.mark.parametrize("q", [0.1, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0])
def test_unequal_two_weight_reference(weights, q):
    result = stats.gene_test(q, weights)
    expected = two_weight_reference(q, weights)
    if result.method in {"imhof", "tilted_imhof"}:
        assert result.p == pytest.approx(expected, rel=2e-4, abs=1e-8)
    else:
        # LR is approximate: test a declared relative probability tolerance.
        assert result.p == pytest.approx(expected, rel=0.12)


@pytest.mark.parametrize("weights", [[1.0, 0.9], [1.0, 0.1], [1.0, 0.001], [1.0, 0.4, 0.2, 0.01]])
def test_monotonic_and_scale_invariant(weights):
    points = np.geomspace(0.01, 3000, 50)
    results = [stats.gene_test(q, weights).log10p for q in points]
    assert np.all(np.diff(results) < 0)
    for scale in [1e-10, 1e10]:
        assert stats.gene_test(10 * scale, np.asarray(weights) * scale).log10p == pytest.approx(
            stats.gene_test(10, weights).log10p, abs=1e-9
        )


def test_mean_limit_includes_skewness():
    lam = np.array([1.0, 0.9])
    p = math.exp(stats._saddlepoint(lam.sum(), lam)[0])
    assert p < 0.4
    assert p == pytest.approx(two_weight_reference(lam.sum(), lam), abs=0.004)


def test_failed_algorithms_raise(monkeypatch):
    monkeypatch.setattr(stats, "imhof_survival", lambda *a: (np.nan, np.nan))

    def fail(*args):
        raise stats.NumericalError("forced failure")

    monkeypatch.setattr(stats, "_saddlepoint", fail)
    monkeypatch.setattr(stats, "tilted_logsf", fail)
    monkeypatch.setattr(stats, "_ltz_logsf", lambda *a: -np.inf)
    with pytest.raises(stats.NumericalError):
        stats.gene_test(20, [1, 0.2])


def test_deep_tail_skips_imhof(monkeypatch):
    def fail(*args):
        pytest.fail("Deep-tail Chernoff gate did not skip integration")

    monkeypatch.setattr(stats, "imhof_survival", fail)
    result = stats.gene_test(10000, [1, 0.5, 0.1])
    assert np.isfinite(result.log10p) and result.p == 0
    assert result.method in {"tilted_imhof", "saddlepoint"}


@pytest.mark.parametrize(
    "q,lam", [(-1, [1]), (np.nan, [1]), (1, []), (1, [-1, 1]), (1, [0]), (1, [np.inf])]
)
def test_invalid_input(q, lam):
    with pytest.raises(ValueError):
        stats.gene_test(q, lam)


def test_no_global_warning_filter_changes():
    before = list(warnings.filters)
    stats.gene_test(2, [1, 0.5])
    assert warnings.filters == before


def test_ltz_is_labeled_and_validated(monkeypatch):
    monkeypatch.setattr(stats, "imhof_survival", lambda *a: (np.nan, np.nan))

    def fail(*args):
        raise stats.NumericalError("forced failure")

    monkeypatch.setattr(stats, "_saddlepoint", fail)
    monkeypatch.setattr(stats, "tilted_logsf", fail)
    result = stats.gene_test(5, [1, 0.5, 0.2])
    assert result.method == "ltz_fallback" and 0 < result.p < 1


def test_subnormal_p_display_has_no_artificial_cutoff():
    result = stats.gene_test(1, [1], single_snp_p=1e-310)
    assert result.p > 0
    assert result.log10p == pytest.approx(-310)


@pytest.mark.parametrize("multiplicities", [(1, 1), (2, 3), (1, 9), (10, 20)])
@pytest.mark.parametrize("ratio", [0.001, 0.1, 0.9])
@pytest.mark.parametrize("multiple", [0.5, 1.0, 3.0, 10.0])
def test_two_group_spectra_independent_beta_integral(multiplicities, ratio, multiple):
    # Radius and direction are independent: the direction's group mass is Beta.
    # This integral uses no characteristic functions or saddlepoint approximation.
    from scipy.stats import chi2

    n, m = multiplicities
    q = multiple * (n + ratio * m)
    value, _ = integrate.quad(
        lambda x: chi2.sf(q / (ratio + (1 - ratio) * x), n + m),
        0,
        1,
        weight="alg",
        wvar=(n / 2 - 1, m / 2 - 1),
        epsabs=1e-100,
        epsrel=1e-10,
        limit=300,
    )
    reference = value / special.beta(n / 2, m / 2)
    result = stats.gene_test(q, [1.0] * n + [ratio] * m)
    assert result.method in {"tilted_imhof", "imhof"}
    assert result.p == pytest.approx(reference, rel=2e-6, abs=1e-14)


def test_dense_center_monotonicity():
    lam = np.array([1.0, 0.1, 0.01])
    q = np.linspace(lam.sum() - 1e-4, lam.sum() + 1e-4, 31)
    values = [stats.gene_test(x, lam).log10p for x in q]
    assert np.all(np.diff(values) < 0)


@pytest.mark.parametrize("ratio", [0.001, 0.1])
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0, 5.0, 30.0, 100.0])
def test_integration_failure_fallback_against_reference(monkeypatch, ratio, q):
    def fail(*args):
        raise stats.NumericalError("forced tilted-integration failure")

    monkeypatch.setattr(stats, "tilted_logsf", fail)
    result = stats.gene_test(q, [1.0, ratio])
    reference = two_weight_reference(q, [1.0, ratio])
    assert result.method in {"imhof", "saddlepoint"}
    tolerance = 2e-4 if result.method == "imhof" else 0.12
    assert result.p == pytest.approx(reference, rel=tolerance)


@pytest.mark.parametrize("q", [1e-20, 4.7e-10, 1e-6, 0.01, 0.02])
@pytest.mark.parametrize("weights", [[1.0, 0.1], [1.0, 0.4, 0.1]])
def test_small_statistic_against_directional_cdf(q, weights):
    # Integrate the CDF directly: subtracting a near-one survival would discard
    # the information needed to verify negative Z statistics.
    if len(weights) == 2:
        cdf = (
            integrate.quad(
                lambda phi: (
                    -math.expm1(
                        -q
                        / (2 * (weights[0] * math.cos(phi) ** 2 + weights[1] * math.sin(phi) ** 2))
                    )
                ),
                0,
                math.pi / 2,
                epsabs=1e-100,
                epsrel=1e-12,
            )[0]
            * 2
            / math.pi
        )
    else:
        cdf = (
            integrate.dblquad(
                lambda u, phi: special.gammainc(
                    1.5,
                    q
                    / (
                        2
                        * (
                            (1 - u * u)
                            * (weights[0] * math.cos(phi) ** 2 + weights[1] * math.sin(phi) ** 2)
                            + weights[2] * u * u
                        )
                    ),
                ),
                0,
                math.pi / 2,
                lambda _: 0,
                lambda _: 1,
                epsabs=1e-100,
                epsrel=1e-11,
            )[0]
            * 2
            / math.pi
        )
    result = stats.gene_test(q, weights)
    assert result.method == "small_q_series"
    assert result.log10p * math.log(10) == pytest.approx(math.log1p(-cdf), rel=2e-13, abs=0)
    assert result.zstat == pytest.approx(special.ndtri(cdf), abs=2e-13)


@pytest.mark.parametrize("rank", [1, 2, 3, 10])
def test_equal_spectrum_near_one_keeps_log_probability(rank):
    q = 1e-10
    result = stats.gene_test(q, np.ones(rank))
    with mp.workdps(100):
        cdf = mp.gammainc(mp.mpf(rank) / 2, 0, mp.mpf(q) / 2) / mp.gamma(mp.mpf(rank) / 2)
        expected = float(mp.log1p(-cdf))
    assert result.log10p * math.log(10) == pytest.approx(expected, rel=2e-13, abs=0)
    assert np.isfinite(result.zstat)


def test_small_statistic_switch_is_continuous_and_monotonic():
    lam = [1.0, 0.4, 0.1]
    q = np.array([0.02 * (1 - 1e-6), 0.02, 0.02 * (1 + 1e-6)])
    values = [stats.gene_test(x, lam).log10p for x in q]
    assert np.all(np.diff(values) < 0)
    assert np.diff(values)[0] == pytest.approx(np.diff(values)[1], rel=1e-5)


@pytest.mark.parametrize("multiplicities", [(2, 3), (10, 20)])
def test_small_statistic_higher_rank_against_beta_cdf(multiplicities):
    n, m = multiplicities
    q, ratio = 0.01, 0.1
    value = integrate.quad(
        lambda x: special.gammainc((n + m) / 2, q / (2 * (ratio + (1 - ratio) * x))),
        0,
        1,
        weight="alg",
        wvar=(n / 2 - 1, m / 2 - 1),
        epsabs=1e-100,
        epsrel=1e-11,
    )[0]
    cdf = value / special.beta(n / 2, m / 2)
    result = stats.gene_test(q, [1.0] * n + [ratio] * m)
    assert result.log10p * math.log(10) == pytest.approx(math.log1p(-cdf), rel=1e-12, abs=0)
