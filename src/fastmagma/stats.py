"""Tail probabilities for positive weighted sums of independent chi-square(1)."""

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy import LowLevelCallable, integrate, optimize, special
from scipy.integrate import IntegrationWarning
from . import _native


class NumericalError(ArithmeticError):
    """A probability could not be evaluated reliably."""


@dataclass(frozen=True)
class GeneResult:
    log10p: float
    zstat: float
    method: str

    @property
    def p(self):
        """Decimal probability; natural floating-point underflow is allowed."""
        return math.exp(self.log10p * math.log(10.0))


def _spectrum(values):
    lam = np.asarray(values, dtype=np.float64)
    if lam.ndim != 1 or not lam.size or not np.all(np.isfinite(lam)):
        raise ValueError("Eigenvalues must be a nonempty finite vector")
    if np.any(lam < 0) or not np.any(lam > 0):
        raise ValueError("Eigenvalues must be nonnegative with positive total")
    return lam[lam > 0]


def _result(logp, method):
    if not np.isfinite(logp) or logp > 0:
        raise NumericalError(f"Invalid log probability from {method}: {logp}")
    return GeneResult(float(logp / math.log(10)), float(-special.ndtri_exp(logp)), method)


def _gamma_logsf(shape, x):
    """Log regularized upper gamma, including tails beyond linear underflow.

    Use the standard modified-Lentz continued fraction only when gammaincc
    underflows, where x is safely above the shape and convergence is rapid.
    """
    p = special.gammaincc(shape, x)
    if p > 0.5:
        return math.log1p(-special.gammainc(shape, x))
    if p > 0:
        return math.log(p)
    if x <= shape:
        raise NumericalError("Unexpected incomplete-gamma underflow")
    tiny = 1e-300
    b = x + 1 - shape
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10001):
        an = -i * (i - shape)
        b += 2
        d = an * d + b
        c = b + an / c
        if abs(d) < tiny:
            d = math.copysign(tiny, d)
        if abs(c) < tiny:
            c = math.copysign(tiny, c)
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 2e-15:
            return -x + shape * math.log(x) - special.gammaln(shape) + math.log(h)
    raise NumericalError("Incomplete-gamma continued fraction did not converge")


def _small_q_logsf(q, lam):
    """Integrate the Gaussian density over a small ellipsoid via its Taylor series.

    Transforming to the unit ball gives CDF = C * E[exp(-A)], where
    A = sum(q * U_i**2 / (2 * lam_i)) and U is uniform in that ball.
    Its squared coordinates, plus the unused radial mass, are Dirichlet
    with parameters (1/2, ..., 1/2, 1). The recurrence below evaluates
    E[A**m]/m! from power sums. A <= 0.1 makes the alternating terms
    decrease rapidly; the next term bounds the truncation error.
    Computing CDF first preserves log survival and Z when survival rounds to 1.
    """
    a = q / (2 * lam)
    if np.max(a) > 0.1:
        raise ValueError("Small-statistic expansion requires q <= 0.2 * min(eigenvalues)")
    shape = len(lam) / 2
    coefficients, power_sums, terms = [1.0], [0.0], [1.0]
    denominator = 1.0
    for order in range(1, 17):
        power_sums.append(float(np.sum(a**order)))
        coefficients.append(
            math.fsum(power_sums[j] * coefficients[order - j] for j in range(1, order + 1))
            / (2 * order)
        )
        denominator *= shape + order
        term = (-1) ** order * coefficients[order] / denominator
        terms.append(term)
        total = math.fsum(terms)
        if abs(term) <= np.finfo(float).eps * total:
            logcdf = (
                shape * (math.log(q) - math.log(2))
                - 0.5 * np.log(lam).sum()
                - special.gammaln(shape + 1)
                + math.log(total)
            )
            return math.log1p(-math.exp(logcdf))
    raise NumericalError("Small-statistic series did not converge")


def imhof_survival(q, lam):
    """Return a probability and QUADPACK absolute-error estimate.

    The estimate is a numerical diagnostic, not a mathematical error bound.
    Warnings are local to this call; importing this package changes no filters.
    """

    integrand = LowLevelCallable(
        _native.imhof_integrand(np.ascontiguousarray(lam, dtype=np.float64), q)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, err = integrate.quad(integrand, 0, np.inf, limit=300, epsabs=1e-10, epsrel=1e-9)
    return 0.5 + val / np.pi, err / np.pi


def tilted_logsf(q, lam):
    """Invert the exponentially tilted Laplace transform in log space.

    For 0 < t < 1/(2*max(lam)), survival equals exp(K(t)-t*q) times
    the integral of Re[exp(K(t+iu)-K(t)-iu*q)/(t+iu)] / pi. Scaling by
    sqrt(K''(t)) and separating sine/cosine terms lets QUADPACK's Fourier
    integrator handle oscillatory tails without subtracting two numbers near 1/2.
    """
    mean = lam.sum()
    if q > mean * 1.0001:
        t = optimize.brentq(
            lambda x: np.sum(lam / (1 - 2 * lam * x)) - q,
            0,
            np.nextafter(0.5 / lam.max(), 0.0),
            xtol=1e-14,
            rtol=1e-14,
        )
    else:
        # Any admissible positive tilt works; keep K(t) small below the mean.
        t = 0.1 / mean
    denominator = 1 - 2 * lam * t
    sd = math.sqrt(2 * np.sum((lam / denominator) ** 2))
    h = t * sd
    beta = 2 * lam / denominator / sd
    exponent = -0.5 * np.log(denominator).sum() - t * q
    cosine_callback, sine_callback = _native.tilted_integrands(beta, h)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        cosine, ce = integrate.quad(
            LowLevelCallable(cosine_callback),
            0,
            np.inf,
            weight="cos",
            wvar=q / sd,
            epsabs=1e-12,
            limlst=200,
            limit=200,
        )
        sine, se = integrate.quad(
            LowLevelCallable(sine_callback),
            0,
            np.inf,
            weight="sin",
            wvar=q / sd,
            epsabs=1e-12,
            limlst=200,
            limit=200,
        )
    value, error = cosine + sine, ce + se
    if not np.isfinite(value) or value <= 0 or not np.isfinite(error) or error > 1e-7 * value:
        raise NumericalError("Tilted integration did not meet its error criterion")
    return exponent + math.log(value / math.pi)


def _saddlepoint(q, lam):
    """Return (approximate log survival, log Chernoff upper bound).

    The mean uses the continuous Lugannani-Rice limit including skewness.
    """
    mean = lam.sum()
    variance = 2 * np.dot(lam, lam)
    if abs(q - mean) < 1e-5 * math.sqrt(variance):
        center = 0.5 - (8 * np.sum(lam**3)) / (6 * math.sqrt(2 * math.pi) * variance**1.5)
        return math.log(center), 0.0

    def kp(t):
        return np.sum(lam / (1 - 2 * lam * t))

    low = -1.0
    while kp(low) > q:
        low *= 2
        if not np.isfinite(low):
            raise NumericalError("Cannot bracket saddlepoint")
    high = np.nextafter(0.5 / lam.max(), 0.0)
    t = optimize.brentq(lambda x: kp(x) - q, low, high, xtol=1e-14, rtol=1e-14)
    k = -0.5 * np.log1p(-2 * lam * t).sum()
    kpp = 2 * np.sum((lam / (1 - 2 * lam * t)) ** 2)
    rate = t * q - k
    if rate <= 0:
        raise NumericalError("Invalid saddlepoint rate")
    w = math.copysign(math.sqrt(2 * rate), t)
    u = t * math.sqrt(kpp)
    if w > 0:
        correction = math.sqrt(math.pi / 2) * special.erfcx(w / math.sqrt(2)) + 1 / u - 1 / w
        if correction <= 0:
            raise NumericalError("Invalid saddlepoint correction")
        logp = -rate - 0.5 * math.log(2 * math.pi) + math.log(correction)
    else:
        p = special.ndtr(-w) + math.exp(-w * w / 2) / math.sqrt(2 * math.pi) * (1 / u - 1 / w)
        if not 0 < p <= 1:
            raise NumericalError("Invalid saddlepoint probability")
        logp = math.log(p)
    return logp, -rate if t > 0 else 0.0


def _ltz_logsf(q, lam):
    from scipy.stats import ncx2

    c1, c2, c3, c4 = (np.sum(lam**j) for j in range(1, 5))
    s1, s2 = c3 / c2**1.5, c4 / c2**2
    if s1**2 > s2:
        a = 1 / (s1 - math.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        df = a**2 - 2 * delta
    else:
        delta, df = 0.0, 1 / s1**2
    x = (q - c1) / math.sqrt(2 * c2) * math.sqrt(2 * (df + 2 * delta)) + df + delta
    return float(ncx2.logsf(x, df, delta))


def gene_test(q, eigenvalues, single_snp_p=None):
    """Evaluate a gene statistic; approximations are identified in the result.

    A small-statistic CDF expansion preserves probabilities near one.
    Tilted Fourier inversion avoids tail cancellation. If its estimated error
    is unacceptable, try ordinary Imhof in the bulk, then explicitly labeled
    saddlepoint/LTZ approximations. Numerical failure never becomes significance.
    """
    lam = _spectrum(eigenvalues)
    if not np.isfinite(q) or q < 0:
        raise ValueError("Statistic must be finite and nonnegative")
    if single_snp_p is not None:
        if not np.isfinite(single_snp_p) or not 0 < single_snp_p <= 1:
            raise ValueError("Single-SNP P must be finite and in (0, 1]")
        if len(lam) != 1:
            raise ValueError("Single-SNP shortcut requires one eigenvalue")
        return _result(math.log(single_snp_p), "exact1snp")
    if q == 0:
        return _result(0.0, "exact_zero")
    scale = lam.max()
    q, lam = q / scale, lam / scale
    if not np.isfinite(q):
        raise NumericalError("Scaled statistic overflow")
    if len(lam) == 1:
        return _result(_gamma_logsf(0.5, q / 2), "exact_rank1")
    if np.all(lam == lam[0]):
        return _result(_gamma_logsf(len(lam) / 2, q / 2), "exact_equal")

    if q <= 0.2 * lam.min():
        return _result(_small_q_logsf(q, lam), "small_q_series")

    try:
        return _result(tilted_logsf(q, lam), "tilted_imhof")
    except (ArithmeticError, ValueError, RuntimeError):
        pass

    saddle_logp, bound = None, 0.0
    try:
        saddle_logp, bound = _saddlepoint(q, lam)
        if not np.isfinite(saddle_logp) or saddle_logp > 0:
            saddle_logp = None
    except (ArithmeticError, ValueError, RuntimeError):
        pass
    if bound >= math.log(1e-6) or saddle_logp is None:
        try:
            p, err = imhof_survival(q, lam)
            if 1e-6 < p < 1 and np.isfinite(err) and 0 <= err < 1e-4 * p:
                return _result(math.log(p), "imhof")
        except (ArithmeticError, ValueError, RuntimeError):
            pass
    if saddle_logp is not None:
        return _result(saddle_logp, "saddlepoint")
    return _result(_ltz_logsf(q, lam), "ltz_fallback")


def effective_parameters(eigenvalues):
    """Effective-rank diagnostic, not MAGMA's verified NPARAM definition."""
    lam = _spectrum(eigenvalues)
    return max(1, int(round(lam.sum() ** 2 / np.dot(lam, lam))))
