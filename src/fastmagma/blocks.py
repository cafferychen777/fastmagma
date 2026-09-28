"""Reusable MAGMA-compatible block models for normalized reference genotypes.

The block model follows an independent mathematical reconstruction: weighted
chi-square tests within balanced contiguous blocks, followed by Brown's method
using squared-LD estimates of dependence. Tail integration remains fastmagma's
accurate implementation rather than reproducing reference integration errors.
"""

from dataclasses import dataclass
import math

import numpy as np

from .genotypes import _symmetric_eigenvalues, correlation_spectrum, magma_correlation
from .stats import GeneResult, _gamma_logsf, _result, effective_parameters, gene_test


@dataclass(frozen=True)
class PreparedGene:
    """Trait-independent spectra and Brown parameters; no genotype copy retained."""

    block_sizes: tuple[int, ...]
    eigenvalues: tuple[np.ndarray, ...]
    nparam: int
    brown_scale: float = 1.0
    brown_df: float = 2.0

    @property
    def eigendecompositions(self):
        """Number of block spectra computed during preparation."""
        return len(self.eigenvalues)

    def test(self, q_array, p_array=None) -> GeneResult:
        """Test per-SNP chi-square statistics, preserving probabilities in log space."""
        q = np.asarray(q_array, dtype=np.float64)
        size = sum(self.block_sizes)
        if q.shape != (size,) or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError("SNP statistics must be a finite nonnegative vector matching the gene")
        p = None if p_array is None else np.asarray(p_array, dtype=np.float64)
        if p is not None and (
            p.shape != (size,) or not np.isfinite(p).all() or np.any((p <= 0) | (p > 1))
        ):
            raise ValueError("SNP probabilities must match the gene and lie in (0, 1]")
        results = []
        start = 0
        for count, spectrum in zip(self.block_sizes, self.eigenvalues):
            stop = start + count
            single = float(p[start]) if count == 1 and p is not None else None
            results.append(gene_test(float(q[start:stop].sum()), spectrum, single_snp_p=single))
            start = stop
        if len(results) == 1:
            return results[0]
        # -2 sum(log P) is stable even when an individual decimal P underflows.
        statistic = -2 * math.log(10) * math.fsum(result.log10p for result in results)
        method = "brown:" + "+".join(sorted({result.method for result in results}))
        return _result(_gamma_logsf(self.brown_df / 2, statistic / (2 * self.brown_scale)), method)


def _magma_parameters(spectrum, n_snps):
    """Moment-based MAGMA NPARAM, with nonnegative round-half-up rounding."""
    concentration = float(np.dot(spectrum, spectrum))
    fraction = np.clip(
        math.sqrt(n_snps / concentration) - 1 / math.sqrt(n_snps),
        0,
        1,
    )
    return math.floor(1 + float(fraction) * (n_snps - 1) + 0.5)


def _sample_indices(start, count, left):
    """Stratified reference-order samples used for between-block squared LD."""
    step = min(count // 50, 5) if count >= 100 else 1
    offset = (count - 1) % step if left else 0
    return np.arange(start + offset, start + count, step)


def prepare_gene(g, model="magma", missing=None) -> PreparedGene:
    """Prepare a gene from centered, unit-L2 genotype columns in reference order.

    MAGMA uses at most min(floor(n_samples / 2), 1000) SNPs per block,
    balances block sizes, and removes eigenvalues at or below 1e-4 times
    the average eigenvalue. The whole model removes only the numerical null
    space, without the MAGMA spectral cutoff.
    Temporary LD arrays are bounded by block size; no full-gene SNP-square
    matrix or block-square matrix is formed for the MAGMA model.
    """
    if model not in {"magma", "whole"}:
        raise ValueError("Model must be 'magma' or 'whole'")
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] == 0 or not np.isfinite(g).all():
        raise ValueError("Genotypes must be a finite matrix with at least two samples and one SNP")
    if missing is not None:
        missing = np.asarray(missing, dtype=bool)
        if missing.shape != g.shape:
            raise ValueError("Missing-call indicators must match the genotype matrix")
        if not missing.any():
            missing = None
    n_samples, n_snps = g.shape
    if model == "whole":
        spectrum = correlation_spectrum(g)
        return PreparedGene((n_snps,), (spectrum,), effective_parameters(spectrum))

    limit = min(n_samples // 2, 1000)
    n_blocks = (n_snps + limit - 1) // limit
    base, extra = divmod(n_snps, n_blocks)
    sizes = tuple(base + (index < extra) for index in range(n_blocks))
    starts = np.cumsum((0,) + sizes[:-1])
    spectra, parameters, variances = [], [], []
    for start, count in zip(starts, sizes):
        block = g[:, start : start + count]
        if missing is None:
            spectrum = correlation_spectrum(block)
            variances.append(float(np.dot(spectrum, spectrum)))
        else:
            corr = magma_correlation(block, missing[:, start : start + count], clip=False)
            variances.append(float(np.einsum("ij,ij->", corr, corr)))
            np.clip(corr, -1, 1, out=corr)
            spectrum = _symmetric_eigenvalues(corr)
        spectrum = spectrum[spectrum > 1e-4 * spectrum[spectrum > 0].sum() / count]
        spectra.append(spectrum)
        parameters.append(_magma_parameters(spectrum, count))
    if n_blocks == 1:
        return PreparedGene(sizes, tuple(spectra), parameters[0])

    # Accumulate the two required sums, avoiding quadratic block storage.
    correlation_sum = 0.0
    covariance_sum = 0.0
    for right in range(1, n_blocks):
        right_indices = _sample_indices(starts[right], sizes[right], left=False)
        right_g = g[:, right_indices]
        right_missing = None if missing is None else missing[:, right_indices]
        for left in range(right):
            left_indices = _sample_indices(starts[left], sizes[left], left=True)
            cross = magma_correlation(
                g[:, left_indices],
                None if missing is None else missing[:, left_indices],
                right_g,
                right_missing,
                clip=False,
            )
            covariance = (
                float(np.einsum("ij,ij->", cross, cross))
                * sizes[left]
                / len(left_indices)
                * sizes[right]
                / len(right_indices)
            )
            rho = float(np.clip(covariance / math.sqrt(variances[left] * variances[right]), 0, 1))
            correlation_sum += rho
            covariance_sum += rho * (3.25 + 0.75 * rho)
    expectation = 2 * n_blocks
    variance = 4 * n_blocks + 2 * covariance_sum
    scale = variance / (2 * expectation)
    degrees = 2 * expectation**2 / variance
    nparam = max(
        max(parameters),
        math.ceil(sum(parameters) * n_blocks / (n_blocks + 2 * correlation_sum)),
    )
    return PreparedGene(sizes, tuple(spectra), nparam, scale, degrees)
