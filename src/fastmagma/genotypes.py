"""Bounded block caching and correlation spectra for PLINK reference panels."""

from collections import OrderedDict

import numpy as np
from scipy.linalg import eigvalsh


def normalize_genotypes(raw):
    """Mean-impute, center, and give each polymorphic column unit L2 norm."""
    g, keep, _, _ = _normalize_genotypes(raw)
    return g, keep


def _normalize_genotypes(raw):
    """Normalize once, retaining the sufficient information for both QC policies."""
    g = np.array(raw, dtype=np.float64, copy=True, order="F")
    valid = np.isfinite(g)
    if np.any(~valid & ~np.isnan(g)):
        raise ValueError("Genotype calls must be finite or NaN")
    count = valid.sum(axis=0)
    mean = np.divide(np.nansum(g, axis=0), count, out=np.zeros(g.shape[1]), where=count > 0)
    g -= mean
    g[~valid] = 0
    lengths = np.sqrt(np.einsum("ij,ij->j", g, g))
    keep = (count >= 2) & (lengths > 1e-12)
    np.divide(g, lengths, out=g, where=keep[None, :])
    g[:, ~keep] = 0
    variance = np.divide(lengths**2, count, out=np.zeros_like(lengths), where=count > 0)
    variance *= g.shape[0] / max(g.shape[0] - 1, 1)
    # --pval activates reduced reference-panel QC in MAGMA (25%, not 5%).
    magma_keep = keep & (count >= 0.75 * g.shape[0]) & (variance > 1e-8)
    return g, keep, ~valid, magma_keep


class GenotypeReader:
    """LRU cache of normalized contiguous SNP blocks, bounded by array bytes."""

    def __init__(self, bed, cache_mb=8, block_snps=256, threads=1):
        self.bed = bed
        self.capacity = int(cache_mb * 2**20)
        self.block_snps = block_snps
        self.threads = threads
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.block_reads = 0

    def _block(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        start = number * self.block_snps
        stop = min(start + self.block_snps, self.bed.sid_count)
        raw = self.bed.read(index=np.s_[:, start:stop], dtype="float64", num_threads=self.threads)
        block = _normalize_genotypes(raw)
        self.block_reads += 1
        size = sum(x.nbytes for x in block)
        while self.cache and self.cache_bytes + size > self.capacity:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= sum(x.nbytes for x in old)
        if size <= self.capacity:
            self.cache[number] = block
            self.cache_bytes += size
        return block

    def read(self, indices):
        """Return normalized genotypes with the original whole-gene QC policy."""
        g, keep, _ = self.read_with_missing(indices, model="whole")
        return g, keep

    def read_with_missing(self, indices, model="magma"):
        """Return retained genotypes, input-order retention mask, and missing calls.

        MAGMA summary-statistic analysis retains SNPs with at most 25% missing
        calls and sample variance above 1e-8. The missing mask has the same
        columns as returned genotypes; complete data return None so the LD
        calculation keeps its fast path.
        """
        if model not in {"magma", "whole"}:
            raise ValueError("Model must be 'magma' or 'whole'")
        indices = np.asarray(indices, dtype=np.int64)
        g = np.empty((self.bed.iid_count, len(indices)), dtype=np.float64, order="F")
        keep = np.empty(len(indices), dtype=bool)
        missing = np.empty(g.shape, dtype=bool, order="F")
        blocks = indices // self.block_snps
        for number in np.unique(blocks):
            select = np.flatnonzero(blocks == number)
            local = indices[select] - number * self.block_snps
            block, block_keep, block_missing, magma_keep = self._block(number)
            if model == "magma":
                block_keep = magma_keep
            g[:, select] = block[:, local]
            keep[select] = block_keep[local]
            missing[:, select] = block_missing[:, local]
        missing = missing[:, keep]
        return g[:, keep], keep, missing if missing.any() else None


def workspace_bytes(n_samples, n_snps, block_snps=256):
    """Conservative numerical-array estimate; excludes Python/I/O/BLAS overhead."""
    d = min(n_samples, n_snps)
    return (
        8 * (4 * n_samples * n_snps + 4 * d * d + 4 * n_samples * block_snps)
        + 3 * n_samples * n_snps
    )


def correlation_spectrum(g):
    """Nonzero eigenvalues of G.T @ G, choosing the smaller Gram matrix."""
    if g.ndim != 2 or min(g.shape) == 0 or not np.isfinite(g).all():
        raise ValueError("Genotypes must be a nonempty finite matrix")
    gram = g.T @ g if g.shape[1] <= g.shape[0] else g @ g.T
    lam = eigvalsh(gram, overwrite_a=True, check_finite=False)
    tolerance = np.finfo(float).eps * max(g.shape) * max(float(lam[-1]), 1.0) * 4
    if lam[0] < -tolerance:
        raise ArithmeticError("Correlation spectrum is not positive semidefinite")
    # Discard only the numerical null space, with a scale-aware threshold.
    lam = lam[lam > tolerance]
    if not len(lam):
        raise ArithmeticError("Correlation spectrum has zero rank")
    return lam


def magma_correlation(g, missing=None, other=None, other_missing=None, *, clip=True):
    """MAGMA LD from SNP-wise centered, unit-L2 columns and missing indicators.

    Pairwise cross-products use their joint observed count, while diagonal
    variances use each SNP's observed count. Thus the correction to imputed
    LD is sqrt(n_i * n_j) / n_ij, with SNP-wise rather than pair-wise means.
    This estimator need not be positive semidefinite. Brown moments use the
    unclipped estimator; only the eigenvalue test clips correlations.
    """
    right = g if other is None else other
    right_missing = missing if other is None else other_missing
    corr = g.T @ right
    if missing is not None or right_missing is not None:
        observed_left = np.ones(g.shape) if missing is None else (~missing).astype(float)
        observed_right = (
            observed_left
            if other is None
            else np.ones(right.shape)
            if right_missing is None
            else (~right_missing).astype(float)
        )
        joint = observed_left.T @ observed_right
        if np.any(joint <= 0):
            raise ValueError("Every SNP pair must have joint observed samples")
        corr *= (
            np.sqrt(observed_left.sum(axis=0)[:, None] * observed_right.sum(axis=0)[None, :])
            / joint
        )
    if clip:
        np.clip(corr, -1, 1, out=corr)
    if other is None:
        np.fill_diagonal(corr, 1)
    return corr
