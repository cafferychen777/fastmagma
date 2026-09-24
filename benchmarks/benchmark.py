"""Reproducible synthetic resource benchmark; emits JSON without plotting.

Run with: python benchmarks/benchmark.py --out results/benchmark.json
Each variant uses an isolated subprocess so peak RSS is comparable.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


def worker(variant):
    from fastmagma.genotypes import correlation_spectrum, normalize_genotypes

    rng = np.random.default_rng(981)
    g, keep = normalize_genotypes(rng.integers(0, 3, size=(150, 1500)))
    g = g[:, keep]
    timings = []
    with threadpool_limits(limits=1):
        for _ in range(3):
            start = time.perf_counter()
            if variant == "snp_gram":
                lam = np.linalg.eigvalsh(g.T @ g)
                lam = lam[lam > 1e-10]
            else:
                lam = correlation_spectrum(g)
            timings.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return dict(
        variant=variant,
        seconds=timings,
        median_seconds=float(np.median(timings)),
        peak_rss_bytes=rss,
        eigenvalues=lam.tolist(),
        samples=150,
        snps=1500,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", choices=["snp_gram", "dual_gram"])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return
    if args.out is None:
        parser.error("--out is required")
    results = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for variant in ["snp_gram", "dual_gram"]:
        run = subprocess.run(
            [sys.executable, __file__, "--worker", variant],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(run.stdout))
    assert np.allclose(results[0]["eigenvalues"], results[1]["eigenvalues"], atol=1e-10, rtol=1e-10)
    for result in results:
        del result["eigenvalues"]
    report = dict(
        platform=platform.platform(),
        python=sys.version,
        numpy=np.__version__,
        scope="Synthetic eigendecomposition microbenchmark, not whole-pipeline speedup",
        variants=results,
        speedup=results[0]["median_seconds"] / results[1]["median_seconds"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
