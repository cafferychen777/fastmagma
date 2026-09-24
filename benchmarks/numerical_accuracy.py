"""Report approximation errors against an independent two-weight integral."""

import argparse
import json
import math
from pathlib import Path

from scipy.integrate import quad

from fastmagma.stats import gene_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for ratio in [0.001, 0.01, 0.1, 0.5, 0.9]:
        for q in [0.1, 1, 2, 5, 10, 30, 100]:
            reference, _ = quad(
                lambda theta: math.exp(
                    -q / (2 * (math.cos(theta) ** 2 + ratio * math.sin(theta) ** 2))
                ),
                0,
                math.pi / 2,
                epsabs=1e-100,
                epsrel=1e-11,
                limit=300,
            )
            reference *= 2 / math.pi
            result = gene_test(q, [1.0, ratio])
            rows.append(
                dict(
                    ratio=ratio,
                    q=q,
                    reference_p=reference,
                    computed_p=result.p,
                    method=result.method,
                    relative_error=abs(result.p / reference - 1),
                )
            )
    report = dict(
        scope="35 deterministic two-weight cases; not a universal error bound",
        max_relative_error=max(r["relative_error"] for r in rows),
        rows=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
