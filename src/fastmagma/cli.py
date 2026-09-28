"""Command-line interface for gene analysis and merging."""

import argparse
from importlib.metadata import version
import logging
import sys


def _positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return n


def _nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer")
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-trait SNP-wise mean gene analysis")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('fastmagma')}")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Analyze one autosome across traits")
    run.add_argument("--chr", type=int, choices=range(1, 23), required=True)
    run.add_argument("--bfile-prefix", required=True)
    run.add_argument("--annot", required=True)
    run.add_argument("--pval-dir", required=True)
    run.add_argument("--traits", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument(
        "--model",
        choices=("magma", "whole"),
        default="magma",
        help="Statistical model: MAGMA-compatible gene blocks (default) or whole-gene test",
    )
    run.add_argument("--threads", type=_positive, default=1)
    run.add_argument("--chunk-rows", type=_positive, default=32768)
    run.add_argument(
        "--cache-mb",
        type=_nonnegative,
        default=8,
        help="Normalized genotype cache capacity in MiB",
    )
    run.add_argument(
        "--workspace-mb",
        type=_positive,
        default=512,
        help="Numerical-array workspace estimate limit in MiB; not a process RSS cap",
    )
    run.add_argument(
        "--block-snps",
        type=_positive,
        default=256,
        help="Genotype I/O block size; does not change the statistical gene blocks",
    )
    run.add_argument("--max-gene-snps", type=_positive, default=100000)
    run.add_argument("--temp-dir", help="Directory for disk-backed trait arrays")
    run.add_argument("--overwrite", action="store_true")
    merge = commands.add_parser("merge", help="Validate and merge chromosome results")
    merge.add_argument("--traits", required=True)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--chrs", required=True)
    merge.add_argument("--allow-missing", action="store_true")
    merge.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Keep help/version fast and free of scientific-library initialization.
    from .pipeline import locked_outputs, merge_chromosomes, run_chromosome
    from .io import parse_chromosomes

    try:
        chromosomes = [args.chr] if args.command == "run" else parse_chromosomes(args.chrs)
        with locked_outputs(args.out_dir, chromosomes, merging=args.command == "merge"):
            (run_chromosome if args.command == "run" else merge_chromosomes)(args)
    except (ValueError, OSError, ArithmeticError, MemoryError, RuntimeError) as exc:
        print(f"fastmagma: error: {exc}", file=sys.stderr)
        return 1
    return 0
