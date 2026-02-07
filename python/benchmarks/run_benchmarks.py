#!/usr/bin/env python3
"""VectorForge Benchmark Runner

Convenient script to run benchmarks with common configurations.

Usage:
    python run_benchmarks.py                    # Run fast benchmarks
    python run_benchmarks.py --all              # Run all benchmarks (including slow)
    python run_benchmarks.py --save baseline    # Save baseline
    python run_benchmarks.py --compare baseline # Compare against baseline
    python run_benchmarks.py --search           # Run only search benchmarks
"""

import argparse
import subprocess
import sys


def run_command(cmd: list[str]) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd)
    print("-" * 80)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="VectorForge Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run fast benchmarks
  %(prog)s --all                    # Run all benchmarks
  %(prog)s --save baseline          # Save baseline results
  %(prog)s --compare baseline       # Compare with baseline
  %(prog)s --search                 # Only search benchmarks
  %(prog)s --indexing --all         # All indexing benchmarks
  %(prog)s --json results.json      # Save JSON output
        """,
    )

    # Test selection
    parser.add_argument(
        "--all", action="store_true", help="Include slow tests (large/xlarge scale)"
    )
    parser.add_argument(
        "--search", action="store_true", help="Run only search benchmarks"
    )
    parser.add_argument(
        "--indexing", action="store_true", help="Run only indexing benchmarks"
    )
    parser.add_argument(
        "--file-processing",
        action="store_true",
        help="Run only file processing benchmarks",
    )
    parser.add_argument(
        "--persistence", action="store_true", help="Run only persistence benchmarks"
    )
    parser.add_argument(
        "--scaling", action="store_true", help="Run only scaling benchmarks"
    )

    # Benchmark options
    parser.add_argument(
        "--save", metavar="NAME", help="Save benchmark results with given name"
    )
    parser.add_argument(
        "--compare", metavar="NAME", help="Compare results against saved benchmark"
    )
    parser.add_argument(
        "--json", metavar="FILE", help="Save results as JSON to specified file"
    )
    parser.add_argument(
        "--histogram", action="store_true", help="Generate histogram (requires pygal)"
    )

    # pytest options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-k", metavar="EXPR", help="Run tests matching expression")
    parser.add_argument(
        "--rounds", type=int, metavar="N", help="Minimum number of rounds (default: 5)"
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = ["pytest", "benchmarks/", "--benchmark-only"]

    # Select test files
    test_files = []
    if args.search:
        test_files.append("test_search_benchmarks.py")
    if args.indexing:
        test_files.append("test_indexing_benchmarks.py")
    if args.file_processing:
        test_files.append("test_file_processing_benchmarks.py")
    if args.persistence:
        test_files.append("test_persistence_benchmarks.py")
    if args.scaling:
        test_files.append("test_scaling_benchmarks.py")

    if test_files:
        cmd[1] = " ".join([f"benchmarks/{f}" for f in test_files])

    # Add markers for slow tests
    if not args.all:
        cmd.extend(["-m", "not slow"])

    # Add benchmark options
    if args.save:
        cmd.append(f"--benchmark-save={args.save}")

    if args.compare:
        cmd.append(f"--benchmark-compare={args.compare}")

    if args.json:
        cmd.append(f"--benchmark-json={args.json}")

    if args.histogram:
        cmd.append("--benchmark-histogram")

    # Add pytest options
    if args.verbose:
        cmd.append("-v")

    if args.k:
        cmd.extend(["-k", args.k])

    if args.rounds:
        cmd.append(f"--benchmark-min-rounds={args.rounds}")

    return run_command(cmd)


if __name__ == "__main__":
    sys.exit(main())
