#!/usr/bin/env python3
"""B+ tree versus learned index, across key distributions and scales.

    python scripts/bench_index.py                 # full sweep to 10M keys
    python scripts/bench_index.py --quick         # stop at 100k
    python scripts/bench_index.py --sizes 1000000

Lookup timings come from the C++ side (`measure_plan`), not from Python. A
pybind11 call costs on the order of a microsecond while a lookup costs single
-digit nanoseconds, so timing through the bridge would measure the bridge.
Build times are wall-clock from here, which is fine -- they are milliseconds
to seconds, where the bridge is noise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from hylis import ColumnIndex, IndexKind, IndexPlan, measure_plan
    from hylis import datasets as ds
except ImportError as exc:  # pragma: no cover - depends on the build having run
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "Build the C++ extensions first:\n"
        "    cmake --preset release && cmake --build build"
    )

DISTRIBUTIONS = ["sequential_gaps", "uniform", "lognormal", "clustered"]
RMI_MODEL_COUNTS = [64, 1024, 16384, 262144]


def require_optimised() -> None:
    import hylis._rmi as rmi

    if not getattr(rmi, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(rmi, '__build_type__', '?')} -- these timings "
            "would be ~5x off and meaningless.\n"
            "Rebuild first:\n"
            "    cmake --preset release && cmake --build build"
        )


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def timed_build(keys, values, plan):
    start = time.perf_counter()
    index = ColumnIndex.build_with(keys, values, plan)
    return index, time.perf_counter() - start


def bench_one(keys, values):
    """Measure a B+ tree and the best RMI configuration for one key set."""
    tree_plan = IndexPlan()
    tree_plan.kind = IndexKind.BPlusTree
    tree_index, tree_build = timed_build(keys, values, tree_plan)
    tree_measured = measure_plan(keys, values, tree_plan)

    best = None
    for models in RMI_MODEL_COUNTS:
        if models > len(keys):
            break
        plan = IndexPlan()
        plan.kind = IndexKind.RMI
        plan.rmi_models = models
        _, build_seconds = timed_build(keys, values, plan)
        measured = measure_plan(keys, values, plan)
        if best is None or measured.ns_per_lookup < best[0].ns_per_lookup:
            best = (measured, build_seconds)

    return {
        "tree": tree_measured,
        "tree_build": tree_build,
        "tree_height": None,
        "rmi": best[0] if best else None,
        "rmi_build": best[1] if best else 0.0,
        "index": tree_index,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", type=str, default="10000,100000,1000000,10000000",
                        help="comma-separated key counts")
    parser.add_argument("--quick", action="store_true", help="stop at 100k keys")
    parser.add_argument("--distributions", type=str, default=",".join(DISTRIBUTIONS))
    args = parser.parse_args(argv)

    require_optimised()

    sizes = [10_000, 100_000] if args.quick else [int(s) for s in args.sizes.split(",")]
    dists = [d.strip() for d in args.distributions.split(",")]

    print("B+ tree vs learned index (RMI)")
    print("lookup timings measured in C++; build times are wall clock\n")

    header = (f"{'distribution':<17}{'n':>11}  {'build ms':>18}  "
              f"{'ns/lookup':>17}  {'speedup':>8}  {'index size':>19}  {'err':>7}")
    print(header)
    print(f"{'':<17}{'':>11}  {'tree':>8} {'rmi':>9}  {'tree':>8} {'rmi':>8}  "
          f"{'':>8}  {'tree':>9} {'rmi':>9}  {'':>7}")
    print("-" * len(header))

    wins = {"tree": 0, "rmi": 0}
    for dist in dists:
        for n in sizes:
            data = ds.synthetic_keys(dist, n=n, seed=0)
            keys = data.keys.tolist()
            values = list(range(len(keys)))

            result = bench_one(keys, values)
            tree, rmi = result["tree"], result["rmi"]
            if rmi is None:
                continue

            speedup = tree.ns_per_lookup / rmi.ns_per_lookup
            wins["rmi" if speedup > 1.0 else "tree"] += 1

            print(f"{dist:<17}{n:>11,}  "
                  f"{result['tree_build']*1000:>8.0f} {result['rmi_build']*1000:>9.0f}  "
                  f"{tree.ns_per_lookup:>8.1f} {rmi.ns_per_lookup:>8.1f}  "
                  f"{speedup:>7.2f}x  "
                  f"{human_bytes(tree.index_bytes):>9} {human_bytes(rmi.index_bytes):>9}  "
                  f"{rmi.max_error:>7,}")
        print()

    print(f"rows where the RMI was faster: {wins['rmi']}, "
          f"where the tree was: {wins['tree']}")
    print()
    print("Reading this table:")
    print("  * 'err' is the RMI's worst prediction error, in records -- the")
    print("    width of the local search a lookup is left to do. It explains")
    print("    why a given row is fast or slow better than any other column.")
    print("  * The RMI's model overhead does not grow with n; the tree's")
    print("    internal nodes do. That gap is the memory argument.")
    print("  * These are read-only workloads. The RMI has no insert at all --")
    print("    it must be rebuilt wholesale -- so a row where it wins on")
    print("    lookups is not by itself an argument to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
