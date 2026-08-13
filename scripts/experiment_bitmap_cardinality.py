#!/usr/bin/env python3
"""E2: at what cardinality does a bitmap index stop beating a B+ tree?

    python scripts/experiment_bitmap_cardinality.py
    python scripts/experiment_bitmap_cardinality.py --quick
    python scripts/experiment_bitmap_cardinality.py --sizes 100000,1000000

A dictionary-encoded bitmap holds one n-bit map per distinct value, so its
memory is `d * n / 8` -- linear in the row count *and* in the distinct count.
At two values that is a quarter of a megabyte per million rows; at ten thousand
it is 1.25 GB, and a composite-key B+ tree, whose size does not depend on `d`
at all, wins outright. Somewhere between those the two cross.

`candidates_for()` needs that crossover to decide when to offer a bitmap, and
the number should be measured rather than asserted, which is what this does.

Four things are timed, because the families differ in *shape* and not just in
speed:

  * **eq**    one value. The bitmap does a dictionary lookup and returns a map;
              the tree descends and walks a run of leaves.
  * **range** a contiguous run of dictionary codes OR-ed, against a leaf walk.
  * **count** the asymmetry the whole family exists for. A bitmap answers by
              popcount and materialises nothing; the tree has to produce every
              row id and then measure the list.
  * **bytes** where the crossover actually comes from.

`bytes/row` against density is reported for its own reason: it is the evidence
that would justify compressed bitmaps (roaring, run-length), which this project
deliberately does not have. That decision should rest on measured density, not
on the general knowledge that compression exists.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    import numpy as np
except ImportError:
    raise SystemExit("needs numpy: pip install -r requirements.txt")

try:
    from hylis import (
        ColumnIndex,
        CompareOp,
        IndexKind,
        IndexPlan,
        KeyEncoding,
        LogicalType,
    )
    from hylis._rmi import measure_plan_typed
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

DISTINCTS = [2, 4, 16, 64, 256, 1024, 4096, 16384]


def require_optimised() -> None:
    import hylis._rmi as rmi

    if not getattr(rmi, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(rmi, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def make_column(n: int, distinct: int, skew: str, as_string: bool, seed: int = 0):
    """A column with a controlled distinct count, sorted by (value, row)."""
    rng = np.random.default_rng(seed)
    if skew == "zipf":
        # A skewed categorical column has both very dense and very sparse
        # bitmaps, which is what a real `category` looks like and what a
        # uniform draw hides.
        weights = 1.0 / np.arange(1, distinct + 1)
        weights /= weights.sum()
        codes = rng.choice(distinct, size=n, p=weights)
    else:
        codes = rng.integers(0, distinct, size=n)

    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    rows = order.astype(np.int64)
    if as_string:
        # Fixed width, so the comparison is about cardinality rather than about
        # how long the strings happen to be.
        values = [f"v{c:08d}" for c in codes]
    else:
        values = codes.astype(np.int64).tolist()
    return values, rows.tolist()


def plan_for(kind, encoding, type_):
    plan = IndexPlan()
    plan.kind = kind
    plan.type = type_
    plan.encoding = encoding
    plan.btree_order = 32
    return plan


def timed(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run_one(n: int, distinct: int, skew: str, as_string: bool, seed: int) -> dict:
    type_ = LogicalType.String if as_string else LogicalType.Int64
    values, rows = make_column(n, distinct, skew, as_string, seed)

    bitmap_plan = plan_for(IndexKind.Bitmap, KeyEncoding.Dictionary, type_)
    tree_plan = plan_for(IndexKind.BPlusTree, KeyEncoding.Composite, type_)

    start = time.perf_counter()
    bitmap = ColumnIndex.build_typed_with(type_, values, rows, bitmap_plan)
    bitmap_build = time.perf_counter() - start

    start = time.perf_counter()
    tree = ColumnIndex.build_typed_with(type_, values, rows, tree_plan)
    tree_build = time.perf_counter() - start

    # Point lookups are timed in C++: a pybind11 crossing costs ~1 us and the
    # lookup itself can be tens of nanoseconds.
    bitmap_eq = measure_plan_typed(type_, values, rows, bitmap_plan).ns_per_lookup
    tree_eq = measure_plan_typed(type_, values, rows, tree_plan).ns_per_lookup

    # Ranges and counts return or avoid whole row lists, so the crossing is
    # amortised and both arms pay it identically.
    rng = np.random.default_rng(seed + 1)
    probe_codes = rng.integers(0, distinct, size=min(20, distinct))
    probes = ([f"v{c:08d}" for c in probe_codes] if as_string
              else probe_codes.astype(np.int64).tolist())

    def ranges(column):
        def go():
            for p in probes:
                column.query(CompareOp.Lt, p)
        return go

    def counts(column):
        def go():
            for p in probes:
                column.count(CompareOp.Lt, p)
        return go

    bitmap_range = timed(ranges(bitmap)) / len(probes)
    tree_range = timed(ranges(tree)) / len(probes)
    bitmap_count = timed(counts(bitmap)) / len(probes)
    tree_count = timed(counts(tree)) / len(probes)

    bitmap_bytes = bitmap.plan().index_bytes
    tree_bytes = tree.plan().index_bytes
    return {
        "n": n,
        "distinct": distinct,
        "density": distinct / n,
        "eq_ratio": tree_eq / bitmap_eq if bitmap_eq else 0.0,
        "range_ratio": tree_range / bitmap_range if bitmap_range else 0.0,
        "count_ratio": tree_count / bitmap_count if bitmap_count else 0.0,
        "bitmap_count_us": bitmap_count * 1e6,
        "tree_count_us": tree_count * 1e6,
        "bytes_ratio": tree_bytes / bitmap_bytes if bitmap_bytes else 0.0,
        "bitmap_bytes": bitmap_bytes,
        "tree_bytes": tree_bytes,
        "bitmap_bytes_per_row": bitmap_bytes / n,
        "build_ratio": tree_build / bitmap_build if bitmap_build else 0.0,
    }


def print_table(rows) -> None:
    header = (f"{'distinct':>9}{'density':>10}{'eq':>9}{'range':>9}{'count':>10}"
              f"{'bytes':>9}{'b/row':>9}{'bitmap MB':>11}{'tree MB':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['distinct']:>9,}{r['density']:>10.5f}"
              f"{r['eq_ratio']:>8.2f}x{r['range_ratio']:>8.2f}x"
              f"{r['count_ratio']:>9.1f}x{r['bytes_ratio']:>8.2f}x"
              f"{r['bitmap_bytes_per_row']:>9.1f}"
              f"{r['bitmap_bytes']/1e6:>11.1f}{r['tree_bytes']/1e6:>10.1f}")
    print()
    print("  Every ratio is tree / bitmap, so above 1.00x the bitmap wins.")


def crossover_of(rows, field) -> str:
    """The first distinct count at which the bitmap stops winning."""
    for r in rows:
        if r[field] < 1.0:
            return f"{r['distinct']:,}"
    return "beyond the sweep"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="100000,1000000")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    require_optimised()
    sizes = [20000] if args.quick else [int(s) for s in args.sizes.split(",")]

    print("E2: where does a bitmap stop beating a composite-key B+ tree?")
    print()

    summary = []
    for n in sizes:
        for skew in ("uniform", "zipf"):
            for as_string in (False, True):
                label = f"n = {n:,}, {skew}, {'string' if as_string else 'int64'} keys"
                print("== " + label + " " + "=" * max(0, 50 - len(label)))
                # d * n / 8 bytes is the whole point of the experiment and
                # also the reason it cannot be run to the end of the sweep:
                # 16,384 values over a million rows is 2 GB of bitmaps. The
                # cutoff is stated rather than silently trimming the table,
                # because a missing row and a row that lost is not the same.
                budget = 512 * 1024 * 1024
                rows = []
                for d in DISTINCTS:
                    if d > max(2, n // 4):
                        break
                    if d * n / 8 > budget:
                        print(f"  (stopping at distinct = {d:,}: the bitmaps "
                              f"alone would be {d * n / 8 / 1e9:.1f} GB, which "
                              f"is the finding rather than a limit)")
                        break
                    rows.append(run_one(n, d, skew, as_string, args.seed))
                print_table(rows)
                summary.append((label, rows))
                print()

    print("=" * 76)
    print("Verdict")
    print("=" * 76)
    for label, rows in summary:
        print(f"  {label}")
        print(f"      memory crossover at distinct = {crossover_of(rows, 'bytes_ratio')}")
        print(f"      lookup crossover at distinct = {crossover_of(rows, 'eq_ratio')}")
        best_count = max(r["count_ratio"] for r in rows)
        print(f"      count() best case            = {best_count:.0f}x")
    print()
    print("  count() is the asymmetry the family exists for, and it does not")
    print("  have a crossover in the same sense: the bitmap answers by popcount")
    print("  whatever the cardinality, while the tree pays for every matching")
    print("  row id it must produce and then discard.")
    print()
    print("  bytes/row is the evidence for compressed bitmaps. Plain bitmaps")
    print("  cost d/8 bytes per row regardless of how the rows are distributed,")
    print("  so a sparse high-cardinality column is where roaring would pay --")
    print("  and it is also where the tree already wins, which is the argument")
    print("  for not adding compression at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
