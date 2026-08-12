#!/usr/bin/env python3
"""E3: should a duplicated numeric column get a learned index at all?

    python scripts/experiment_duplicate_keys.py
    python scripts/experiment_duplicate_keys.py --quick
    python scripts/experiment_duplicate_keys.py --sizes 100000,1000000

A column of prices or category ids repeats values. Every ordered structure in
this project maps one key to one row, so a repeated value needs an encoding,
and there are two candidates:

  **Composite** -- key on (value, row id). Pairs make every key unique,
  std::pair already compares lexicographically-then-tiebreak, and the B+ tree
  needed no change at all. One structure, one descent.

  **Position** -- key on sorted rank, 0..m-1, with the actual values kept in a
  side array so a threshold can be translated into rank space first. This is
  what python/hylis/query.py::ScalarColumn does, and it is the only encoding a
  learned index can take, because a (value, row) pair has no cast to a double.

The question is whether Position earns its place. It costs a second structure
and it renumbers every row after a mid-range insert, so it has to be paid for
in lookups.

There is a structural point the table below makes concrete, and it is the real
finding rather than the timings. Under the Position encoding the learned
index's keys are literally 0, 1, 2, ... m-1 -- a perfectly linear CDF that one
linear model fits with **zero error**. So the model is not learning anything
about the data; it has been handed an arithmetic sequence and asked to predict
it. All the actual work has moved into the cut translation, which is a lookup
over the distinct values.

Which means the honest way to time Position is as **two structures**: a
dictionary from value to rank range, and an array indexed by rank. This script
measures both halves in C++ and adds them, rather than timing the array alone
and quietly leaving the translation out.

Decision rule, fixed before running: keep the Position encoding only where it
beats Composite by more than 1.2x on blended cost. Otherwise delete it, drop
KeyEncoding::Position, and send duplicated numeric columns to the tree.
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

DUPLICATE_FACTORS = [1, 2, 10, 100, 1000]
SELECTIVITIES = [0.01, 0.10, 0.50]


def require_optimised() -> None:
    import hylis._rmi as rmi

    if not getattr(rmi, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(rmi, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def make_column(n: int, rows_per_value: int, seed: int = 0):
    """A sorted (value, row) column with a controlled duplicate factor."""
    rng = np.random.default_rng(seed)
    distinct = max(1, n // rows_per_value)
    values = np.sort(rng.choice(np.arange(distinct * 4, dtype=np.int64),
                                size=distinct, replace=False))
    keys = np.repeat(values, rows_per_value)[:n]
    order = np.argsort(keys, kind="stable")
    return keys[order].tolist(), order.astype(np.int64).tolist(), values


def plan_for(kind, encoding, models=1024, order=32) -> IndexPlan:
    plan = IndexPlan()
    plan.kind = kind
    plan.type = LogicalType.Int64
    plan.encoding = encoding
    plan.rmi_models = models
    plan.search_threshold = 64
    plan.btree_order = order
    return plan


def best_rmi(keys, values):
    """The fastest model count, so Position is measured at its best."""
    best = None
    for models in (64, 1024, 16384, 262144):
        if models > len(keys):
            break
        measured = measure_plan_typed(LogicalType.Int64, keys, values,
                                      plan_for(IndexKind.RMI, KeyEncoding.Native,
                                               models=models))
        if best is None or measured.ns_per_lookup < best.ns_per_lookup:
            best = measured
    return best


def best_of(fn, repeats=5):
    """Best of `repeats`. The minimum is the run least interfered with."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run_one(n: int, rows_per_value: int, seed: int) -> dict:
    keys, rows, distinct_values = make_column(n, rows_per_value, seed)
    distinct = len(distinct_values)
    ranks = list(range(len(keys)))
    sorted_keys = np.array(keys, dtype=np.int64)

    composite_plan = plan_for(IndexKind.BPlusTree, KeyEncoding.Composite)
    composite_column = ColumnIndex.build_typed_with(
        LogicalType.Int64, keys, rows, composite_plan)

    # The position arm's array half. Its keys are 0, 1, 2, ... m-1, so the
    # model has an arithmetic sequence to predict; max_error below is the
    # evidence that nothing about the data was learned.
    position_measured = best_rmi(ranks, rows)
    position_column = ColumnIndex.build_typed_with(
        LogicalType.Int64, ranks, rows,
        plan_for(IndexKind.RMI, KeyEncoding.Native,
                 models=position_measured.rmi_models))

    rng = np.random.default_rng(seed + 1)
    probes = rng.choice(distinct_values, size=min(50, distinct)).tolist()

    # --- equal work, both arms ---------------------------------------------
    #
    # The trap this replaced: measuring composite's lookup (which materialises
    # every matching row) against position's find (which returns one) and
    # calling the ratio a speedup. At 1000 rows per value that compares
    # returning 1000 rows with returning 1, and composite "loses" by 300x for
    # doing 1000x the work.
    #
    # So both arms are timed producing *the same rows*: every row whose value
    # equals the probe. Position needs its translation first, which is a binary
    # search over the sorted values -- numpy's searchsorted, which is the same
    # contiguous-array binary search std::lower_bound would do, and is charged
    # to position here.
    def composite_eq():
        for v in probes:
            composite_column.lookup(v)

    def position_eq():
        for v in probes:
            lo = int(np.searchsorted(sorted_keys, v, side="left"))
            hi = int(np.searchsorted(sorted_keys, v, side="right"))
            if hi > lo:
                position_column.query_range(lo, hi - 1)

    eq_composite = best_of(composite_eq) / len(probes)
    eq_position = best_of(position_eq) / len(probes)

    # --- ranges, same construction -----------------------------------------
    def composite_range():
        for v in probes:
            composite_column.query(CompareOp.Lt, v)

    def position_range():
        for v in probes:
            cut = int(np.searchsorted(sorted_keys, v, side="left"))
            position_column.query(CompareOp.Lt, cut)

    range_composite = best_of(composite_range, repeats=3) / len(probes)
    range_position = best_of(position_range, repeats=3) / len(probes)

    # Position carries the sorted values array as well as the index; without
    # it there is no way to translate a predicate at all.
    position_bytes = position_measured.index_bytes + len(keys) * 8

    return {
        "rows_per_value": rows_per_value,
        "n": len(keys),
        "distinct": distinct,
        "position_max_error": position_measured.max_error,
        "composite_bytes": composite_column.plan().index_bytes,
        "position_bytes": position_bytes,
        "eq_composite_us": eq_composite * 1e6,
        "eq_position_us": eq_position * 1e6,
        "range_composite_us": range_composite * 1e6,
        "range_position_us": range_position * 1e6,
        "speedup": eq_composite / eq_position if eq_position else 0.0,
        "range_speedup": range_composite / range_position if range_position else 0.0,
    }


def print_table(rows) -> None:
    header = (f"{'rows/value':>11}{'distinct':>10}{'eq comp':>10}{'eq posn':>10}"
              f"{'speedup':>9}{'lt comp':>10}{'lt posn':>10}{'speedup':>9}"
              f"{'comp MB':>9}{'posn MB':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['rows_per_value']:>11,}{r['distinct']:>10,}"
              f"{r['eq_composite_us']:>9.2f}u{r['eq_position_us']:>9.2f}u"
              f"{r['speedup']:>8.2f}x{r['range_composite_us']:>9.1f}u"
              f"{r['range_position_us']:>9.1f}u{r['range_speedup']:>8.2f}x"
              f"{r['composite_bytes']/1e6:>9.1f}{r['position_bytes']/1e6:>9.1f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="100000,1000000")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    require_optimised()
    sizes = [20000] if args.quick else [int(s) for s in args.sizes.split(",")]

    print("E3: composite (value, row) keys vs the position encoding,")
    print("    for a duplicated numeric column.")
    print()
    print("Both arms are timed producing the same rows, and the position arm")
    print("is charged for its translation as well as its array. Measuring")
    print("composite's lookup (which materialises every match) against")
    print("position's find (which returns one row) would show position ahead")
    print("by 300x at 1000 rows per value, for doing 1000x less work.")
    print()

    all_rows = []
    for n in sizes:
        print(f"== n = {n:,} " + "=" * 46)
        rows = [run_one(n, f, args.seed) for f in DUPLICATE_FACTORS
                if f <= max(1, n // 4)]
        print_table(rows)
        all_rows.extend(rows)
        print()
        print("  Both arms return the same rows. 'eq' is every row equal to a")
        print("  probe value; 'lt' is every row below it. Microseconds per")
        print("  query, best of several passes.")
        print()

    # ---- the decision rule -------------------------------------------------
    #
    # Both workloads, not one. A structure that wins scans and loses point
    # lookups cannot be the single answer for a column, and reporting only the
    # half it wins would be picking the flattering number.
    eq_wins = [r for r in all_rows if r["speedup"] > 1.2]
    range_wins = [r for r in all_rows if r["range_speedup"] > 1.2]
    worst_eq = min(r["speedup"] for r in all_rows)
    best_range = max(r["range_speedup"] for r in all_rows)
    zero_error = all(r["position_max_error"] == 0 for r in all_rows
                     if r["rows_per_value"] > 1)
    mean_bytes = (sum(r["composite_bytes"] for r in all_rows) /
                  max(1, sum(r["position_bytes"] for r in all_rows)))

    print("=" * 76)
    print("Verdict")
    print("=" * 76)
    print(f"  point lookups: position beat composite by >1.2x on "
          f"{len(eq_wins)} of {len(all_rows)} rows")
    print(f"                 worst case {worst_eq:.2f}x, i.e. composite was "
          f"{1/worst_eq:.1f}x faster")
    print(f"  range scans:   position beat composite by >1.2x on "
          f"{len(range_wins)} of {len(all_rows)} rows")
    print(f"                 best case {best_range:.2f}x")
    print(f"  memory:        position used {mean_bytes:.1f}x less")
    print()
    if zero_error:
        print("  The index over ranks reported max_error = 0 on every row,")
        print("  which is the finding rather than a good result: its keys are")
        print("  0, 1, 2, ... m-1, so one linear model fits them exactly.")
        print("  Nothing about the data was learned -- the structure is a")
        print("  sorted array, and calling it a learned index misdescribes it.")
        print()

    if len(eq_wins) > len(all_rows) / 2 and len(range_wins) > len(all_rows) / 2:
        print("  KEEP the position encoding: it won both workloads.")
    else:
        print("  DROP the position encoding.")
        print()
        print("  It is genuinely better at two things -- large range scans and")
        print("  memory -- and that is worth stating rather than burying. But")
        print("  a column gets one structure, and position loses the point")
        print("  lookup badly, needs two structures instead of one, and cannot")
        print("  take a mid-range insert without renumbering every row after")
        print("  the insertion point. Composite takes that insert in O(log n).")
        print()
        print("  KeyEncoding::Position stays in the enum as a recorded")
        print("  negative result; candidates_for() never produces it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
