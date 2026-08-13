#!/usr/bin/env python3
"""E4: how should two predicates be intersected?

    python scripts/experiment_conjunction.py
    python scripts/experiment_conjunction.py --quick
    python scripts/experiment_conjunction.py -n 200000

The query planner took a single predicate deliberately, on the grounds that two
are set intersection over row-id lists and introduce no new decision. Bitmaps
made that false, and this measures by how much.

  **sorted merge** costs `O(m1 + m2)` -- it has to produce both row-id lists
  and walk them. Cheap when the predicates are selective, and it is the only
  option when either column is an ordered index.

  **bitmap AND** costs `O(n / 64)` words *whatever matches*. Flat in
  selectivity, which is the shape a merge cannot have: at 90% selectivity the
  merge is walking most of the table twice and the AND is doing the same 1,563
  word operations it does at 0.1%.

So the two curves must cross, and where they cross is the finding.

A confound worth naming: which strategy runs is decided by which families the
columns got, so the arms differ in their columns as well as their algorithm.
That is not avoidable and it is not really a flaw -- "which algorithm" and
"which column type" are the same question here, because a tree cannot produce
a bit set and a bitmap over a high-cardinality column is not a structure anyone
would build. The selectivities are matched across arms so the comparison is at
least about the same amount of matching.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
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
        ColumnDef,
        IndexKind,
        LogicalType,
        PredOp,
        Record,
        RecordStore,
        Schema,
        Table,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

SELECTIVITIES = [0.001, 0.01, 0.1, 0.5, 0.9]
BUCKETS = 1000  # so a selectivity of 0.001 is reachable on a bitmap column


def require_optimised() -> None:
    import hylis._table as t

    if not getattr(t, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(t, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def schema_of() -> Schema:
    return Schema([
        # Low cardinality: these get bitmaps.
        ColumnDef("bucket_a", LogicalType.Int64),
        ColumnDef("bucket_b", LogicalType.Int64),
        # Unique: these get an ordered index.
        ColumnDef("wide_a", LogicalType.Int64),
        ColumnDef("wide_b", LogicalType.Int64),
    ])


def build(root: Path, n: int, columns, seed: int = 0):
    """A table whose four columns hold the same information twice.

    `bucket_*` is a value in [0, BUCKETS) and `wide_*` is that same value
    scaled into a unique integer, so `bucket_a < k` and `wide_a < k * n` select
    exactly the same rows. Matched selectivity, different index families.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, BUCKETS, size=n)
    b = rng.integers(0, BUCKETS, size=n)

    store = RecordStore(str(root))
    table = Table(store, schema_of())
    rows = []
    for i in range(n):
        rows.append(Record(i, {
            "bucket_a": str(int(a[i])),
            "bucket_b": str(int(b[i])),
            # Unique by construction: bucket in the high digits, row in the low.
            "wide_a": str(int(a[i]) * n + i),
            "wide_b": str(int(b[i]) * n + i),
        }))
    table.put_batch(rows)
    for column, kind in columns:
        # Pinned rather than chosen. choose_index times lookups, and a bitmap
        # loses those from about 64 distinct values upward because it decodes
        # a whole matching set where a tree walks a leaf run -- so asking it
        # nicely for a bitmap over 1000 buckets returns a tree, and the arm
        # this experiment exists to measure would never run.
        table.create_index_as(column, kind)
    return table


def timed(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run_arm(table, columns, n, sel_a, sel_b, wide):
    cut_a = int(sel_a * BUCKETS)
    cut_b = int(sel_b * BUCKETS)
    if wide:
        predicates = [(columns[0], PredOp.Lt, cut_a * n),
                      (columns[1], PredOp.Lt, cut_b * n)]
    else:
        predicates = [(columns[0], PredOp.Lt, cut_a),
                      (columns[1], PredOp.Lt, cut_b)]

    keys, trace = table.select_all(predicates)
    elapsed = timed(lambda: table.select_all(predicates))
    strategy = "bitmap AND" if "bitmap AND" in trace.reason else "sorted merge"
    return elapsed, len(keys), strategy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.quick:
        args.n = 20000

    require_optimised()
    print("E4: which intersection strategy, and where do they cross?")
    print(f"    {args.n:,} rows, {BUCKETS} buckets, best of five passes")
    print()

    root = Path(tempfile.mkdtemp(prefix="hylis_e4_"))
    try:
        bitmaps = build(root / "bitmap", args.n,
                        [("bucket_a", IndexKind.Bitmap),
                         ("bucket_b", IndexKind.Bitmap)], args.seed)
        trees = build(root / "tree", args.n,
                      [("wide_a", IndexKind.BPlusTree),
                       ("wide_b", IndexKind.BPlusTree)], args.seed)

        print(f"  bucket_a got {bitmaps.info('bucket_a').kind}, "
              f"wide_a got {trees.info('wide_a').kind}")
        print()

        header = (f"{'sel a':>8}{'sel b':>8}{'matched':>10}{'bitmap AND':>13}"
                  f"{'sorted merge':>15}{'ratio':>9}")
        print(header)
        print("-" * len(header))

        crossed_at = None
        previous_winner = None
        for sel in SELECTIVITIES:
            and_time, matched, and_strategy = run_arm(
                bitmaps, ["bucket_a", "bucket_b"], args.n, sel, sel, wide=False)
            merge_time, merge_matched, merge_strategy = run_arm(
                trees, ["wide_a", "wide_b"], args.n, sel, sel, wide=True)

            assert and_strategy == "bitmap AND", and_strategy
            assert merge_strategy == "sorted merge", merge_strategy
            if matched != merge_matched:
                print(f"  MISMATCH: {matched} vs {merge_matched} rows at "
                      f"selectivity {sel}")

            ratio = merge_time / and_time if and_time else 0.0
            print(f"{sel:>8.3f}{sel:>8.3f}{matched:>10,}"
                  f"{and_time*1e6:>12.1f}u{merge_time*1e6:>14.1f}u"
                  f"{ratio:>8.2f}x")

            winner = "and" if ratio > 1.0 else "merge"
            if previous_winner is not None and winner != previous_winner:
                crossed_at = sel
            previous_winner = winner

        print()
        print("  ratio is merge / AND, so above 1.00x the bitmap AND wins.")
        print()
        print("=" * 70)
        print("Verdict")
        print("=" * 70)
        if crossed_at is not None:
            print(f"  The two cross near selectivity {crossed_at:g}: below it")
            print("  the merge has little to walk and wins, above it the AND's")
            print("  flat cost takes over.")
        else:
            print(f"  No crossover in this range -- {previous_winner} won every")
            print("  row, so there is nothing to dispatch on and Table should")
            print("  keep preferring the bitmap AND whenever it is available.")
        print()
        print("  What the table does not show, and should be said: the AND is")
        print("  only reachable when every predicate is a bitmap column over")
        print("  the same rows. That is a property of the schema, not of the")
        print("  query, so Table cannot choose between these two strategies at")
        print("  runtime the way a planner chooses between scan and graph --")
        print("  it takes whichever one the columns make possible.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
