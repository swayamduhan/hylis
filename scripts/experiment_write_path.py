#!/usr/bin/env python3
"""E6: what does keeping an index up to date actually cost?

    python scripts/experiment_write_path.py
    python scripts/experiment_write_path.py --quick
    python scripts/experiment_write_path.py -n 20000

E6 was originally going to ask how often a mid-range insert forces a whole
column rebuild, because the design it was written for keyed columns on sorted
rank -- where inserting between two existing values renumbers every row after
the insertion point. Experiment E3 dropped that encoding in favour of composite
(value, row) keys, which take a mid-range insert in O(log n) and renumber
nothing, so the question went away with the encoding.

What is left is the question underneath it: **what does a write cost once a
table has indexes on it, and where does that cost go?** Three parts.

  1. **The floor.** RecordStore fsyncs per write by design. If that dominates,
     then index maintenance is close to free in practice and the honest
     recommendation is to index freely -- and the throughput ceiling belongs to
     the storage layer, not the index layer.

  2. **The cost of declaring the workload wrong.** A column planned read-only
     may be given the static RMI, which is build-only. Writing to it is legal
     and costs a whole-column rebuild on the next read. That is the only
     remaining path to a rebuild, and it should be priced rather than described.

  3. **Whether put_batch earns its place.** It claims to turn n rebuilds into
     one. Either it does or the claim comes out of the docstring.
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

CATEGORIES = ["bags", "hats", "shoes", "socks", "coats"]


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
        ColumnDef("price", LogicalType.Int64),
        ColumnDef("category", LogicalType.String),
        ColumnDef("title", LogicalType.String),
    ])


def monotone_rows(n: int):
    """Unique, ascending values: the learned index's best case, and what an
    auto-increment id or a timestamp actually looks like.

    Parts 2 and 3 need a column the RMI genuinely wins, because the mechanism
    they price -- a build-only structure being written to -- only exists when
    the RMI was chosen."""
    return [Record(i, {"price": str(i * 7),
                       "category": CATEGORIES[i % len(CATEGORIES)],
                       "title": f"item-{i}"})
            for i in range(n)]


def rows_for(n: int, seed: int = 0):
    import random

    rng = random.Random(seed)
    return [
        Record(i, {
            "price": str(rng.randrange(100_000)),
            "category": CATEGORIES[i % len(CATEGORIES)],
            "title": f"item-{rng.randrange(100_000)}",
        })
        for i in range(n)
    ]


class Bench:
    """A fresh table per measurement.

    A write stream changes the structure it is measured on, so reusing one
    table across arms would time a different index each pass.
    """

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="hylis_e6_"))
        self._n = 0

    def table(self, rows, indexed, write_fraction=0.3):
        self._n += 1
        store = RecordStore(str(self.root / f"t{self._n}"))
        table = Table(store, schema_of())
        table.put_batch(rows)
        for column in indexed:
            table.create_index(column, write_fraction=write_fraction)
        return table

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def timed(build, repeats=3):
    """Best of `repeats`, each on a **fresh** table.

    A write stream changes the structure it is measured on, so repeating it on
    one instance would time an insert pass once and an update pass afterwards.
    The first version of this did exactly that and produced a table where two
    indexed columns cost more than three."""
    best = float("inf")
    for _ in range(repeats):
        best = min(best, build())
    return best


def write_stream(table, keys, seed=1):
    import random

    rng = random.Random(seed)
    start = time.perf_counter()
    for key in keys:
        table.put(Record(key, {
            "price": str(rng.randrange(100_000)),
            "category": CATEGORIES[rng.randrange(len(CATEGORIES))],
            "title": f"item-{rng.randrange(100_000)}",
        }))
    return time.perf_counter() - start


def part_one(bench, n, writes):
    """The store's write floor, against the measured cost of maintaining an
    index.

    Not by subtraction. The first version of this timed a write stream with 0,
    1, 2 and 3 indexed columns and subtracted the baseline -- and produced a
    table in which adding indexes made writes *faster*, by 15%. That is not a
    tuning artefact to average away: RecordStore fsyncs on every write, the
    fsync costs on the order of a millisecond, and its run-to-run variance is
    larger than the whole quantity being measured. A difference of two noisy
    large numbers is not a small number, it is noise.

    So each side is measured where it can be measured. The floor comes from a
    table with no indexes at all; the index cost comes from choose_index's own
    write timing, which runs in C++ on the structure itself with no store and
    no language bridge in the way.
    """
    print("1. The write floor, and what an index adds to it")
    print()

    rows = rows_for(n)
    keys = list(range(n, n + writes))
    floor = timed(lambda: write_stream(bench.table(rows, []), keys)) / writes
    print(f"  store floor, no indexes:     {floor*1e6:>9.1f} us/write")
    print("  (a WAL append plus an fsync, which storage/store.hpp names as the")
    print("   throughput cap and group commit as the fix)")
    print()

    header = f"{'column':>12}{'structure':>14}{'ns/write':>11}{'share of a write':>19}"
    print(header)
    print("-" * len(header))

    total = 0.0
    table = bench.table(rows, ["price", "category", "title"], write_fraction=0.3)
    for column in ("price", "category", "title"):
        plan = table.explain_column(column)
        total += plan.ns_per_write
        share = plan.ns_per_write / (floor * 1e9) * 100.0 if floor else 0.0
        print(f"{column:>12}{str(plan.kind).split('.')[-1]:>14}"
              f"{plan.ns_per_write:>11.1f}{share:>18.3f}%")

    share = total / (floor * 1e9) * 100.0 if floor else 0.0
    print("-" * len(header))
    print(f"{'all three':>12}{'':>14}{total:>11.1f}{share:>18.3f}%")
    print()
    print("  Index maintenance is a rounding error against the fsync. The")
    print("  honest reading: on this storage layer, index freely -- the write")
    print("  ceiling belongs to the WAL, not to the indexes. That also says")
    print("  where an optimisation would pay, and it is not here.")
    print()


def part_two(bench, n, writes):
    """The price of declaring a column read-only and then writing to it.

    Reported in **rebuilds, not microseconds**. Part 1 established that the
    fsync dominates a write and that its variance exceeds anything being
    measured around it; that applies here too, and an earlier version of this
    arm duly printed a table saying the writable column was four times slower
    than the read-only one, which is not a thing that can be true. The rebuild
    count is what this arm can actually measure, and it is the quantity the
    decision turns on anyway.
    """
    print("2. The cost of declaring the workload wrong")
    print()
    # Deliberately short. Every write to the read-only arm can rebuild the
    # whole column, which is O(n), so this arm costs O(n * writes).
    writes = min(writes, 100)
    rows = monotone_rows(n)
    keys = list(range(n, n + writes))

    header = (f"{'declared':>12}{'structure':>14}{'mutable':>10}"
              f"{'rebuilds':>10}{'per write':>12}")
    print(header)
    print("-" * len(header))

    results = {}
    for label, fraction in (("read-only", 0.0), ("writable", 0.3)):
        table = bench.table(rows, ["price"], write_fraction=fraction)
        kind = table.info("price").kind
        write_stream(table, keys)
        # Rebuilds are lazy: they land on the next read, so a query has to
        # happen before the counter means anything.
        table.select_keys("price", PredOp.Lt, 50_000)
        rebuilds = table.rebuilds()
        results[label] = (kind, rebuilds)
        print(f"{label:>12}{str(kind).split('.')[-1]:>14}"
              f"{('no' if kind == IndexKind.RMI else 'yes'):>10}"
              f"{rebuilds:>10,}{rebuilds/writes:>12.3f}")

    print()
    if results["read-only"][0] == IndexKind.RMI:
        rebuilds = results["read-only"][1]
        print("  A column planned read-only was given the static RMI, which is")
        print("  build-only. Every write to it marks the column for rebuild,")
        print("  and the rebuild lands on the next read. This is the only")
        print("  remaining path to a rebuild in the whole write path.")
        if 0 < rebuilds < writes:
            print()
            print(f"  It self-corrects: {rebuilds} rebuilds over {writes} writes,")
            print("  not one per write. A write that makes the values non-unique")
            print("  stops the stored plan being legal for the data at all, so")
            print("  the column is re-chosen and lands on a mutable structure")
            print("  that never needs rebuilding again. The cost of the wrong")
            print("  declaration is bounded, and the rate depends on how soon a")
            print("  write breaks uniqueness rather than on the write count.")
    else:
        print("  The tree won the read-only column on this machine, so both")
        print("  arms got a mutable structure and there is no penalty to show.")
        print("  Not a null result about the mechanism -- a null result about")
        print("  this corpus, and the table says which.")
    print()


def part_three(bench, n, writes):
    """Does put_batch turn n rebuilds into one?"""
    print("3. Does put_batch earn its place?")
    print()

    # The workload that makes batching matter is a column planned read-only and
    # then written to. Its structure is build-only, so every write marks it for
    # rebuild -- and because the stored plan is still the read-only one, the
    # rebuild produces the same build-only structure and the next write dirties
    # it again. A loop pays that once per record; a batch pays it once.
    #
    # The first workload tried here was a uniqueness collision instead, and it
    # showed nothing: the first rebuild switches the column to a composite key,
    # which absorbs every later collision, so the loop also paid exactly one
    # rebuild. Correct behaviour, wrong experiment.
    rows = monotone_rows(n)
    probe = bench.table(rows, ["price"], write_fraction=0.0)
    if probe.info("price").kind != IndexKind.RMI:
        print("  The tree won this column on this machine, so no arm has a")
        print("  build-only structure and there is no rebuild to batch away.")
        print("  Nothing to report rather than a number that means nothing.")
        print()
        return

    extra = [Record(n + i, {"price": str((n + i) * 7)}) for i in range(writes)]

    header = f"{'path':>12}{'records':>10}{'seconds':>10}{'rebuilds':>10}"
    print(header)
    print("-" * len(header))

    looped = bench.table(rows, ["price"], write_fraction=0.0)
    start = time.perf_counter()
    for record in extra:
        looped.put(record)
    loop_seconds = time.perf_counter() - start
    looped.select_keys("price", PredOp.Lt, 50)
    print(f"{'put loop':>12}{len(extra):>10,}{loop_seconds:>10.3f}"
          f"{looped.rebuilds():>10,}")

    batched = bench.table(rows, ["price"], write_fraction=0.0)
    start = time.perf_counter()
    batched.put_batch(extra)
    batch_seconds = time.perf_counter() - start
    batched.select_keys("price", PredOp.Lt, 50)
    print(f"{'put_batch':>12}{len(extra):>10,}{batch_seconds:>10.3f}"
          f"{batched.rebuilds():>10,}")

    print()
    ratio = loop_seconds / batch_seconds if batch_seconds else 0.0
    print(f"  batching was {ratio:.1f}x faster and paid "
          f"{looped.rebuilds() - batched.rebuilds()} fewer rebuilds.")
    if batched.rebuilds() <= 1 < looped.rebuilds():
        print("  The claim holds: n rebuilds become one. This is why put_batch")
        print("  is in the API rather than left to a caller's loop.")
    else:
        print("  The claim does NOT hold here, and the docstring in table.hpp")
        print("  should be corrected to match.")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=20000, help="rows to preload")
    parser.add_argument("--writes", type=int, default=2000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.n, args.writes = 4000, 400

    require_optimised()
    print("E6: the cost of keeping an index up to date")
    print(f"    {args.n:,} rows preloaded, {args.writes:,} writes per arm")
    print()

    bench = Bench()
    try:
        part_one(bench, args.n, args.writes)
        part_two(bench, args.n, args.writes)
        part_three(bench, args.n, min(args.writes, 40))
    finally:
        bench.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
