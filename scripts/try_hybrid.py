#!/usr/bin/env python3
"""Interactive playground for the whole system: table, indexes, planner.

    python scripts/try_hybrid.py --demo            # scripted walkthrough
    python scripts/try_hybrid.py --load 20000
    python scripts/try_hybrid.py --sift            # SIFT10K + synthetic attributes

Type `help` at the prompt for the command list.

The other two playgrounds each show one module. This one shows the join: a
durable record store, a per-column index chosen by measurement, a predicate,
a vector search over the survivors, and the planner's own account of why it
chose what it chose. `recover` closes the store and reopens it, which is the
only place the whole stack is visible at once -- the rows come back from the
write-ahead log, and the index *decisions* come back from the catalog without
being re-measured.

Where the vectors live: in the table, as a declared column, stored outside the
record. A 128-float embedding base64s to ~700 bytes, so putting one in the JSON
write-ahead log would make the log the dominant cost of the system; the floats
go to the vector index instead and reach disk as a `.fvecs` sidecar at
checkpoint. The consequence is worth knowing before you type `put_vector`:
**embeddings are not write-ahead logged**, so `recover` shows them coming back
from the last save rather than from the log.
"""

from __future__ import annotations

import argparse
import os
import shlex
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
        Metric,
        PlanKind,
        PredOp,
        Record,
        RecordStore,
        Schema,
        Table,
        VectorPlan,
        VectorStructure,
    )
    from hylis import datasets as ds
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "Build the C++ extensions first:\n"
        "    cmake --preset release && cmake --build build"
    )


OPS = {
    "eq": PredOp.Eq, "lt": PredOp.Lt, "le": PredOp.Le,
    "gt": PredOp.Gt, "ge": PredOp.Ge,
    "prefix": PredOp.Prefix, "contains": PredOp.Contains, "isnull": PredOp.IsNull,
}
KINDS = {"btree": IndexKind.BPlusTree, "rmi": IndexKind.RMI,
         "bitmap": IndexKind.Bitmap, "dynamic": IndexKind.DynamicRMI}

HELP = """
  data
    load <n> [dim]           synthetic shop rows + clustered embeddings
    sift                     SIFT10K vectors with synthetic attributes
    put <key> col=val ...    insert or update one record (durable)
    del <key>                delete one record
    get <key>                fetch one record from the store
    checkpoint               snapshot, truncate the WAL, save catalog+vectors
    recover                  close and reopen: WAL replay, catalog replay,
                             and the embeddings back from the .fvecs sidecar

  index
    index <col> [write_frac] build or re-tune, choosing by measurement
    as <col> <kind>          force a structure: btree rmi bitmap dynamic
    drop <col>               remove an index; the column still answers by scan
    columns                  every column, what indexes it, and what that cost
    catalog                  the persisted decisions

  query
    where <col> <op> <val>   structured only; op: eq lt le gt ge prefix
                             contains isnull
    and <col> <op> <val> ... two or more predicates, comma-separated
    count <col> <op> <val>   how many match, without producing them
    knn <k> [qi]             vector only, graph and brute force side by side
    like <key> [k]           more-like-this, seeded by a row already stored
    hybrid <col> <op> <val> <k> [qi]    predicate and vector together
    explain <col> <op> <val> <k>        the plan and why, no vector work
    plans <col> <op> <val> <k> [qi]     every legal plan, timed and compared
    calibrate                measure this corpus's crossover and adopt it
    ef <n>                   beam width; 0 for the index default

  vectors
    vectors                  the vector column: rows, orphans, memory
    compact                  reclaim rows deletion could not (a full rebuild)

  verify
    check                    every query path against a brute-force scan
    stats / help / quit
"""

CATEGORIES = ["bags", "hats", "shoes", "socks", "coats"]
TITLES = ["nike air", "nike zoom", "adidas run", "puma go", "nikon lens",
          "reebok classic"]


def schema_of(dim: int) -> Schema:
    return Schema([
        ColumnDef("price", LogicalType.Int64),
        ColumnDef("category", LogicalType.String),
        ColumnDef("title", LogicalType.String),
        ColumnDef("in_stock", LogicalType.Bool),
        ColumnDef("created_at", LogicalType.Timestamp),
        # A price band. Int64 and low-cardinality, which is what makes the
        # bitmap plan reachable from here: the vector filter works in row ids,
        # so a bit set can only be used as a mask when the column is int64
        # *and* its positions are the vector column's rows.
        ColumnDef("band", LogicalType.Int64),
        # The embedding. Declared in the schema, stored outside the record.
        ColumnDef("image", LogicalType.Vector, dim),
    ])


class Playground:
    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="hylis_demo_"))
        self.store: RecordStore | None = None
        self.table: Table | None = None
        self.base: np.ndarray | None = None
        self.queries: np.ndarray | None = None
        self.ef = 0
        self.source = "(empty)"

    # -- loading -----------------------------------------------------------

    def load(self, n: int, dim: int = 32, sift: bool = False) -> None:
        if sift:
            try:
                vectors = ds.load_sift("siftsmall")
            except FileNotFoundError as exc:
                print(f"  {exc}")
                return
            self.source = "SIFT10K + synthetic attributes"
        else:
            vectors = ds.random_vectors(n=n, dim=dim, n_queries=50, seed=0,
                                        n_clusters=max(2, n // 100))
            self.source = f"synthetic {n}x{dim}"

        self.base = vectors.base
        self.queries = vectors.queries
        n = vectors.n

        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = RecordStore(str(self.root / "shop"))
        self.table = Table(self.store, schema_of(vectors.dim))

        rng = np.random.default_rng(0)
        rows = []
        for i in range(n):
            price = int(rng.integers(0, 50_000))
            rows.append(Record(i, {
                "price": str(price),
                "band": str(price // 10_000),
                "category": CATEGORIES[i % len(CATEGORIES)],
                "title": TITLES[int(rng.integers(0, len(TITLES)))],
                "in_stock": "true" if i % 4 else "false",
                "created_at": f"2026-{1 + i % 12:02d}-{1 + i % 28:02d}T09:00:00Z",
            }))
        start = time.perf_counter()
        result = self.table.put_batch(rows)
        load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        self.table.create_vector_index(
            "image", VectorPlan(structure=VectorStructure.Graph, metric=Metric.L2,
                                M=16, ef_construction=200))
        self.table.put_vectors("image", list(range(n)), self.base)
        vector_seconds = time.perf_counter() - start

        info = self.table.vector_info("image")
        print(f"  {result.rows_created:,} records written and fsynced in "
              f"{load_seconds:.1f}s")
        print(f"  {info.rows:,} x {info.dim}-d embeddings attached in "
              f"{vector_seconds*1000:.0f} ms "
              f"({info.memory_bytes/1e6:.1f} MB, exact index and graph)")
        print("  the embeddings are a schema column stored outside the record:")
        print("  a 128-float vector base64s to ~700 bytes, which would make the")
        print("  JSON write-ahead log the dominant cost of the whole system")

    def _require(self) -> bool:
        if self.table is None:
            print("  nothing loaded; try `load 5000`")
            return False
        return True

    # -- index -------------------------------------------------------------

    def create_index(self, column: str, write_fraction: float = 0.0) -> None:
        if not self._require():
            return
        start = time.perf_counter()
        info = self.table.create_index(column, write_fraction=write_fraction)
        elapsed = time.perf_counter() - start
        self._describe_one(info)
        print(f"    chosen by building and timing every legal candidate "
              f"({elapsed*1000:.0f} ms)")

    def create_index_as(self, column: str, kind_name: str) -> None:
        if not self._require():
            return
        kind = KINDS.get(kind_name)
        if kind is None:
            print(f"  unknown kind {kind_name!r}; one of {', '.join(KINDS)}")
            return
        try:
            info = self.table.create_index_as(column, kind)
        except (ValueError, RuntimeError) as exc:
            print(f"  {exc}")
            return
        self._describe_one(info)
        print("    forced, not measured -- choose_index times lookups and")
        print("    writes, which is the wrong benchmark for a counting column")

    def _describe_one(self, info) -> None:
        kind = str(info.kind).split(".")[-1]
        encoding = str(info.encoding).split(".")[-1]
        print(f"  {info.name}: {kind}/{encoding}, {info.rows:,} rows, "
              f"{info.distinct:,} distinct, {info.index_bytes/1e6:.2f} MB, "
              f"{info.ns_per_lookup:.1f} ns/lookup")

    def columns(self) -> None:
        if not self._require():
            return
        header = (f"  {'column':<12}{'type':>10}{'index':>12}{'encoding':>12}"
                  f"{'rows':>9}{'distinct':>10}{'MB':>8}{'ns':>8}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for info in self.table.describe():
            if not info.indexed:
                print(f"  {info.name:<12}{str(info.type).split('.')[-1]:>10}"
                      f"{'-- scan --':>12}")
                continue
            print(f"  {info.name:<12}{str(info.type).split('.')[-1]:>10}"
                  f"{str(info.kind).split('.')[-1]:>12}"
                  f"{str(info.encoding).split('.')[-1]:>12}"
                  f"{info.rows:>9,}{info.distinct:>10,}"
                  f"{info.index_bytes/1e6:>8.2f}{info.ns_per_lookup:>8.1f}")

    def catalog(self) -> None:
        if not self._require():
            return
        print("  persisted decisions (plans, never the fitted models):")
        for name in self.table.catalog.columns():
            plan = self.table.catalog.get(name)
            print(f"    {name:<12} {str(plan.kind).split('.')[-1]:<12}"
                  f"{plan.ns_per_lookup:>8.1f} ns  {plan.index_bytes/1e6:>6.2f} MB")

    def drop(self, column: str) -> None:
        if not self._require():
            return
        print(f"  {'dropped' if self.table.drop_index(column) else 'no index on'} "
              f"{column}; it still answers, by scanning")

    # -- structured queries -------------------------------------------------

    def _predicate(self, column: str, op_name: str, value: str):
        op = OPS.get(op_name)
        if op is None:
            print(f"  unknown op {op_name!r}; one of {', '.join(OPS)}")
            return None
        if op == PredOp.IsNull:
            return column, op, None
        try:
            typed = self.table.schema.parse(column, value)
        except (ValueError, RuntimeError) as exc:
            print(f"  {exc}")
            return None
        return column, op, typed

    def where(self, column: str, op_name: str, value: str) -> None:
        if not self._require():
            return
        parsed = self._predicate(column, op_name, value)
        if parsed is None:
            return
        start = time.perf_counter()
        rows, trace = self.table.select(*parsed)
        elapsed = time.perf_counter() - start
        print(f"  {trace.matched:,} rows in {elapsed*1e3:.2f} ms  [{trace.reason}]")
        for r in rows[:8]:
            print(f"    {r.key:>7}  {r.columns.get('category','?'):<8}"
                  f"{r.columns.get('price','?'):>8}  {r.columns.get('title','')}")
        if len(rows) > 8:
            print(f"    ... and {len(rows)-8:,} more")

    def conjunction(self, text: str) -> None:
        if not self._require():
            return
        parts = [p.strip().split() for p in text.split(",")]
        predicates = []
        for part in parts:
            if len(part) != 3:
                print("  each predicate is `col op value`, comma-separated")
                return
            parsed = self._predicate(*part)
            if parsed is None:
                return
            predicates.append(parsed)
        start = time.perf_counter()
        keys, trace = self.table.select_all(predicates)
        elapsed = time.perf_counter() - start
        print(f"  {len(keys):,} rows in {elapsed*1e3:.2f} ms  [{trace.reason}]")
        print(f"    {keys[:12]}{' ...' if len(keys) > 12 else ''}")

    def count(self, column: str, op_name: str, value: str) -> None:
        if not self._require():
            return
        parsed = self._predicate(column, op_name, value)
        if parsed is None:
            return
        start = time.perf_counter()
        n = self.table.count(*parsed)
        elapsed = time.perf_counter() - start
        info = self.table.info(column)
        print(f"  {n:,} rows in {elapsed*1e6:.1f} us")
        if info.indexed and info.kind == IndexKind.Bitmap:
            print("    answered by popcount: no row id was produced")
        else:
            print("    the row list was built and then measured; a bitmap column")
            print("    would answer this without materialising anything")

    # -- vector and hybrid --------------------------------------------------

    def knn(self, k: int, qi: int = 0) -> None:
        """The graph and brute force on the same query, side by side.

        Both stay selectable on purpose. The exact scan is the only exact
        answer, so it is what recall is defined against -- a graph that
        replaced it would leave "is it correct?" unanswerable.
        """
        if not self._require() or self.queries is None:
            return
        q = self.queries[qi % len(self.queries)]

        start = time.perf_counter()
        approx = self.table.knn("image", q, k=k, ef=self.ef)
        graph_us = (time.perf_counter() - start) * 1e6
        start = time.perf_counter()
        truth = self.table.knn("image", q, k=k, exact=True)
        exact_us = (time.perf_counter() - start) * 1e6

        overlap = len(set(n.key for n in approx) & set(n.key for n in truth))
        print(f"  graph {graph_us:>8.0f} us   brute force {exact_us:>8.0f} us   "
              f"recall@{k} {overlap}/{len(truth)}")
        for rank, n in enumerate(approx):
            record = self.table.get(n.key)
            label = record.columns.get("category", "?") if record else "?"
            print(f"    {rank:>3}. key {n.key:<7} d={n.score:<12.4f} {label}")

    def like(self, key: int, k: int = 5) -> None:
        """More-like-this, seeded by a row already in the table."""
        if not self._require():
            return
        record = self.table.get(key)
        if record is None:
            print(f"  no record {key}")
            return
        try:
            found = self.table.knn_by_key("image", key, k=k, ef=self.ef)
        except (ValueError, RuntimeError) as exc:
            print(f"  {exc}")
            return
        print(f"  like {key} ({record.columns.get('title','?')}, "
              f"{record.columns.get('category','?')}):")
        print("    the seed is excluded; a row is always its own nearest")
        print("    neighbour, so returning it would spend one of the k")
        for rank, n in enumerate(found):
            other = self.table.get(n.key)
            print(f"    {rank:>3}. key {n.key:<7} d={n.score:<12.4f} "
                  f"{other.columns.get('title','?') if other else '?'}")

    def hybrid(self, column: str, op_name: str, value: str, k: int, qi: int = 0) -> None:
        if not self._require() or self.queries is None:
            return
        parsed = self._predicate(column, op_name, value)
        if parsed is None:
            return
        q = self.queries[qi % len(self.queries)]

        start = time.perf_counter()
        found, trace = self.table.hybrid([parsed], "image", q, k=k, ef=self.ef)
        elapsed = time.perf_counter() - start

        print(f"  rows from the table: {trace.structured.reason}")
        self._show_plan(trace)
        print(f"  {len(found)} neighbours in {elapsed*1e6:.0f} us")
        for rank, n in enumerate(found):
            record = self.table.get(n.key)
            got = record.columns.get(column, "?") if record else "?"
            print(f"    {rank:>3}. key {n.key:<7} d={n.score:<12.4f} {column}={got}")

    def _show_plan(self, trace) -> None:
        plan = trace.plan
        print(f"  plan: {str(plan.kind).split('.')[-1]}  "
              f"({plan.matched_rows:,}/{plan.corpus_rows:,} rows, "
              f"{plan.selectivity:.1%})")
        print(f"    {plan.reason}")
        if plan.selectivity_was_free:
            print("    the selectivity cost a popcount; no row id was produced")
        if trace.without_vector:
            print(f"    {trace.without_vector:,} matching rows carry no embedding "
                  f"and so cannot be neighbours")

    def explain(self, column: str, op_name: str, value: str, k: int) -> None:
        if not self._require():
            return
        parsed = self._predicate(column, op_name, value)
        if parsed is None:
            return
        start = time.perf_counter()
        trace = self.table.explain_hybrid([parsed], "image", k)
        elapsed = time.perf_counter() - start
        self._show_plan(trace)
        print(f"    deciding took {elapsed*1e6:.1f} us")

    def plans(self, column: str, op_name: str, value: str, k: int, qi: int = 0) -> None:
        """Run every legal plan on the same query and compare.

        The point is the last column. Post-filter is what a system without a
        planner does, and it can return fewer than k rows however large the
        corpus is -- that is a correctness cost, not a slow one.
        """
        if not self._require() or self.queries is None:
            return
        parsed = self._predicate(column, op_name, value)
        if parsed is None:
            return
        q = self.queries[qi % len(self.queries)]
        chosen = self.table.explain_hybrid([parsed], "image", k)

        print(f"  {chosen.plan.matched_rows:,}/{chosen.plan.corpus_rows:,} rows "
              f"match ({chosen.plan.selectivity:.1%}); planner chose "
              f"{str(chosen.plan.kind).split('.')[-1]}")
        print()
        print(f"    {'plan':<24}{'us':>10}{'rows':>7}{'agrees':>9}")
        print("    " + "-" * 48)

        truth = None
        for kind in (PlanKind.PreFilter, PlanKind.FilteredGraph,
                     PlanKind.BitmapFilteredGraph, PlanKind.PostFilter):
            if not self.table.plan_available(kind, [parsed], "image"):
                print(f"    {str(kind).split('.')[-1]:<24}{'n/a':>10}"
                      f"{'':>7}{'':>9}   needs a bitmap column over these rows")
                continue
            start = time.perf_counter()
            for _ in range(3):
                found, _ = self.table.hybrid_with(kind, [parsed], "image", q,
                                                  k=k, ef=self.ef)
            elapsed = (time.perf_counter() - start) / 3
            keys = [n.key for n in found]
            if truth is None:
                truth = keys
            agrees = "yes" if keys == truth else "NO"
            print(f"    {str(kind).split('.')[-1]:<24}{elapsed*1e6:>10.0f}"
                  f"{len(keys):>7}{agrees:>9}")
        print()
        print("    post-filter returning fewer than k is not slowness, it is a")
        print("    wrong answer -- and it is what a system without a planner does")

    def calibrate(self) -> None:
        if not self._require() or self.queries is None:
            return
        before = self.table.prefilter_threshold
        measured = self.table.calibrate("image", self.queries[:30], k=10,
                                        ef=self.ef or 64)
        print(f"  crossover measured at {measured:.1%} on this corpus "
              f"(was {before:.0%})")
        print("  the inherited default came from a different corpus at a")
        print("  different ef, and the crossover moves with both")
        print("  no scalar column is involved: the synthetic filters are cut")
        print("  from the vector column's own rows")

    # -- vector maintenance -------------------------------------------------

    def vectors(self) -> None:
        if not self._require():
            return
        for info in self.table.describe_vectors():
            if not info.indexed:
                print(f"  {info.name}: declared, no index yet")
                continue
            print(f"  {info.name}: {info.rows:,} rows x {info.dim}-d, "
                  f"{str(info.structure).split('.')[-1]}, "
                  f"{str(info.metric).split('.')[-1]}, "
                  f"{info.memory_bytes/1e6:.1f} MB")
            print(f"    {info.orphans:,} orphaned rows -- HNSW cannot give a node")
            print("    back, so a delete masks rather than removes and the space")
            print("    returns at `compact`")
            if info.rows_are_keys:
                print("    the row id is the record key, which is what lets a")
                print("    table bitmap be handed to a vector search as a mask")
            else:
                print("    the row id is no longer the record key, so a table")
                print("    bitmap can no longer mask a vector search here: bit")
                print("    position i would name a different row in each")

    def compact(self) -> None:
        if not self._require():
            return
        info = self.table.vector_info("image")
        if info.orphans == 0:
            print("  nothing to reclaim")
            return
        start = time.perf_counter()
        freed = self.table.compact_vectors("image")
        print(f"  reclaimed {freed:,} rows in "
              f"{(time.perf_counter()-start)*1000:.0f} ms")
        print("    a full graph rebuild, which is why this is a command rather")
        print("    than something a delete does on its own")

    # -- durability ---------------------------------------------------------

    def put(self, key: int, assignments: list[str]) -> None:
        if not self._require():
            return
        # A token with no `=` continues the previous value, so `title=nike air`
        # works without quoting -- which is how anyone actually types it.
        columns: dict[str, str] = {}
        last = None
        for item in assignments:
            if "=" in item:
                name, _, value = item.partition("=")
                columns[name] = value
                last = name
            elif last is not None:
                columns[last] += " " + item
            else:
                print(f"  expected col=value, got {item!r}")
                return
        try:
            result = self.table.put(Record(key, columns))
        except (ValueError, RuntimeError) as exc:
            print(f"  refused: {exc}")
            return
        print(f"  {'inserted' if result.created else 'updated'} {key}: "
              f"{result.indexes_touched} index touches, "
              f"{result.rebuilds_triggered} rebuilds")

    def delete(self, key: int) -> None:
        if not self._require():
            return
        result = self.table.erase(key)
        print(f"  erased {key}: {result.indexes_touched} index touches"
              if result.indexes_touched or self.table.get(key) is None
              else f"  {key} was not present")

    def get(self, key: int) -> None:
        if not self._require():
            return
        record = self.table.get(key)
        print(f"  {record.columns if record else 'no such record'}")

    def checkpoint(self) -> None:
        if not self._require():
            return
        start = time.perf_counter()
        self.table.checkpoint()
        info = self.table.vector_info("image")
        print(f"  snapshot written, WAL truncated, catalog and "
              f"{info.rows:,} embeddings saved "
              f"({(time.perf_counter()-start)*1000:.0f} ms)")
        print("    the .fvecs sidecar is the only durability the embeddings")
        print("    have: they never go through the write-ahead log, because a")
        print("    128-float vector would dominate a JSON log")

    def recover(self) -> None:
        """Close the store and reopen it from disk.

        The moment the whole stack is visible at once: the records come back
        from the write-ahead log, and the index decisions come back from the
        catalog without being re-measured -- which is the expensive half.
        """
        if not self._require():
            return
        indexed = [c.name for c in self.table.describe() if c.indexed]
        before = len(self.table)
        self.table.checkpoint()
        self.store.close()
        self.table = None
        self.store = None

        start = time.perf_counter()
        self.store = RecordStore(str(self.root / "shop"))
        self.table = Table.open(self.store)
        reopen = time.perf_counter() - start

        info = self.table.vector_info("image")
        print(f"  reopened in {reopen*1000:.0f} ms")
        print(f"    {len(self.table):,} records recovered (was {before:,})")
        print(f"    schema read from disk: {len(self.table.schema)} columns")
        print(f"    catalog holds {len(self.table.catalog.columns())} decisions")
        print(f"    {info.rows:,} embeddings read back from the .fvecs sidecar")
        print("    the HNSW graph is not stored; it was replayed from the same")
        print("    insertions, and because the seed is fixed it is the *same*")
        print("    graph rather than a similar one")
        for name in indexed:
            start = time.perf_counter()
            self.table.create_index(name)
            print(f"    {name}: rebuilt in {(time.perf_counter()-start)*1000:.0f} ms "
                  f"by replaying the stored plan, not re-measuring it")

    # -- verification -------------------------------------------------------

    def check(self) -> None:
        """Every index against a brute-force scan of the store."""
        if not self._require():
            return
        problems = []
        try:
            self.table.validate()
        except (RuntimeError, ValueError) as exc:
            problems.append(f"table.validate: {exc}")

        rows = list(self.table.scan())
        for probe in (0, 12_500, 25_000, 49_999):
            want = sorted(r.key for r in rows
                          if int(r.columns.get("price", "-1")) < probe)
            got, _ = self.table.select_keys("price", PredOp.Lt, probe)
            if got != want:
                problems.append(f"price < {probe}: {len(got)} vs {len(want)}")
        for value in CATEGORIES:
            want = sorted(r.key for r in rows if r.columns.get("category") == value)
            got, _ = self.table.select_keys("category", PredOp.Eq, value)
            if got != want:
                problems.append(f"category == {value}: {len(got)} vs {len(want)}")
            if self.table.count("category", PredOp.Eq, value) != len(want):
                problems.append(f"count(category == {value}) disagrees with select")

        # The vector half, against numpy rather than against ourselves.
        if self.base is not None and self.queries is not None:
            live = self.table.vector_keys("image")
            index = {key: i for i, key in enumerate(live)}
            corpus = self.base[[key for key in live]]
            for q in self.queries[:3]:
                want = [live[i] for i in
                        np.argsort(((corpus - q) ** 2).sum(axis=1))[:5]]
                got = [n.key for n in self.table.knn("image", q, k=5, exact=True)]
                if got != want:
                    problems.append(f"exact knn disagrees with numpy: {got} vs {want}")
                    break
            # And the filter really filters.
            hits, _ = self.table.hybrid([("category", PredOp.Eq, "shoes")],
                                        "image", self.queries[0], k=10, ef=self.ef)
            wrong = [h.key for h in hits
                     if self.table.get(h.key).columns.get("category") != "shoes"]
            if wrong:
                problems.append(f"hybrid returned non-matching rows: {wrong}")
            if index and set(h.key for h in hits) - set(live):
                problems.append("hybrid returned a row with no embedding")

        if problems:
            print("  FAILED")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"  ok: every index agrees with a scan of all "
                  f"{len(rows):,} records, and the exact vector search agrees "
                  f"with numpy")

    def stats(self) -> None:
        if not self._require():
            return
        print(f"  source   {self.source}")
        print(f"  records  {len(self.table):,}")
        info = self.table.vector_info("image")
        print(f"  vectors  {info.rows:,} x {info.dim}-d, {info.orphans:,} orphaned, "
              f"{info.memory_bytes/1e6:.1f} MB")
        print(f"  ef       {self.ef or 'index default'}")
        print(f"  planner  crossover {self.table.prefilter_threshold:.0%}, "
              f"built per query from the table's own indexes")
        print(f"  writes   fsync per write; index maintenance is ~0.3% of one")
        print(f"  rebuilds {self.table.rebuilds()}")


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------


def split_command(line: str) -> list[str]:
    """Tokenise, keeping Windows paths intact (see scripts/try_btree.py)."""
    if os.name != "nt":
        return shlex.split(line)
    parts = shlex.split(line, posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p
            for p in parts]


def dispatch(pg: Playground, line: str) -> bool:
    try:
        parts = split_command(line)
    except ValueError as exc:
        print(f"  {exc}")
        return True
    if not parts:
        return True

    cmd, args = parts[0].lower(), parts[1:]
    try:
        if cmd in ("quit", "exit", "q"):
            return False
        elif cmd in ("help", "?", "h"):
            print(HELP)
        elif cmd == "load":
            pg.load(int(args[0]) if args else 5000,
                    int(args[1]) if len(args) > 1 else 32)
        elif cmd == "sift":
            pg.load(0, sift=True)
        elif cmd == "index":
            pg.create_index(args[0], float(args[1]) if len(args) > 1 else 0.0)
        elif cmd == "as":
            pg.create_index_as(args[0], args[1].lower())
        elif cmd == "drop":
            pg.drop(args[0])
        elif cmd == "columns":
            pg.columns()
        elif cmd == "catalog":
            pg.catalog()
        elif cmd == "where":
            pg.where(args[0], args[1].lower(), args[2] if len(args) > 2 else "")
        elif cmd == "and":
            pg.conjunction(" ".join(args))
        elif cmd == "count":
            pg.count(args[0], args[1].lower(), args[2] if len(args) > 2 else "")
        elif cmd == "knn":
            pg.knn(int(args[0]) if args else 5,
                   int(args[1]) if len(args) > 1 else 0)
        elif cmd == "like":
            pg.like(int(args[0]), int(args[1]) if len(args) > 1 else 5)
        elif cmd == "hybrid":
            pg.hybrid(args[0], args[1].lower(), args[2], int(args[3]),
                      int(args[4]) if len(args) > 4 else 0)
        elif cmd == "explain":
            pg.explain(args[0], args[1].lower(), args[2],
                       int(args[3]) if len(args) > 3 else 10)
        elif cmd == "plans":
            pg.plans(args[0], args[1].lower(), args[2],
                     int(args[3]) if len(args) > 3 else 10,
                     int(args[4]) if len(args) > 4 else 0)
        elif cmd == "calibrate":
            pg.calibrate()
        elif cmd == "vectors":
            pg.vectors()
        elif cmd == "compact":
            pg.compact()
        elif cmd == "ef":
            pg.ef = max(0, int(args[0]))
            print(f"  ef = {pg.ef or 'index default'}")
        elif cmd == "put":
            pg.put(int(args[0]), args[1:])
        elif cmd in ("del", "delete", "rm"):
            pg.delete(int(args[0]))
        elif cmd == "get":
            pg.get(int(args[0]))
        elif cmd == "checkpoint":
            pg.checkpoint()
        elif cmd == "recover":
            pg.recover()
        elif cmd in ("check", "c"):
            pg.check()
        elif cmd in ("stats", "s"):
            pg.stats()
        else:
            print(f"  unknown command {cmd!r}; type `help`")
    except (IndexError, ValueError) as exc:
        print(f"  bad arguments for {cmd!r}: {exc}")
        print("  type `help` for usage")
    return True


DEMO = [
    "load 20000",
    "index price",
    "index category",
    "as in_stock bitmap",
    "as band bitmap",
    "columns",
    "vectors",
    "where category eq shoes",
    "where title prefix nike",
    "count in_stock eq true",
    "and category eq shoes, price lt 25000",
    "knn 5",
    "like 0 5",
    "explain price lt 5000 10",
    "explain price lt 45000 10",
    "explain band lt 4 10",
    "hybrid price lt 25000 5",
    "plans band lt 4 5",
    "put 0 price=99 band=0 category=hats title=nike air in_stock=true created_at=2026-08-13",
    "get 0",
    "del 1",
    "vectors",
    "compact",
    "check",
    "recover",
    "check",
    "stats",
]


def run_demo() -> None:
    pg = Playground()
    print("A scripted walkthrough of the whole system. Every command below can")
    print("be typed at the prompt; run without --demo to explore.\n")
    for line in DEMO:
        print(f"> {line}")
        dispatch(pg, line)
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--load", type=int, metavar="N",
                        help="preload N synthetic shop rows")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--sift", action="store_true",
                        help="preload SIFT10K with synthetic attributes")
    parser.add_argument("--demo", action="store_true",
                        help="run a scripted walkthrough and exit")
    args = parser.parse_args(argv)

    if args.demo:
        run_demo()
        return 0

    pg = Playground()
    if args.sift:
        pg.load(0, sift=True)
    elif args.load:
        pg.load(args.load, args.dim)

    print("hybrid playground: table, indexes, planner. `help`, `quit`.")
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not dispatch(pg, line):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
