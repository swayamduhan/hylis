#!/usr/bin/env python3
"""Interactive playground for the B+ tree.

    python scripts/try_btree.py                  # empty tree, order 32
    python scripts/try_btree.py --order 4        # small order: splits happen fast
    python scripts/try_btree.py --load clustered:5000
    python scripts/try_btree.py --csv mydata.csv
    python scripts/try_btree.py --demo           # scripted walkthrough, no typing

Type `help` at the prompt for the command list.

A shadow Python dict is kept alongside the tree so `check` can diff the two
at any point -- the same differential idea the C++ fuzz test uses, exposed
interactively so you can try to break the tree by hand.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from hylis import BPlusTree, CompareOp
except ImportError as exc:  # pragma: no cover - depends on the build having run
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "Build the C++ extensions first:\n"
        "    cmake --preset default && cmake --build build"
    )


OPS = {
    "eq": CompareOp.Eq,
    "lt": CompareOp.Lt,
    "le": CompareOp.Le,
    "gt": CompareOp.Gt,
    "ge": CompareOp.Ge,
}

HELP = """
  insert <key> <value>     insert or overwrite            (alias: put)
  find <key>               look up one key                (alias: get)
  erase <key>              remove a key                   (alias: delete)
  has <key>                membership test

  range <lo> <hi>          values for keys in [lo, hi]
  lt|le|gt|ge|eq <value>   values matching the predicate
  keys [limit]             all keys, ascending
  items [limit]            all (key, value) pairs

  load <dist> <n> [seed]   fill from a generated dataset
                           dist: uniform lognormal sequential_gaps clustered
  csv <path> [kcol] [vcol] fill from a CSV file
  random <n> [max]         insert n random keys

  stats                    size, height, fill, order
  validate                 run the full invariant check
  check                    diff every key against a shadow dict
  bench [n]                time lookups vs dict vs sorted list
  clear                    empty the tree
  help / quit
"""


class Playground:
    def __init__(self, order: int = 32) -> None:
        self.tree = BPlusTree(order=order)
        self.shadow: dict[int, int] = {}

    # -- mutation ---------------------------------------------------------

    def insert(self, key: int, value: int) -> None:
        was_new = self.tree.insert(key, value)
        self.shadow[key] = value
        print(f"  {'inserted' if was_new else 'overwrote'} {key} -> {value}"
              f"   (size {len(self.tree)}, height {self.tree.height()})")

    def erase(self, key: int) -> None:
        removed = self.tree.erase(key)
        self.shadow.pop(key, None)
        if removed:
            print(f"  erased {key}   (size {len(self.tree)}, "
                  f"height {self.tree.height()})")
        else:
            print(f"  {key} was not present")

    def clear(self) -> None:
        self.tree.clear()
        self.shadow.clear()
        print("  cleared")

    def bulk_insert(self, pairs, label: str) -> None:
        start = time.perf_counter()
        for key, value in pairs:
            self.tree.insert(int(key), int(value))
            self.shadow[int(key)] = int(value)
        elapsed = time.perf_counter() - start
        n = len(self.tree)
        print(f"  loaded {label}: {n:,} keys in {elapsed*1000:.0f} ms "
              f"({elapsed/max(n,1)*1e6:.2f} us/insert)")
        self.stats()

    # -- queries ----------------------------------------------------------

    def find(self, key: int) -> None:
        got = self.tree.find(key)
        expected = self.shadow.get(key)
        flag = "" if got == expected else f"   MISMATCH (dict says {expected})"
        print(f"  find({key}) = {got}{flag}")

    def has(self, key: int) -> None:
        print(f"  {key} in tree = {self.tree.contains(key)}")

    def range(self, lo: int, hi: int) -> None:
        values = self.tree.range(lo, hi)
        expected = [v for k, v in sorted(self.shadow.items()) if lo <= k <= hi]
        self._show_results(f"range({lo}, {hi})", values, expected)

    def predicate(self, op_name: str, value: int) -> None:
        values = self.tree.range_query(OPS[op_name], value)
        keep = {
            "eq": lambda k: k == value,
            "lt": lambda k: k < value,
            "le": lambda k: k <= value,
            "gt": lambda k: k > value,
            "ge": lambda k: k >= value,
        }[op_name]
        expected = [v for k, v in sorted(self.shadow.items()) if keep(k)]
        self._show_results(f"{op_name}({value})", values, expected)

    def _show_results(self, label: str, values: list, expected: list) -> None:
        """Print a result set, truncated, and flag any disagreement.

        range/range_query return values only, not keys -- that is what the
        query planner wants (record ids to fetch), but it means you cannot
        read the matching keys off the output. Use `items` for pairs.
        """
        ok = values == expected
        head = values[:12]
        tail = "" if len(values) <= 12 else f" ... (+{len(values)-12} more)"
        print(f"  {label} -> {len(values)} value(s){'' if ok else '  MISMATCH'}")
        print(f"    {head}{tail}")
        if not ok:
            print(f"    dict expected {len(expected)}: {expected[:12]}")

    def keys(self, limit: int = 20) -> None:
        all_keys = self.tree.keys()
        print(f"  {len(all_keys)} keys: {all_keys[:limit]}"
              f"{'' if len(all_keys) <= limit else ' ...'}")

    def items(self, limit: int = 20) -> None:
        all_items = self.tree.items()
        print(f"  {len(all_items)} items: {all_items[:limit]}"
              f"{'' if len(all_items) <= limit else ' ...'}")

    # -- inspection -------------------------------------------------------

    def stats(self) -> None:
        n = len(self.tree)
        order = self.tree.order()
        height = self.tree.height()
        print(f"  size    {n:,}")
        print(f"  order   {order}   (max {order - 1} keys per node)")
        print(f"  height  {height}   ", end="")
        if n:
            # A perfectly packed tree of this order would need this many
            # levels; comparing shows how much slack the splits left behind.
            import math
            ideal = max(1, math.ceil(math.log(max(n, 1), order)))
            print(f"(a perfectly packed tree of order {order} would be {ideal})")
        else:
            print()
        if n:
            print(f"  lookups cost ~{height} node visits vs "
                  f"{n:,} for a linear scan")

    def validate(self) -> None:
        try:
            self.tree.validate()
            print("  all invariants hold")
        except RuntimeError as exc:
            print(f"  INVARIANT VIOLATED: {exc}")

    def check(self) -> None:
        """Full differential check against the shadow dict."""
        problems = []
        if len(self.tree) != len(self.shadow):
            problems.append(
                f"size {len(self.tree)} != dict size {len(self.shadow)}")

        tree_items = self.tree.items()
        dict_items = sorted(self.shadow.items())
        if tree_items != dict_items:
            for i, (a, b) in enumerate(zip(tree_items, dict_items)):
                if a != b:
                    problems.append(f"first difference at position {i}: "
                                    f"tree {a} vs dict {b}")
                    break
            else:
                problems.append("one is a prefix of the other")

        try:
            self.tree.validate()
        except RuntimeError as exc:
            problems.append(f"invariant: {exc}")

        if problems:
            print("  FAILED")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"  ok: {len(self.tree):,} keys match the dict exactly, "
                  f"invariants hold")

    def bench(self, n_queries: int = 20_000) -> None:
        """Time the tree against a dict and a sorted list.

        The interesting result is not that dict wins on point lookups -- it
        is a hash table, so of course it does. It is that dict cannot answer
        the range query at all, and the sorted list can only do so by being
        rebuilt on every insert. That tradeoff is the reason B+ trees exist.
        """
        if not self.shadow:
            print("  tree is empty; load something first")
            return

        keys = list(self.shadow)
        rng = random.Random(0)
        probes = [rng.choice(keys) for _ in range(n_queries)]
        sorted_keys = sorted(keys)

        start = time.perf_counter()
        for k in probes:
            self.tree.find(k)
        tree_t = time.perf_counter() - start

        start = time.perf_counter()
        for k in probes:
            self.shadow.get(k)
        dict_t = time.perf_counter() - start

        import bisect
        start = time.perf_counter()
        for k in probes:
            i = bisect.bisect_left(sorted_keys, k)
            _ = sorted_keys[i] == k
        bisect_t = time.perf_counter() - start

        print(f"  {n_queries:,} point lookups over {len(keys):,} keys")
        print(f"    B+ tree (C++)      {tree_t*1e6/n_queries:7.2f} us/query")
        print(f"    dict    (hash)     {dict_t*1e6/n_queries:7.2f} us/query"
              f"   <- O(1), but cannot do ranges")
        print(f"    bisect  (sorted)   {bisect_t*1e6/n_queries:7.2f} us/query"
              f"   <- O(log n), but O(n) inserts")

        lo = sorted_keys[len(sorted_keys) // 4]
        hi = sorted_keys[len(sorted_keys) // 2]
        start = time.perf_counter()
        got = self.tree.range(lo, hi)
        range_t = time.perf_counter() - start
        print(f"  range({lo}, {hi}) returned {len(got):,} values "
              f"in {range_t*1000:.2f} ms")
        print("    a dict would have to scan all "
              f"{len(keys):,} keys to answer this")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_dataset(pg: Playground, dist: str, n: int, seed: int = 0) -> None:
    try:
        from hylis import datasets
    except ImportError:
        print("  needs numpy: pip install -r requirements.txt")
        return
    try:
        data = datasets.synthetic_keys(dist, n=n, seed=seed)
    except ValueError as exc:
        print(f"  {exc}")
        return
    pg.bulk_insert(data.items(), f"{data.name} [{data.description}]")


def load_csv(pg: Playground, path: str, kcol: str = "0", vcol: str = "1") -> None:
    """Load (key, value) pairs from a CSV.

    Columns may be given by index or by header name. Rows whose key does not
    parse as an integer are skipped and counted, rather than aborting the
    load -- real CSVs have headers and junk rows.
    """
    file = Path(path)
    if not file.exists():
        print(f"  no such file: {file}")
        return

    skipped = 0
    pairs = []
    with open(file, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        print("  empty file")
        return

    header = rows[0]
    by_name = not kcol.lstrip("-").isdigit()
    if by_name:
        try:
            ki, vi = header.index(kcol), header.index(vcol)
        except ValueError:
            print(f"  columns {kcol!r}/{vcol!r} not in header {header}")
            return
        body = rows[1:]
    else:
        ki, vi = int(kcol), int(vcol)
        # Treat row 0 as a header only if it does not parse as data.
        body = rows if _is_int(rows[0][ki] if ki < len(rows[0]) else "") else rows[1:]

    for row in body:
        if max(ki, vi) >= len(row):
            skipped += 1
            continue
        try:
            pairs.append((int(row[ki]), int(float(row[vi]))))
        except ValueError:
            skipped += 1

    if not pairs:
        print(f"  no usable rows (skipped {skipped})")
        return
    pg.bulk_insert(pairs, f"{file.name} (skipped {skipped} unusable row(s))")


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------


def split_command(line: str) -> list[str]:
    """Tokenise an input line, keeping Windows paths intact.

    shlex's POSIX mode treats a backslash as an escape character, so
    `csv C:\\data\\keys.csv` arrives as `C:datakeys.csv`. Non-POSIX mode
    leaves backslashes alone but keeps the quote characters attached to the
    token, so those are stripped by hand.
    """
    if os.name != "nt":
        return shlex.split(line)
    parts = shlex.split(line, posix=False)
    return [
        p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p
        for p in parts
    ]


def dispatch(pg: Playground, line: str) -> bool:
    """Run one command. Returns False to quit."""
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
        elif cmd in ("insert", "put", "i"):
            pg.insert(int(args[0]), int(args[1]) if len(args) > 1 else int(args[0]))
        elif cmd in ("find", "get", "f"):
            pg.find(int(args[0]))
        elif cmd in ("erase", "delete", "del", "rm"):
            pg.erase(int(args[0]))
        elif cmd == "has":
            pg.has(int(args[0]))
        elif cmd in ("range", "r"):
            pg.range(int(args[0]), int(args[1]))
        elif cmd in OPS:
            pg.predicate(cmd, int(args[0]))
        elif cmd == "keys":
            pg.keys(int(args[0]) if args else 20)
        elif cmd == "items":
            pg.items(int(args[0]) if args else 20)
        elif cmd == "load":
            load_dataset(pg, args[0], int(args[1]),
                         int(args[2]) if len(args) > 2 else 0)
        elif cmd == "csv":
            load_csv(pg, args[0], *(args[1:3] or ("0", "1")))
        elif cmd == "random":
            hi = int(args[1]) if len(args) > 1 else 1_000_000
            rng = random.Random()
            pg.bulk_insert(
                ((rng.randrange(hi), i) for i in range(int(args[0]))),
                f"{args[0]} random keys < {hi}")
        elif cmd in ("stats", "s"):
            pg.stats()
        elif cmd in ("validate", "v"):
            pg.validate()
        elif cmd in ("check", "c"):
            pg.check()
        elif cmd in ("bench", "b"):
            pg.bench(int(args[0]) if args else 20_000)
        elif cmd == "clear":
            pg.clear()
        else:
            print(f"  unknown command {cmd!r}; type `help`")
    except (IndexError, ValueError) as exc:
        print(f"  bad arguments for {cmd!r}: {exc}")
        print("  type `help` for usage")

    return True


DEMO = [
    "insert 50 500", "insert 20 200", "insert 70 700", "insert 10 100",
    "insert 30 300", "insert 60 600", "insert 80 800",
    "stats",
    "find 30", "find 99",
    "range 20 60",
    "lt 50", "ge 60",
    "items",
    "erase 20", "erase 20",
    "check",
    "load clustered 20000",
    "bench 20000",
    "check",
]


def run_demo(order: int) -> None:
    pg = Playground(order=order)
    print(f"B+ tree demo, order {order} "
          f"(small on purpose: nodes hold {order - 1} keys, so splits "
          f"happen almost immediately)\n")
    for line in DEMO:
        print(f"> {line}")
        dispatch(pg, line)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--order", type=int, default=32,
                        help="branching order (>=3). Small values make splits "
                             "and merges happen constantly -- good for poking at.")
    parser.add_argument("--load", metavar="DIST:N",
                        help="preload a generated dataset, e.g. clustered:5000")
    parser.add_argument("--csv", metavar="PATH", help="preload from a CSV file")
    parser.add_argument("--demo", action="store_true",
                        help="run a scripted walkthrough and exit")
    args = parser.parse_args(argv)

    if args.demo:
        run_demo(max(args.order if args.order != 32 else 4, 3))
        return 0

    try:
        pg = Playground(order=args.order)
    except ValueError as exc:
        print(f"bad order: {exc}", file=sys.stderr)
        return 2

    if args.load:
        dist, _, n = args.load.partition(":")
        load_dataset(pg, dist, int(n or 1000))
    if args.csv:
        load_csv(pg, args.csv)

    print(f"B+ tree playground (order {pg.tree.order()}). "
          f"`help` for commands, `quit` to exit.")
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
