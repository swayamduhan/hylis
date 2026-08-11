#!/usr/bin/env python3
"""B+ tree vs static RMI vs dynamic RMI, on the SOSD benchmark corpora.

    python scripts/fetch_data.py sosd            # how to get the files
    python scripts/bench_sosd.py                 # everything in data/
    python scripts/bench_sosd.py --limit 10000000
    python scripts/bench_sosd.py --synthetic     # no download needed

SOSD (Marcus et al., "Benchmarking Learned Indexes", VLDB 2021) is the
standard corpus for the claim a learned index makes, and it is the honest
place to check one specific prediction this project already committed to in
writing. cpp/hylis/index/rmi.hpp says:

    "What that does *not* fix is a discontinuity. A CDF with a cliff in it
     (the `clustered` generator, or SOSD's `fb`) has unbounded second
     derivative, so whichever model straddles the cliff eats the full error
     no matter how large M gets. That is exactly the case where a B+ tree --
     which is distribution-free -- should win."

That was written before `fb` had ever been run. The `fb` row below is the
test of it, and if it comes out the other way the header comment is what
gets corrected.

Lookup timings come from measure_plan() on the C++ side. Timing through
pybind11 would charge the index for a bridge crossing that costs more than
the lookup does.
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
        DynamicConfig,
        DynamicRMIndex,
        IndexKind,
        IndexPlan,
        RMIndex,
        choose_index,
        measure_plan,
    )
    from hylis import datasets as ds
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )


def require_optimised() -> None:
    import hylis._rmi as r

    if not getattr(r, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(r, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def corpora(args):
    """Whatever SOSD files are present, or synthetic stand-ins."""
    if not args.synthetic:
        data_dir = ds.default_data_dir()
        found = sorted(
            p for p in data_dir.iterdir()
            if p.is_file() and p.name.lower().endswith(("uint32", "uint64"))
        ) if data_dir.exists() else []
        if args.only:
            wanted = set(args.only.split(","))
            found = [p for p in found if any(w in p.name for w in wanted)]
        if found:
            for path in found:
                yield ds.load_sosd(path, limit=args.limit)
            return
        print("No SOSD files in data/; falling back to synthetic stand-ins.")
        print("    python scripts/fetch_data.py sosd\n")

    # The synthetic shapes are not SOSD, and the table says so. `clustered` is
    # the one that mimics `fb`: tight clumps separated by enormous gaps, which
    # is the shape piecewise-linear fitting cannot converge on.
    n = min(args.limit or 2_000_000, 2_000_000)
    for name in ("uniform", "lognormal", "clustered"):
        keys = ds.synthetic_keys(name, n=n, seed=0).keys.astype(np.int64)
        yield ds.SosdKeys(keys=keys, name=f"synthetic:{name}",
                          total_read=len(keys), dropped=0)


def bench(corpus, args) -> None:
    keys = [int(k) for k in corpus.keys]
    values = list(range(len(keys)))
    n = len(keys)

    print(f"\n{'=' * 78}")
    print(f"{corpus.name}  --  {n:,} unique keys", end="")
    if corpus.dropped:
        print(f", {corpus.dropped:,} duplicates dropped "
              f"({corpus.duplicate_fraction:.2%})", end="")
    print(f"\n{'=' * 78}")

    # How hard this corpus is for a learned index, before building anything:
    # the position error of a single linear fit to its CDF. Not R^2 — the
    # clustered generator scores ~0.995 there while costing 45x the error of
    # uniform, because the damage is local and a global R^2 cannot see it.
    kd = corpus.as_dataset()
    mean_err, max_err = kd.position_error()
    print(f"one linear model over the whole CDF: mean error {mean_err:,.0f} "
          f"records, max {max_err:,.0f}")

    rows = []

    tree = IndexPlan()
    tree.kind = IndexKind.BPlusTree
    tree.btree_order = 32
    rows.append(("B+ tree", measure_plan(keys, values, tree)))

    for models in (1024, 16384, 262144):
        if models > n:
            break
        plan = IndexPlan()
        plan.kind = IndexKind.RMI
        plan.rmi_models = models
        plan.search_threshold = 64
        rows.append((f"RMI M={models:,}", measure_plan(keys, values, plan)))

    header = f"{'index':<16}{'ns/lookup':>12}{'max error':>12}{'size':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    best_name, best_ns = None, float("inf")
    for name, plan in rows:
        print(f"{name:<16}{plan.ns_per_lookup:>12.1f}{plan.max_error:>12,}"
              f"{human_bytes(plan.index_bytes):>12}")
        if plan.ns_per_lookup < best_ns:
            best_name, best_ns = name, plan.ns_per_lookup

    tree_ns = rows[0][1].ns_per_lookup
    verdict = "the B+ tree" if best_name == "B+ tree" else best_name
    print(f"\n  fastest: {verdict}", end="")
    if best_name != "B+ tree":
        print(f" -- {tree_ns / best_ns:.2f}x the B+ tree")
    else:
        rmi_best = min(p.ns_per_lookup for nm, p in rows[1:]) if len(rows) > 1 else 0
        if rmi_best:
            print(f" -- {rmi_best / tree_ns:.2f}x faster than the best RMI")
    print()

    # And what the auto-tuner picks when it measures for itself.
    chosen = choose_index(keys, values)
    kind = "btree" if chosen.kind == IndexKind.BPlusTree else "rmi"
    print(f"  choose_index() picked: {kind}", end="")
    if chosen.kind == IndexKind.RMI:
        print(f" with M={chosen.rmi_models:,}", end="")
    print(f" at {chosen.ns_per_lookup:.1f} ns")

    if args.dynamic:
        bench_dynamic(keys, values, args)


def bench_dynamic(keys, values, args) -> None:
    """What the same corpus costs once the index has to accept writes."""
    cfg = DynamicConfig()
    cfg.second_stage_size = 1024

    start = time.perf_counter()
    static = RMIndex(1024)
    static.build(keys, values)
    static_build = time.perf_counter() - start

    start = time.perf_counter()
    dynamic = DynamicRMIndex(cfg)
    dynamic.build(keys, values)
    dynamic_build = time.perf_counter() - start

    # An append-shaped batch, which is the write pattern the incremental merge
    # is actually good at. A scattered one is measured in
    # scripts/experiment_merge_threshold.py and is much less flattering.
    batch = max(1, len(keys) // 100)
    k = keys[-1]
    start = time.perf_counter()
    for i in range(batch):
        k += 1000
        dynamic.insert(k, i)
    dynamic.merge()
    write_seconds = time.perf_counter() - start

    s = dynamic.stats()
    updates = s.models_shifted + s.models_refitted
    shifted = 100.0 * s.models_shifted / updates if updates else 0.0
    print(f"\n  writable: build {dynamic_build:.2f}s vs {static_build:.2f}s static;"
          f" {batch:,} appends + merge in {write_seconds:.2f}s")
    print(f"            {shifted:.1f}% of model updates were O(1) shifts, "
          f"{s.keys_rescanned / max(len(keys), 1):.2f}x n keys rescanned")
    dynamic.validate()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=10_000_000,
                        help="keys to read per corpus; 0 for all of them. The "
                             "full 200M would need ~1.6GB resident per file.")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated substrings, e.g. fb,books")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip SOSD and use synthetic stand-ins")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false",
                        help="skip the writable-index rows")
    parser.set_defaults(dynamic=True)
    args = parser.parse_args(argv)
    if args.limit == 0:
        args.limit = None

    require_optimised()

    any_run = False
    for corpus in corpora(args):
        bench(corpus, args)
        any_run = True

    if not any_run:
        raise SystemExit("nothing to benchmark")

    print("\nReading this:")
    print("  * 'max error' is the widest window a lookup can be forced into,")
    print("    in records. It is what a discontinuous CDF inflates, and it is")
    print("    the mechanism behind any row where the B+ tree wins.")
    print("  * ns/lookup is measured inside C++ by measure_plan(), on this")
    print("    machine, against a probe set that is deliberately part misses.")
    print("  * choose_index() is the auto-tuner making the same decision by")
    print("    building and timing every candidate -- the CDFShop stance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
