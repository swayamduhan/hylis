#!/usr/bin/env python3
"""Does a discontinuous CDF ever let a B+ tree beat the learned index?

    python scripts/experiment_discontinuity.py
    python scripts/experiment_discontinuity.py --quick
    python scripts/experiment_discontinuity.py -n 2000000

The answer this measured is **no**, and that contradicts what this project
originally claimed in writing. cpp/hylis/index/rmi.hpp used to say:

    "A CDF with a cliff in it has unbounded second derivative, so whichever
     model straddles the cliff eats the full error no matter how large M gets.
     That is exactly the case where a B+ tree -- which is distribution-free --
     should win."

The reasoning is sound as far as it goes and the conclusion does not follow.
This script is what settled it, and the header now carries the correction.

The experiment
--------------
Hold n fixed and sweep the number of discontinuities. `clustered` keys are
dense clumps separated by wide gaps, so the cluster count *is* the cliff
count: at 64 clusters a handful of models straddle a cliff, and at n/2 nearly
every model does. If the original claim held anywhere, it would hold at the
right-hand end of this table.

Why the claim fails
-------------------
Two things the original reasoning missed, both visible in the columns below:

  * Only models straddling a cliff are hurt, and there are at most as many of
    those as there are cliffs. Lookup cost follows the typical model, not the
    worst one -- so `max_error` can be enormous while `ns/lookup` barely moves.

  * A hurt model does not degrade into a scan. Above `search_threshold` the
    window search falls back to binary search, so its cost is capped at
    O(log n) over a *contiguous array* -- which is already cheaper than a B+
    tree's pointer-chased descent. The model can only ever degrade into
    something the tree is a slower version of.

Timings come from measure_plan() inside C++. Timing through pybind11 would
charge each index for a bridge crossing that costs more than the lookup.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from hylis import IndexKind, IndexPlan, RMIndex, measure_plan
    from hylis import datasets as ds
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

MODELS = [1024, 16384, 262144]


def require_optimised() -> None:
    import hylis._rmi as r

    if not getattr(r, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(r, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def cliff_counts(n: int) -> list[int]:
    """From "a few cliffs" to "nearly every key starts one"."""
    counts = [64, 1_000, 10_000, 100_000, n // 2]
    return sorted({c for c in counts if 1 < c <= n // 2})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=500_000, help="keys per row")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.quick:
        args.n = min(args.n, 100_000)

    require_optimised()

    print(f"n = {args.n:,} keys, 'clustered' shape, so cliffs == cluster count")
    print(f"log2(n) = {math.log2(args.n):.1f}, the probe budget a B+ tree "
          f"descent is competing against\n")

    header = (f"{'cliffs':>9}{'btree':>9}{'RMI 1k':>9}{'RMI 16k':>9}"
              f"{'RMI 262k':>10}{'best RMI':>10}{'speedup':>9}"
              f"{'max_err':>10}{'probes':>8}{'winner':>8}")
    print(header)
    print("-" * len(header))

    tree_wins = 0
    rows = 0
    for cliffs in cliff_counts(args.n):
        keys = [int(k) for k in ds.synthetic_keys(
            "clustered", n=args.n, seed=args.seed, n_clusters=cliffs).keys]
        values = list(range(len(keys)))

        tree = IndexPlan()
        tree.kind = IndexKind.BPlusTree
        tree.btree_order = 32
        tree_plan = measure_plan(keys, values, tree)

        learned = []
        for m in MODELS:
            plan = IndexPlan()
            plan.kind = IndexKind.RMI
            plan.rmi_models = m
            plan.search_threshold = 64
            learned.append(measure_plan(keys, values, plan))

        best = min(learned, key=lambda p: p.ns_per_lookup)

        # The two columns that explain the result: a huge worst-case window
        # alongside a probe count that refuses to follow it upward.
        probe_index = RMIndex(1024, 64)
        probe_index.build(keys, values)
        stats = probe_index.stats()
        worst_probes = max(probe_index.probes(k) for k in keys[::max(1, len(keys) // 500)])

        winner = "btree" if tree_plan.ns_per_lookup < best.ns_per_lookup else "RMI"
        tree_wins += winner == "btree"
        rows += 1
        speedup = tree_plan.ns_per_lookup / best.ns_per_lookup

        print(f"{cliffs:>9,}{tree_plan.ns_per_lookup:>9.1f}"
              f"{learned[0].ns_per_lookup:>9.1f}{learned[1].ns_per_lookup:>9.1f}"
              f"{learned[2].ns_per_lookup:>10.1f}{best.ns_per_lookup:>10.1f}"
              f"{speedup:>8.2f}x{stats.max_error:>10,}{worst_probes:>8}{winner:>8}")

    print()
    if tree_wins == 0:
        print(f"  The B+ tree won 0 of {rows} rows. The prediction that a")
        print("  discontinuous CDF favours it does not hold at any cliff density.")
        print()
        print("  Read the last three columns together: max_err grows into the")
        print("  thousands while probes stays flat and ns/lookup barely moves.")
        print("  That is the whole explanation -- a bad model costs a bounded")
        print("  search over a contiguous array, and a B+ tree descent is a")
        print("  slower version of exactly that.")
        print()
        print("  The B+ tree's real claim is mutability, not any shape of data.")
        print("  That is what dynamic_rmi.hpp exists to answer.")
    else:
        print(f"  The B+ tree won {tree_wins} of {rows} rows. The original")
        print("  prediction holds in that regime; rmi.hpp should be restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
