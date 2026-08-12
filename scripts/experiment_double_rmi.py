#!/usr/bin/env python3
"""E1: does a learned index work on float64 keys?

    python scripts/experiment_double_rmi.py            # the full sweep
    python scripts/experiment_double_rmi.py --quick
    python scripts/experiment_double_rmi.py --sizes 100000,1000000,10000000

The typed column layer has to decide whether `Double` belongs in the same row
as `Int64` in the index-family table, or in the same row as `String`. That is
not a question to answer by intuition, so this measures it.

Two things could go wrong, and only one of them is fatal:

  * **Exactness.** RMIndex's guarantee comes from error bounds measured in
    *position* space, so it should survive any key type that sorts. If it does
    not, `Double` cannot have a learned index at any speed, and that is the
    end of the discussion. RMIndex.validate() checks every key falls inside
    its predicted window and is findable, so a clean return is the gate.

  * **Error magnitude.** float64 resolution is wildly non-uniform in value
    space -- there are as many representable doubles between 1e-8 and 1e-7 as
    between 1e7 and 1e8 -- so a piecewise-linear fit to the CDF may behave
    quite differently from the integer case. This is a speed question, not a
    correctness one.

The last arm exists to provoke exactly that: values log-spaced across sixteen
orders of magnitude, which is the worst case the representation allows. It is
reported separately because it is a stress test rather than a plausible column.

Decision rule, fixed before running: adopt the Double RMI iff it is exact in
every arm AND beats the B+ tree on at least three of the five realistic
distributions at the largest size. Otherwise Double columns get the tree only,
and that becomes a recorded measurement rather than an assumption.
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
    from hylis import ColumnIndex, IndexKind, IndexPlan, KeyEncoding, LogicalType
    from hylis._rmi import measure_plan_typed
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

# The five realistic shapes, plus one deliberate stress case.
DISTRIBUTIONS = ["uniform", "lognormal", "cents", "latlong", "clustered"]
STRESS = "wide_range"
MODEL_COUNTS = [64, 1024, 16384, 262144]


def require_optimised() -> None:
    import hylis._rmi as rmi

    if not getattr(rmi, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(rmi, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def make_keys(distribution: str, n: int, seed: int = 0) -> np.ndarray:
    """Sorted, unique float64 keys with a given shape.

    De-duplication is done after sorting and the count is what the caller gets,
    because both structures require strictly ascending unique keys and quietly
    changing n between arms would make the rows incomparable.
    """
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        v = rng.uniform(0.0, 1.0, size=n)
    elif distribution == "lognormal":
        v = rng.lognormal(mean=0.0, sigma=2.0, size=n)
    elif distribution == "cents":
        # Money done properly is an integer number of cents; this is what
        # happens when someone stores it as a float anyway. Values are exactly
        # representable, so the CDF is as clean as an integer column's.
        v = rng.integers(0, 100_000_000, size=n).astype(np.float64) / 100.0
    elif distribution == "latlong":
        # Six decimal places, the usual GPS precision, over a bounded range.
        v = np.round(rng.uniform(-180.0, 180.0, size=n), 6)
    elif distribution == "clustered":
        # Dense clumps with wide empty gaps: the shape that carries ~45x the
        # position error of uniform for integer keys.
        centres = rng.uniform(0.0, 1e6, size=max(2, n // 1000))
        v = rng.choice(centres, size=n) + rng.normal(0.0, 0.01, size=n)
    elif distribution == STRESS:
        # Sixteen orders of magnitude. Half the keys live below 1.0, where
        # float64 has almost all of its resolution, and the CDF over the raw
        # value is therefore a near-vertical wall at the origin.
        v = np.power(10.0, rng.uniform(-8.0, 8.0, size=n))
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    v = np.unique(np.sort(v))
    return v


def tree_plan() -> IndexPlan:
    plan = IndexPlan()
    plan.kind = IndexKind.BPlusTree
    plan.type = LogicalType.Double
    plan.encoding = KeyEncoding.Native
    plan.btree_order = 32
    return plan


def rmi_plan(models: int) -> IndexPlan:
    plan = IndexPlan()
    plan.kind = IndexKind.RMI
    plan.type = LogicalType.Double
    plan.encoding = KeyEncoding.Native
    plan.rmi_models = models
    plan.search_threshold = 64
    return plan


def exactness_holds(keys: list, values: list, plan: IndexPlan) -> bool:
    """Every key inside its predicted window and findable.

    Checked in C++ by ColumnIndex.validate(), which walks all n keys. Doing it
    across the bridge would cost a microsecond per key and measure pybind11.
    """
    column = ColumnIndex.build_typed_with(LogicalType.Double, keys, values, plan)
    try:
        column.validate()
        return True
    except (RuntimeError, ValueError):
        return False


def run_one(distribution: str, n: int, seed: int, verbose: bool) -> dict:
    keys_np = make_keys(distribution, n, seed)
    keys = keys_np.tolist()
    values = list(range(len(keys)))
    actual = len(keys)

    start = time.perf_counter()
    tree = measure_plan_typed(LogicalType.Double, keys, values, tree_plan())
    tree_build_ms = (time.perf_counter() - start) * 1000.0

    best = None
    best_models = 0
    best_build_ms = 0.0
    exact_everywhere = True
    for models in MODEL_COUNTS:
        if models > actual:
            break
        plan = rmi_plan(models)
        start = time.perf_counter()
        measured = measure_plan_typed(LogicalType.Double, keys, values, plan)
        build_ms = (time.perf_counter() - start) * 1000.0
        if not exactness_holds(keys, values, plan):
            exact_everywhere = False
            if verbose:
                print(f"    NOT EXACT at M={models}")
        if best is None or measured.ns_per_lookup < best.ns_per_lookup:
            best, best_models, best_build_ms = measured, models, build_ms

    assert best is not None
    return {
        "distribution": distribution,
        "n": actual,
        "exact": exact_everywhere,
        "models": best_models,
        "max_error": best.max_error,
        "rmi_ns": best.ns_per_lookup,
        "tree_ns": tree.ns_per_lookup,
        "speedup": tree.ns_per_lookup / best.ns_per_lookup if best.ns_per_lookup else 0.0,
        "rmi_bytes": best.index_bytes,
        "tree_bytes": tree.index_bytes,
        "rmi_build_ms": best_build_ms,
        "tree_build_ms": tree_build_ms,
    }


def print_table(rows: list) -> None:
    header = (f"{'distribution':>14}{'n':>10}{'exact':>7}{'models':>8}"
              f"{'max_err':>10}{'rmi ns':>9}{'tree ns':>9}{'speedup':>9}"
              f"{'rmi MB':>9}{'tree MB':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['distribution']:>14}{r['n']:>10,}"
              f"{('yes' if r['exact'] else 'NO'):>7}{r['models']:>8,}"
              f"{r['max_error']:>10,}{r['rmi_ns']:>9.1f}{r['tree_ns']:>9.1f}"
              f"{r['speedup']:>8.2f}x{r['rmi_bytes']/1e6:>9.1f}"
              f"{r['tree_bytes']/1e6:>9.1f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="100000,1000000",
                        help="comma-separated key counts (default 100000,1000000)")
    parser.add_argument("--quick", action="store_true",
                        help="one small size, for a smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    require_optimised()
    sizes = [20000] if args.quick else [int(s) for s in args.sizes.split(",")]

    print("E1: is a learned index exact and competitive over float64 keys?")
    print("Baseline is a B+ tree over the same keys, both timed in C++.")
    print("The RMI row reports the best model count of "
          f"{MODEL_COUNTS}.\n")

    all_rows = []
    for n in sizes:
        print(f"== n = {n:,} " + "=" * 40)
        rows = []
        for dist in DISTRIBUTIONS:
            rows.append(run_one(dist, n, args.seed, args.verbose))
        print_table(rows)

        stress = run_one(STRESS, n, args.seed, args.verbose)
        print()
        print("  stress arm, reported apart because it is not a plausible column:")
        print_table([stress])
        all_rows.append((n, rows, stress))
        print()

    # ---- the decision rule, applied to the largest size -------------------
    n, rows, stress = all_rows[-1]
    exact = all(r["exact"] for r in rows) and stress["exact"]
    wins = sum(1 for r in rows if r["speedup"] > 1.0)

    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    print(f"  exactness held in every arm:        "
          f"{'yes' if exact else 'NO'}  (including the stress arm)")
    print(f"  RMI beat the tree on:               {wins} of {len(rows)} "
          f"realistic distributions at n={n:,}")

    verdict = exact and wins >= 3
    print()
    if verdict:
        print("  ADOPT. Double joins Int64 and Timestamp as a type a learned")
        print("  index may serve. candidates_for() already offers it; this is")
        print("  the measurement that says it should.")
    else:
        print("  DO NOT ADOPT. Double columns get the B+ tree only, and")
        print("  type_supports_rmi(Double) should return False.")

    if exact:
        ratio = stress["max_error"] / max(1, max(r["max_error"] for r in rows))
        print()
        print("  On the stress arm the guarantee held but the error bound grew")
        print(f"  {ratio:.0f}x against the worst realistic arm. That is the shape")
        print("  of the risk with float64 keys: the models get worse, the")
        print("  answers do not. Exactness is measured per key at build time,")
        print("  so a bad fit costs comparisons and never correctness.")
    return 0 if verdict else 0


if __name__ == "__main__":
    raise SystemExit(main())
