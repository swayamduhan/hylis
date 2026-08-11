#!/usr/bin/env python3
"""What more second-stage models actually buy, per key distribution.

    python scripts/experiment_curvature.py
    python scripts/experiment_curvature.py -n 200000

The direct answer to "what happens when the distribution is not linear".

A single line through a curved CDF is hopeless, which is why an RMI has a
second stage at all: stage 1 cuts the key space into M slices and each slice
gets its own line. That is piecewise-linear approximation, and for a smooth
curve the error of a linear fit over an interval of width h is bounded by
(h^2/8)*max|f''| -- so error should fall roughly quadratically in M.

What that cannot fix is a *discontinuity*. A CDF with a cliff has unbounded
second derivative, so whichever model straddles the cliff absorbs the full
error however many models are available. This script shows both behaviours
side by side.
"""

from __future__ import annotations

import argparse
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

DISTRIBUTIONS = ["sequential_gaps", "uniform", "lognormal", "clustered"]
MODEL_COUNTS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]


def require_optimised() -> None:
    import hylis._rmi as rmi

    if not getattr(rmi, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(rmi, '__build_type__', '?')} -- timings would "
            "be meaningless.\n    cmake --preset release && cmake --build build"
        )


def sweep(dist: str, n: int, with_timing: bool) -> None:
    data = ds.synthetic_keys(dist, n=n, seed=0)
    keys = data.keys.tolist()
    values = list(range(len(keys)))

    single = data.position_error()
    print(f"\n{dist}  ({data.description})")
    print(f"  one global line would need a window of {single[1]:,.0f} records "
          f"(mean {single[0]:,.0f})")
    print(f"  {'models':>8}  {'max err':>10}  {'mean err':>10}  "
          f"{'empty':>14}  {'ns/lookup':>10}  {'vs M=1':>8}")

    baseline = None
    for models in MODEL_COUNTS:
        if models > n:
            break
        index = RMIndex(models=models)
        index.build(keys, values)
        stats = index.stats()
        if baseline is None:
            baseline = stats.mean_error

        ns = ""
        if with_timing:
            plan = IndexPlan()
            plan.kind = IndexKind.RMI
            plan.rmi_models = models
            ns = f"{measure_plan(keys, values, plan).ns_per_lookup:10.1f}"

        ratio = baseline / stats.mean_error if stats.mean_error > 0 else float("inf")
        print(f"  {models:>8,}  {stats.max_error:>10,}  {stats.mean_error:>10,.1f}  "
              f"{stats.empty_models:>6,}/{stats.models:<7,}  {ns:>10}  {ratio:>7.0f}x")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=100_000, help="keys per distribution")
    parser.add_argument("--distributions", type=str, default=",".join(DISTRIBUTIONS))
    parser.add_argument("--no-timing", action="store_true",
                        help="skip lookup timing (error only, much faster)")
    args = parser.parse_args(argv)

    if not args.no_timing:
        require_optimised()

    print(f"Error vs second-stage model count, n={args.n:,} per distribution")

    for dist in [d.strip() for d in args.distributions.split(",")]:
        sweep(dist, args.n, not args.no_timing)

    print()
    print("What to look for:")
    print("  * lognormal is curved but continuous, so mean error keeps falling")
    print("    as models are added -- curvature is bought off, cheaply.")
    print("  * clustered is stepped, not curved. Past the point where each")
    print("    cluster owns a model, extra models are routed nothing and the")
    print("    error stops moving entirely. That floor is set by the gaps in")
    print("    the data, not by the model budget, and is the reason a")
    print("    distribution-free B+ tree still has a job.")
    print("  * max error can plateau while mean error keeps improving: stage 1")
    print("    routes by predicted position, so a skewed CDF piles its dense")
    print("    region into the first few models however large M is. That is")
    print("    the limitation scripts/experiment_stage1.py investigates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
