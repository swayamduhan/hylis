#!/usr/bin/env python3
"""Does the hybrid query planner pick the right plan?

    python scripts/bench_planner.py                 # synthetic corpus
    python scripts/bench_planner.py --sift          # SIFT10K + synthetic attributes
    python scripts/bench_planner.py --quick

A hybrid query is a structured predicate and a vector search together:

    SELECT id FROM t
    WHERE  price < $t
    ORDER BY distance(embedding, $q)
    LIMIT  10

There are three ways to run it and the right one depends entirely on how
selective the predicate is. This times all three at each selectivity, notes
which actually won, and reports how often the planner agreed.

That last number is the one that matters. Everything else in the test suite
checks the planner is *correct*; this checks it is *useful*, which is a
different question and the only one the module exists to answer.

Two failure modes are reported rather than hidden:

  * **regret** -- how much slower the plan the planner chose was than the one
    that actually won. A planner that is wrong 20% of the time but only ever
    by 3% is a better planner than one that is wrong 5% of the time and
    catastrophically so.
  * **short** -- rows the post-filter plan failed to return. It is the plan a
    system without a planner uses, and it can silently return fewer than k.
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
        CompareOp,
        FlatIndex,
        HnswIndex,
        HybridPlanner,
        Metric,
        PlanKind,
        Predicate,
    )
    from hylis import datasets as ds
    from hylis.query import ScalarColumn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

SELECTIVITIES = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]


def require_optimised() -> None:
    import hylis._planner as p

    if not getattr(p, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(p, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def load(args):
    if args.sift:
        try:
            vectors = ds.load_sift("siftsmall")
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc}\n    python scripts/fetch_data.py siftsmall")
    else:
        n = 5000 if args.quick else 20000
        vectors = ds.random_vectors(n=n, dim=32, n_queries=200, seed=0,
                                    n_clusters=max(2, n // 200))
    return ds.make_hybrid(vectors, seed=0)


def timed(fn, repeats):
    """Best of `repeats`. The minimum is the run least interfered with."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sift", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=64)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="the planner's pre-filter crossover")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure the crossover on this corpus first and "
                             "adopt it, instead of using --threshold. The "
                             "inherited 0.5 came from a different corpus at a "
                             "different ef, and the crossover moves.")
    args = parser.parse_args(argv)
    if args.quick:
        args.queries = 20

    require_optimised()
    corpus = load(args)
    base = corpus.vectors.base
    queries = corpus.vectors.queries[: args.queries]
    n, dim = corpus.vectors.n, corpus.vectors.dim

    exact = FlatIndex(dim)
    exact.add_batch(base)
    graph = HnswIndex(dim, Metric.L2, M=16, ef_construction=200)
    graph.add_batch(base)

    price = ScalarColumn(corpus.attributes["price"])
    planner = HybridPlanner(args.threshold)
    price.attach_to(planner, "price")
    planner.set_exact(exact)
    planner.set_graph(graph)

    print(f"{corpus.name}: {n:,} x {dim}-d, {len(queries)} queries, k={args.k}, "
          f"ef={args.ef}")
    if args.calibrate:
        inherited = args.threshold
        measured = planner.calibrate("price", queries, k=args.k, ef=args.ef)
        args.threshold = measured
        print(f"calibrated: crossover measured at {measured:.1%} on this corpus "
              f"(the inherited default was {inherited:.0%})\n")
    else:
        print(f"planner crossover set to {args.threshold:.0%} "
              f"-- --calibrate measures it instead\n")

    header = (f"{'selectivity':>12}{'rows':>9}{'prefilter':>11}{'graph':>10}"
              f"{'postfilter':>12}{'winner':>12}{'chose':>12}{'regret':>9}"
              f"{'short':>7}")
    print(header)
    print("-" * len(header))

    agreed = 0
    rows = 0
    total_regret = 0.0
    worst_regret = 1.0
    worst_at = 0.0
    crossover = None
    previous_winner = None

    for selectivity in SELECTIVITIES:
        cut = price.key_cut_for_selectivity(selectivity)
        predicate = Predicate("price", CompareOp.Lt, cut)
        matched = planner.explain(predicate, args.k).matched_rows
        if matched <= args.k:
            continue  # every match is in the answer; there is no plan to pick

        def run(kind):
            def go():
                for q in queries:
                    planner.search_with(kind, predicate, q, k=args.k, ef=args.ef)
            return go

        repeats = 3
        t_pre = timed(run(PlanKind.PreFilter), repeats)
        t_graph = timed(run(PlanKind.FilteredGraph), repeats)
        t_post = timed(run(PlanKind.PostFilter), repeats)

        # How often post-filter came up short. This is not a timing result --
        # it is the correctness cost of the plan a system without a planner
        # would have used.
        short = 0
        for q in queries:
            got = planner.search_with(PlanKind.PostFilter, predicate, q,
                                      k=args.k, ef=args.ef)
            short += args.k - len(got)

        times = {PlanKind.PreFilter: t_pre, PlanKind.FilteredGraph: t_graph}
        winner = min(times, key=times.get)
        chosen = planner.explain(predicate, args.k).kind
        regret = times[chosen] / times[winner]

        agreed += chosen == winner
        rows += 1
        total_regret += regret
        if regret > worst_regret:
            worst_regret, worst_at = regret, selectivity

        if previous_winner is not None and winner != previous_winner and crossover is None:
            crossover = selectivity
        previous_winner = winner

        def name(kind):
            return {PlanKind.PreFilter: "prefilter",
                    PlanKind.FilteredGraph: "graph"}[kind]

        print(f"{selectivity:>11.1%}{matched:>9,}{t_pre*1e3:>10.1f}m"
              f"{t_graph*1e3:>9.1f}m{t_post*1e3:>11.1f}m"
              f"{name(winner):>12}{name(chosen):>12}"
              f"{regret:>8.2f}x{short:>7}")

    print()
    if rows:
        print(f"  planner agreed with the measured winner on {agreed}/{rows} rows"
              f" ({agreed / rows:.0%})")
        print(f"  mean regret {total_regret / rows:.2f}x, worst {worst_regret:.2f}x"
              f" at {worst_at:.1%} selectivity")
    if crossover is not None:
        print(f"  measured crossover between {crossover:.1%} and the row above it;"
              f" planner threshold is {args.threshold:.0%}")
    else:
        print("  no crossover in this range -- one plan won everywhere, so the")
        print("  planner had nothing to decide and its agreement rate is not")
        print("  evidence of anything.")

    print("\nReading this table:")
    print("  * 'regret' is how much slower the chosen plan was than the winner.")
    print("    1.00x means it chose correctly. A planner that is often wrong by")
    print("    a little beats one that is rarely wrong by a lot, so this matters")
    print("    more than the agreement rate does.")
    print("  * 'short' counts rows post-filter failed to return across all")
    print("    queries. It is not slow, it is *wrong*, and it is what a system")
    print("    without a planner does.")
    print("  * rows where matches <= k are skipped: every match is in the answer")
    print("    then, so there is no plan to choose and timing it proves nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
