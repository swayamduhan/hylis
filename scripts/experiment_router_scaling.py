#!/usr/bin/env python3
"""Does the neural router's usefulness grow with N?

    python scripts/experiment_router_scaling.py               # SIFT1M subsets
    python scripts/experiment_router_scaling.py --quick
    python scripts/experiment_router_scaling.py --random 200000x32

Module 6 measured +31% QPS at ef=10 on SIFT1M against roughly parity at
SIFT10K, which suggests the benefit scales. Two subsets is not a trend, so
this sweeps n across a log-spaced range on nested prefixes of one corpus --
each n a subset of the next, so nothing changes but the size.

Compared at matched recall, not matched ef
------------------------------------------
QPS at a fixed ef is not comparable across n: a beam of 20 explores a smaller
fraction of a larger graph, so the same ef sits at a different point on the
recall curve. Each method's ef is therefore searched for the value that hits a
target recall, and throughput is compared there. This is the ann-benchmarks
convention, and the reason the module 6 table is only readable because it
prints recall alongside.

Separating mechanism from outcome
---------------------------------
Two explanations were available for module 6's result, and the table reports
both so the trend is explained rather than merely observed:

  descent    nodes the hierarchy costs to reach layer 0. Measured at only ~4
             even at a million vectors, so "the router eliminates traversal"
             was already the wrong story -- there is almost nothing to
             eliminate. If the QPS gain grows while this stays flat, the gain
             is entry-point *quality*.

  top_p=1    the router seeded with one entry point instead of two. Seeding
             the beam from two places helps on its own, whatever the weights
             say, so quoting top_p=2 against the descent would credit the
             network with an effect it did not produce. scripts/
             train_router.py already splits this; every row here carries it.
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
    from hylis import FlatIndex, HnswIndex, Metric
    from hylis import datasets as ds
    from hylis._hnsw import NeuralRouter
    from hylis.router import RouterMLP, build_training_set, kmeans, router_json
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

SIZES = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
EF_CANDIDATES = [10, 12, 16, 20, 28, 40, 56, 80, 112, 160, 224, 320]


def require_optimised() -> None:
    import hylis._hnsw as h

    if not getattr(h, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(h, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def load_corpus(args):
    if args.random:
        n, _, dim = args.random.partition("x")
        v = ds.random_vectors(n=int(n), dim=int(dim or 32), n_queries=args.queries,
                              seed=0, n_clusters=max(2, int(n) // 200))
        return v, f"random-{v.n}x{v.dim}"
    try:
        return ds.load_sift("sift"), "sift1m"
    except FileNotFoundError:
        print("SIFT1M not present; using a synthetic corpus instead.")
        print("    python scripts/fetch_data.py sift\n")
        v = ds.random_vectors(n=200_000, dim=32, n_queries=args.queries, seed=0,
                              n_clusters=1000)
        return v, f"random-{v.n}x{v.dim}"


def recall_at_ef(graph, queries, truth, k, ef, use_router):
    ids = graph.search_batch(queries, k=k, ef=ef, use_router=use_router)[0]
    return ds.recall_at_k(ids, truth, k=k)


def qps_at_ef(graph, queries, k, ef, use_router, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        graph.search_batch(queries, k=k, ef=ef, use_router=use_router)
        best = min(best, time.perf_counter() - start)
    return len(queries) / best


def ef_for_recall(graph, queries, truth, k, target, use_router):
    """Smallest ef from the ladder that reaches `target` recall.

    A ladder rather than a bisection: ef is an integer knob with a coarse,
    monotone effect, and the search is over a dozen values -- bisecting would
    add machinery to save two evaluations. Returns None if even the widest
    beam falls short, which is reported rather than silently clamped.
    """
    for ef in EF_CANDIDATES:
        if recall_at_ef(graph, queries, truth, k, ef, use_router) >= target:
            return ef
    return None


def train_router_for(base, clusters, seed, epochs, samples):
    km = kmeans(base, n_clusters=clusters, seed=seed)
    ts = build_training_set(base, km.assignment, n_samples=samples, seed=seed)
    model = RouterMLP(dim=base.shape[1], clusters=km.centroids.shape[0],
                      hidden=64, seed=seed)
    model.fit(ts.x_train, ts.y_train, epochs=epochs)
    return NeuralRouter.from_json(router_json(model, km.medoids, vectors=base))


def descent_nodes(graph, queries, use_router, sample=50):
    """Graph nodes spent just getting to layer 0."""
    spent = []
    for q in queries[:sample]:
        graph.search(q, 10, 64, use_router)
        spent.append(graph.last_routing_visited)
    return float(np.mean(spent)) if spent else 0.0


def run_one(corpus, n, args, target):
    base = np.ascontiguousarray(corpus.base[:n])
    queries = corpus.queries[: args.queries]

    truth = ds.compute_ground_truth(base, queries, k=args.k)

    graph = HnswIndex(base.shape[1], Metric.L2, M=16, ef_construction=200)
    graph.reserve(n)
    graph.add_batch(base)

    clusters = args.clusters if args.clusters else max(
        16, int(round(np.sqrt(n))))
    router = train_router_for(base, clusters, args.seed, args.epochs,
                              min(args.samples, n * 2))
    graph.set_router(router)

    rows = {}
    for label, use_router, top_p in (("descent", False, 2),
                                     ("routed", True, 2),
                                     ("routed_p1", True, 1)):
        graph.router_top_p = top_p
        ef = ef_for_recall(graph, queries, truth, args.k, target, use_router)
        if ef is None:
            rows[label] = None
            continue
        rows[label] = {
            "ef": ef,
            "qps": qps_at_ef(graph, queries, args.k, ef, use_router),
            "recall": recall_at_ef(graph, queries, truth, args.k, ef, use_router),
        }
    graph.router_top_p = 2

    return {
        "n": n,
        "clusters": clusters,
        "descent_nodes": descent_nodes(graph, queries, False),
        "levels": graph.stats().levels,
        "rows": rows,
    }


def report(results, target, args) -> None:
    print(f"\n{'=' * 84}")
    print(f"Matched recall@{args.k} >= {target:.2f}. "
          f"'gain' is routed QPS over descent QPS at that recall.")
    print("=" * 84)
    header = (f"{'n':>10}{'C':>7}{'lvls':>6}{'descent':>9}"
              f"{'ef_d':>6}{'ef_r':>6}{'QPS desc':>11}{'QPS rout':>11}"
              f"{'gain':>8}{'p1 gain':>9}")
    print(header)
    print("-" * len(header))

    for r in results:
        d = r["rows"]["descent"]
        rt = r["rows"]["routed"]
        p1 = r["rows"]["routed_p1"]
        if d is None or rt is None:
            print(f"{r['n']:>10,}{r['clusters']:>7}{r['levels']:>6}"
                  f"{r['descent_nodes']:>9.1f}"
                  f"{'--':>6}{'--':>6}{'target recall unreachable':>39}")
            continue
        gain = rt["qps"] / d["qps"]
        p1_gain = (p1["qps"] / d["qps"]) if p1 else float("nan")
        print(f"{r['n']:>10,}{r['clusters']:>7}{r['levels']:>6}"
              f"{r['descent_nodes']:>9.1f}{d['ef']:>6}{rt['ef']:>6}"
              f"{d['qps']:>11,.0f}{rt['qps']:>11,.0f}"
              f"{gain:>7.2f}x{p1_gain:>8.2f}x")

    print("\nReading this table:")
    print("  * 'descent' is nodes spent reaching layer 0. If it stays flat")
    print("    while 'gain' rises, the router is not saving traversal -- it is")
    print("    starting the beam somewhere better. Those are different claims")
    print("    and only one of them is supported by this column.")
    print("  * 'p1 gain' is the router restricted to one entry point. The gap")
    print("    between it and 'gain' is what the second entry point bought,")
    print("    which the network did not earn. Quote 'p1 gain' when claiming")
    print("    a result for learned routing specifically.")
    print("  * 'C' is the cluster count. With --clusters unset it scales as")
    print("    sqrt(n), so the partition keeps pace with the corpus; pin it")
    print("    to hold it fixed and see the confound.")

    usable = [r for r in results
              if r["rows"]["descent"] and r["rows"]["routed"]]

    # If every row landed on the smallest ef the ladder offers, the target
    # recall was reachable without widening the beam at any size -- so no row
    # is actually at its matched-recall operating point and the gains are not
    # comparable across n. Worth refusing to draw a trend from.
    floor = EF_CANDIDATES[0]
    at_floor = sum(1 for r in usable
                   if r["rows"]["descent"]["ef"] == floor
                   and r["rows"]["routed"]["ef"] == floor)
    if usable and at_floor == len(usable):
        print(f"\n  WARNING: every row matched recall at ef={floor}, the bottom")
        print(f"  of the ladder. recall@{args.k} >= {target:.2f} is too easy for this")
        print(f"  corpus, so nothing here is a matched-recall comparison and the")
        print(f"  gain column is not a trend. Re-run with a higher --recall, or")
        print(f"  on a corpus where the target actually costs something.")
        return

    if len(usable) < 2:
        return

    first, last = usable[0], usable[-1]
    g0 = first["rows"]["routed"]["qps"] / first["rows"]["descent"]["qps"]
    g1 = last["rows"]["routed"]["qps"] / last["rows"]["descent"]["qps"]
    print(f"\n  {first['n']:,} -> {last['n']:,}: gain {g0:.2f}x -> {g1:.2f}x, "
          f"descent {first['descent_nodes']:.1f} -> {last['descent_nodes']:.1f} nodes")

    if g1 > g0 * 1.05:
        print("  The benefit grows with n over this range.")
    elif g1 < g0 * 0.95:
        print("  The benefit SHRINKS with n. Reported as measured.")
    else:
        print("  The benefit is flat in n over this range. Reported as measured.")

    spread = max(r["descent_nodes"] for r in usable) - \
        min(r["descent_nodes"] for r in usable)
    if spread < 2.0:
        print(f"  Descent cost barely moved ({spread:.1f} nodes across the whole")
        print("  range), so whatever the gain column does, it is not explained")
        print("  by traversal being eliminated.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--random", metavar="NxDIM")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200,
                        help="query set size. Exhaustive ground truth is "
                             "computed per n, so this is the main cost knob.")
    parser.add_argument("--recall", type=float, default=0.90,
                        help="the recall each method's ef is matched to")
    parser.add_argument("--sizes", type=str, default="",
                        help="comma-separated; default log-spaced to 1M")
    parser.add_argument("--clusters", type=int, default=0,
                        help="fixed cluster count; default scales as sqrt(n)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    require_optimised()
    corpus, label = load_corpus(args)

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    elif args.quick:
        sizes = [5_000, 10_000, 20_000, 40_000]
    else:
        sizes = SIZES
    sizes = [n for n in sizes if n <= corpus.n]
    if not sizes:
        raise SystemExit(f"{label} holds only {corpus.n:,} vectors")

    print(f"Router scaling on {label}: {corpus.dim}-d, "
          f"{min(args.queries, corpus.n_queries)} queries, k={args.k}")
    print(f"n = {', '.join(f'{n:,}' for n in sizes)}")

    results = []
    for n in sizes:
        print(f"  n={n:,} ...", end="", flush=True)
        start = time.perf_counter()
        results.append(run_one(corpus, n, args, args.recall))
        print(f" {time.perf_counter() - start:.0f}s")

    report(results, args.recall, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
