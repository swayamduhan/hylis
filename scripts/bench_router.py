#!/usr/bin/env python3
"""Four vector indexes on the same corpus, measured the same way.

    python scripts/bench_router.py                  # SIFT10K
    python scripts/bench_router.py --sift1m         # the scale that matters
    python scripts/bench_router.py --random 50000x64

  flat      exhaustive scan, exact by construction -- the oracle
  hnswlib   the reference implementation, a benchmark baseline only
  hnsw      ours, hierarchy descent
  routed    ours, same graph, entry points from the neural router

The controlled comparison is `hnsw` against `routed`: one graph, one seed,
searched two ways, so the only variable is routing. `flat_only` is reported
separately because it changes two things at once (no hierarchy *and* a router),
and its value is the memory it saves rather than the speed.

Trains a router first if one has not been saved for this corpus.
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
    import hylis._hnswlib as hnswlib_module
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

EF_SWEEP = [10, 20, 40, 80, 160]


def require_optimised() -> None:
    import hylis._hnsw as h

    if not getattr(h, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(h, '__build_type__', '?')} -- timings would be "
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


def load_corpus(args):
    if args.random:
        n, _, dim = args.random.partition("x")
        v = ds.random_vectors(n=int(n), dim=int(dim or 64), n_queries=200, seed=0,
                              n_clusters=max(2, int(n) // 100))
        return v, f"random-{v.n}x{v.dim}"
    variant = "sift" if args.sift1m else "siftsmall"
    try:
        return ds.load_sift(variant), variant
    except FileNotFoundError as exc:
        if args.sift1m:
            raise SystemExit(f"{exc}\n    python scripts/fetch_data.py sift")
        v = ds.random_vectors(n=20000, dim=64, n_queries=200, seed=0, n_clusters=200)
        return v, f"random-{v.n}x{v.dim}"


def get_router(corpus, label, args):
    """Load a saved router for this corpus, or train one now."""
    path = Path(args.router or (ds.default_data_dir() / f"router-{label}.json"))
    if path.exists() and not args.retrain:
        print(f"  using the router at {path}")
        return NeuralRouter.load(str(path))

    print(f"  no saved router for {label}; training one "
          f"({args.clusters} clusters)")
    km = kmeans(corpus.base, n_clusters=args.clusters, seed=0)
    exact = None
    if args.samples * corpus.n <= 2e9:
        exact = FlatIndex(corpus.dim)
        exact.add_batch(corpus.base)
    ts = build_training_set(corpus.base, km.assignment, n_samples=args.samples,
                            seed=0, exact_index=exact)
    model = RouterMLP(dim=corpus.dim, clusters=km.centroids.shape[0],
                      hidden=args.hidden, seed=0)
    model.fit(ts.x_train, ts.y_train, epochs=args.epochs)
    blob = router_json(model, km.medoids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(blob, encoding="utf-8")
    print(f"  trained and saved to {path}")
    return NeuralRouter.from_json(blob)


def timed(fn, *fn_args):
    start = time.perf_counter()
    fn(*fn_args)
    return time.perf_counter() - start


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sift1m", action="store_true")
    parser.add_argument("--random", metavar="NxDIM")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--router", type=str, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--queries", type=int, default=0,
                        help="cap the query set; 0 uses all of it. Worth "
                             "setting on SIFT1M, where an exhaustive scan of "
                             "all 10,000 queries costs several minutes on its "
                             "own and the reference line does not need that "
                             "many to be stable.")
    args = parser.parse_args(argv)

    require_optimised()
    corpus, label = load_corpus(args)
    k = args.k

    if args.queries and args.queries < corpus.n_queries:
        corpus = ds.VectorDataset(
            corpus.base, corpus.queries[: args.queries],
            None if corpus.ground_truth is None else corpus.ground_truth[: args.queries],
            corpus.name, corpus.metric)

    print(f"Four indexes on {label}: {corpus.n:,} x {corpus.dim}-d, "
          f"{corpus.n_queries} queries, k={k}\n")

    truth = corpus.ground_truth
    if truth is None or truth.shape[1] < k:
        truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=k)
    truth = truth[:, :k]

    # -- build ------------------------------------------------------------
    flat = FlatIndex(corpus.dim)
    flat.reserve(corpus.n)
    flat_build = timed(flat.add_batch, corpus.base)

    graph = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    graph.reserve(corpus.n)
    graph_build = timed(graph.add_batch, corpus.base)

    router = get_router(corpus, label, args)
    graph.set_router(router)

    lib = None
    lib_build = 0.0
    if hnswlib_module.available:
        lib = hnswlib_module.HnswlibIndex(corpus.dim, Metric.L2, capacity=corpus.n,
                                          M=16, ef_construction=200)
        lib_build = timed(lib.add_batch, corpus.base)

    flat_only = HnswIndex(corpus.dim, Metric.L2, 16, 200, 100, True)
    flat_only.reserve(corpus.n)
    flat_only_build = timed(flat_only.add_batch, corpus.base)
    flat_only.set_router(router)

    stats = graph.stats()
    flat_stats = flat_only.stats()
    print()
    print(f"build   flat {flat_build*1000:7.0f} ms   hnsw {graph_build*1000:7.0f} ms"
          f"   hnswlib {lib_build*1000:7.0f} ms")
    print(f"graph   {human_bytes(stats.graph_bytes)} over "
          f"{human_bytes(stats.total_bytes - stats.graph_bytes)} of vectors; "
          f"{stats.levels} levels {stats.layer_population}")
    print(f"        layer-0-only build: {human_bytes(flat_stats.graph_bytes)} "
          f"({(stats.graph_bytes - flat_stats.graph_bytes) / max(stats.graph_bytes,1):.1%}"
          f" of the graph was the hierarchy)")
    print()

    # -- recall / throughput ---------------------------------------------
    header = (f"{'index':<10}{'ef':>5}{'recall@' + str(k):>12}{'QPS':>11}"
              f"{'visited':>10}{'routing':>9}")
    print(header)
    print("-" * len(header))

    flat_seconds = timed(flat.search_batch, corpus.queries, k)
    flat_qps = corpus.n_queries / flat_seconds
    print(f"{'flat':<10}{'-':>5}{1.0:>12.4f}{flat_qps:>11,.0f}"
          f"{corpus.n:>10,}{'-':>9}")

    def sweep(name, run, visited_of=None):
        for ef in EF_SWEEP:
            seconds = timed(run, ef)
            ids = run(ef)
            recall = ds.recall_at_k(ids, truth, k=k)
            visited = routing = "-"
            if visited_of is not None:
                v, r = visited_of(ef)
                visited, routing = f"{v:,.0f}", f"{r:,.0f}"
            print(f"{name:<10}{ef:>5}{recall:>12.4f}"
                  f"{corpus.n_queries/seconds:>11,.0f}{visited:>10}{routing:>9}")

    if lib is not None:
        sweep("hnswlib", lambda ef: lib.search_batch(corpus.queries, k=k, ef=ef)[0])

    def graph_visits(ef, use_router):
        v, r = [], []
        for q in corpus.queries[: min(50, corpus.n_queries)]:
            graph.search(q, k, ef, use_router)
            v.append(graph.last_visited)
            r.append(graph.last_routing_visited)
        return float(np.mean(v)), float(np.mean(r))

    sweep("hnsw",
          lambda ef: graph.search_batch(corpus.queries, k=k, ef=ef, use_router=False)[0],
          lambda ef: graph_visits(ef, False))
    sweep("routed",
          lambda ef: graph.search_batch(corpus.queries, k=k, ef=ef, use_router=True)[0],
          lambda ef: graph_visits(ef, True))
    sweep("flat-only",
          lambda ef: flat_only.search_batch(corpus.queries, k=k, ef=ef,
                                            use_router=True)[0])

    print()
    print("Reading this table:")
    print("  * 'routing' is the graph nodes spent just reaching layer 0. The")
    print("    router shows 0 there because its cost is arithmetic, not")
    print("    traversal -- but note how small the descent's figure already is.")
    print("  * hnsw vs routed is the controlled comparison: same graph, same")
    print("    seed, one variable. Any difference is routing and nothing else.")
    print("  * flat-only has no hierarchy at all, so it changes two things at")
    print("    once; its result is the memory line above, not the speed.")
    print("  * hnswlib is here to show whether our HNSW is competitive. A")
    print("    router that beats a slow baseline would have proved nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
