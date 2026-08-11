#!/usr/bin/env python3
"""Exhaustive scan versus HNSW: recall/QPS curve and the filtered crossover.

    python scripts/bench_vector.py                  # SIFT10K if present, else synthetic
    python scripts/bench_vector.py --random 20000x64
    python scripts/bench_vector.py --skip-filtered

Two results, and the second is the one the query planner needs.

1. The recall/QPS curve. This is how the ANN field reports results
   (ann-benchmarks convention): sweep ef, and for each point plot the recall
   achieved against the throughput it cost. FlatIndex sits on the chart as a
   fixed reference — recall 1.0 by definition, at whatever QPS an exhaustive
   scan manages.

2. The filtered crossover. Filtered brute force is O(|allowed|) and exact, so
   it gets *cheaper* as a predicate tightens. Filtered HNSW must traverse
   non-matching nodes to stay connected, so it gets *more* expensive. Those
   curves cross, and the selectivity at which they do is exactly what the
   module 8 planner has to predict in order to pick a plan.
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

EF_SWEEP = [10, 20, 40, 80, 160, 320]
SELECTIVITIES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


def require_optimised() -> None:
    import hylis._hnsw as hnsw

    if not getattr(hnsw, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(hnsw, '__build_type__', '?')} -- these timings "
            "would be several times off and meaningless.\n"
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
        v = ds.random_vectors(n=int(n), dim=int(dim or 64), n_queries=200,
                              seed=0, n_clusters=max(2, int(n) // 100))
        return v, f"random {v.n}x{v.dim} (clustered)"
    try:
        s = ds.load_sift("siftsmall")
        return s, "SIFT10K"
    except FileNotFoundError:
        v = ds.random_vectors(n=20000, dim=64, n_queries=200, seed=0, n_clusters=200)
        return v, f"random {v.n}x{v.dim} (clustered; run fetch_data.py for SIFT)"


def timed_search(index, queries, k, ef=None):
    start = time.perf_counter()
    if ef is None:
        index.search_batch(queries, k=k)
    else:
        index.search_batch(queries, k=k, ef=ef)
    return time.perf_counter() - start


def recall_curve(corpus, label, k):
    print(f"\n=== Recall / throughput on {label} "
          f"({corpus.n:,} x {corpus.dim}-d, {corpus.n_queries} queries, k={k}) ===\n")

    flat = FlatIndex(corpus.dim)
    flat.reserve(corpus.n)
    start = time.perf_counter()
    flat.add_batch(corpus.base)
    flat_build = time.perf_counter() - start

    truth = corpus.ground_truth
    if truth is None or truth.shape[1] < k:
        truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=k)

    start = time.perf_counter()
    graph = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    graph.reserve(corpus.n)
    graph.add_batch(corpus.base)
    graph_build = time.perf_counter() - start
    graph.validate()

    stats = graph.stats()
    print(f"build:   flat {flat_build*1000:7.0f} ms      hnsw {graph_build*1000:7.0f} ms")
    print(f"memory:  vectors {human_bytes(stats.total_bytes - stats.graph_bytes)}"
          f"      graph {human_bytes(stats.graph_bytes)}"
          f"  ({stats.graph_bytes / max(stats.total_bytes - stats.graph_bytes, 1):.0%} overhead)")
    print(f"levels:  {stats.levels}   layer population {stats.layer_population}")
    print(f"         mean degree at layer 0 = {stats.mean_degree_l0:.1f}, "
          f"{stats.reachable:,}/{stats.nodes:,} nodes reachable")
    print()

    flat_seconds = timed_search(flat, corpus.queries, k)
    flat_qps = corpus.n_queries / flat_seconds
    print(f"{'index':<10}{'ef':>6}{'recall@' + str(k):>12}{'QPS':>12}"
          f"{'visited':>10}{'speedup':>10}")
    print("-" * 60)
    print(f"{'flat':<10}{'-':>6}{1.0:>12.4f}{flat_qps:>12,.0f}"
          f"{corpus.n:>10,}{'1.00x':>10}")

    best = None
    for ef in EF_SWEEP:
        seconds = timed_search(graph, corpus.queries, k, ef)
        ids, _ = graph.search_batch(corpus.queries, k=k, ef=ef)
        recall = ds.recall_at_k(ids, truth[:, :k], k=k)

        visited = []
        for query in corpus.queries[: min(50, corpus.n_queries)]:
            graph.search(query, k=k, ef=ef)
            visited.append(graph.last_visited)

        qps = corpus.n_queries / seconds
        print(f"{'hnsw':<10}{ef:>6}{recall:>12.4f}{qps:>12,.0f}"
              f"{np.mean(visited):>10,.0f}{qps / flat_qps:>9.1f}x")
        if best is None and recall >= 0.95:
            best = (ef, recall, qps / flat_qps)

    print()
    if best:
        print(f"First ef clearing 0.95 recall: ef={best[0]} "
              f"(recall {best[1]:.4f}) at {best[2]:.1f}x the throughput of "
              f"an exhaustive scan.")
    else:
        print("No ef in the sweep reached 0.95 recall -- widen EF_SWEEP or "
              "check the graph.")
    return flat, graph, truth


def filtered_crossover(corpus, flat, graph, k):
    print(f"\n=== Filtered search: where exhaustive beats the graph ===\n")

    rng = np.random.default_rng(0)
    attribute = rng.uniform(0, 1000, size=corpus.n)
    queries = corpus.queries[: min(50, corpus.n_queries)]

    print(f"{'selectivity':>12}{'allowed':>10}{'flat us':>10}{'hnsw us':>10}"
          f"{'hnsw visited':>14}{'recall':>9}{'winner':>10}")
    print("-" * 76)

    crossover = None
    previous_winner = None
    for selectivity in SELECTIVITIES:
        threshold = ds.threshold_for_selectivity(attribute, selectivity)
        allowed = np.flatnonzero(attribute < threshold)
        if len(allowed) < k:
            continue
        allowed_list = allowed.tolist()

        start = time.perf_counter()
        exact_results = [flat.search_filtered(q, k, allowed_list) for q in queries]
        flat_us = (time.perf_counter() - start) / len(queries) * 1e6

        start = time.perf_counter()
        graph_results = [graph.search_filtered(q, k, allowed_list, 100) for q in queries]
        graph_us = (time.perf_counter() - start) / len(queries) * 1e6

        visited = []
        for q in queries[:10]:
            graph.search_filtered(q, k, allowed_list, 100)
            visited.append(graph.last_visited)

        hits = 0
        for approx, exact in zip(graph_results, exact_results):
            truth_ids = {n.id for n in exact}
            hits += sum(1 for n in approx if n.id in truth_ids)
        recall = hits / max(sum(len(r) for r in exact_results), 1)

        winner = "flat" if flat_us <= graph_us else "hnsw"
        if previous_winner == "flat" and winner == "hnsw":
            crossover = selectivity
        previous_winner = winner

        print(f"{selectivity:>11.1%}{len(allowed):>10,}{flat_us:>10.0f}"
              f"{graph_us:>10.0f}{np.mean(visited):>14,.0f}{recall:>9.3f}"
              f"{winner:>10}")

    print()
    if crossover is not None:
        print(f"Crossover at roughly {crossover:.1%} selectivity: below it the "
              f"exhaustive scan wins, above it the graph does.")
    else:
        print("No crossover inside the sweep -- one plan won throughout.")
    print()
    print("Why this happens:")
    print("  * Filtered brute force costs O(|allowed|) and is exact, so a")
    print("    tighter predicate makes it strictly cheaper.")
    print("  * The graph must step *through* non-matching nodes to stay")
    print("    connected, so a tighter predicate makes it visit more, and its")
    print("    recall falls as the beam runs out of surviving candidates.")
    print("  * That is the decision the module 8 planner has to make per query,")
    print("    and this table is the data it needs to make it.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--random", metavar="NxDIM", help="use synthetic vectors")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--skip-filtered", action="store_true")
    args = parser.parse_args(argv)

    require_optimised()
    corpus, label = load_corpus(args)
    flat, graph, _ = recall_curve(corpus, label, args.k)
    if not args.skip_filtered:
        filtered_crossover(corpus, flat, graph, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
