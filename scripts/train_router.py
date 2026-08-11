#!/usr/bin/env python3
"""Train the neural router. One command, nothing to collect by hand.

    python scripts/train_router.py                    # SIFT10K if present
    python scripts/train_router.py --sift             # explicit
    python scripts/train_router.py --random 20000x64
    python scripts/train_router.py --clusters 512 --epochs 40

What training actually involves
-------------------------------
There is no dataset to gather and nothing to label. Everything is derived from
the corpus itself:

  1. k-means partitions the base vectors into C clusters. Each cluster's
     *medoid* -- the real node nearest its centroid -- becomes a candidate
     entry point, because a centroid is an average and usually not a stored
     vector, while a graph search has to start at an actual node.

  2. Training inputs are base vectors with Gaussian jitter. A router trained
     only on exact stored vectors would never have seen a point that is not
     already in the corpus, which is every real query.

  3. Labels are the cluster each sample came from. Self-supervised, seeded,
     reproducible.

  4. A 1-hidden-layer MLP learns query -> cluster. Weights are exported to
     JSON and loaded by the C++ index, because inference has to be in C++: a
     query costs tens of microseconds and a Python callback per query would
     cost more than the query does.

What "did it work" means
------------------------
Not classification accuracy. A router can pick the "wrong" cluster and still
land the beam somewhere excellent, or pick the right one and land it badly.
So the number that decides is end-to-end recall of the whole search, measured
against FlatIndex. Accuracy is printed as a diagnostic only.

The script also reports two opponents worth beating: the hierarchy descent it
replaces, and nearest-centroid routing, which needs no network at all.
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
    from hylis.router import (
        NearestCentroidRouter,
        RouterMLP,
        build_training_set,
        export_router,
        kmeans,
        router_json,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )


def load_corpus(args):
    if args.random:
        n, _, dim = args.random.partition("x")
        v = ds.random_vectors(n=int(n), dim=int(dim or 64), n_queries=200, seed=0,
                              n_clusters=max(2, int(n) // 100))
        return v, f"random {v.n}x{v.dim}"
    variant = "sift" if args.sift1m else "siftsmall"
    try:
        s = ds.load_sift(variant)
        return s, variant
    except FileNotFoundError as exc:
        if args.sift or args.sift1m:
            raise SystemExit(str(exc))
        print(f"  ({exc.args[0].splitlines()[0]}; falling back to synthetic)")
        v = ds.random_vectors(n=20000, dim=64, n_queries=200, seed=0, n_clusters=200)
        return v, f"random {v.n}x{v.dim}"


def recall_of(index, corpus, truth, k, ef, use_router):
    ids, _ = index.search_batch(corpus.queries, k=k, ef=ef, use_router=use_router)
    return ds.recall_at_k(ids, truth[:, :k], k=k)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sift", action="store_true", help="use SIFT10K")
    parser.add_argument("--sift1m", action="store_true", help="use SIFT1M")
    parser.add_argument("--random", metavar="NxDIM")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=20,
                        help="beam width for the end-to-end check; the entry "
                             "point matters most when the beam is narrow")
    parser.add_argument("--out", type=str, default=None,
                        help="where to write the weights (default data/router.json)")
    parser.add_argument("--cheap-labels", action="store_true",
                        help="label by source cluster instead of exact nearest "
                             "neighbour; faster, but the target then nearly "
                             "coincides with nearest-centroid")
    parser.add_argument("--exact-label-budget", type=float, default=2e9,
                        help="distance-computation budget above which labelling "
                             "falls back to the cheap target")
    args = parser.parse_args(argv)

    corpus, label = load_corpus(args)
    out_path = args.out or str(ds.default_data_dir() / f"router-{label}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Training a router for {label}: {corpus.n:,} x {corpus.dim}-d\n")

    # 1. Partition ---------------------------------------------------------
    print(f"[1/4] k-means, {args.clusters} clusters")
    km = kmeans(corpus.base, n_clusters=args.clusters, seed=args.seed)
    sizes = np.bincount(km.assignment, minlength=km.centroids.shape[0])
    print(f"      {km.iterations} iterations in {km.seconds:.1f}s")
    print(f"      cluster sizes: min {sizes.min():,}  median "
          f"{int(np.median(sizes)):,}  max {sizes.max():,}")
    print(f"      {int((sizes == 0).sum())} empty, "
          f"{len(set(km.medoids.tolist()))} distinct medoids\n")

    # 2. Training data -----------------------------------------------------
    print(f"[2/4] building {args.samples:,} training samples from the corpus")
    exact_index = None
    cost = args.samples * corpus.n
    if not args.cheap_labels and cost <= args.exact_label_budget:
        print(f"      labelling by exact nearest neighbour "
              f"({cost/1e9:.1f}G distance computations)")
        exact_index = FlatIndex(corpus.dim)
        exact_index.reserve(corpus.n)
        exact_index.add_batch(corpus.base)
    else:
        print("      labelling by source cluster (exact search too expensive here)")
        print("      NOTE: this target is very close to nearest-centroid, so the")
        print("            network can at best tie the free baseline")

    start = time.perf_counter()
    ts = build_training_set(corpus.base, km.assignment, n_samples=args.samples,
                            jitter=args.jitter, seed=args.seed,
                            exact_index=exact_index)
    print(f"      {len(ts.x_train):,} train / {len(ts.x_val):,} validation, "
          f"jitter {args.jitter}, built in {time.perf_counter()-start:.1f}s\n")

    # 3. Train -------------------------------------------------------------
    print(f"[3/4] training a {corpus.dim} -> {args.hidden} -> {args.clusters} MLP "
          f"for {args.epochs} epochs")
    start = time.perf_counter()
    model = RouterMLP(dim=corpus.dim, clusters=km.centroids.shape[0],
                      hidden=args.hidden, seed=args.seed)
    model.fit(ts.x_train, ts.y_train, epochs=args.epochs)
    train_seconds = time.perf_counter() - start

    accuracy = float((model.predict(ts.x_val) == ts.y_val).mean())
    baseline = NearestCentroidRouter(km.centroids)
    baseline_accuracy = float((baseline.predict(ts.x_val) == ts.y_val).mean())
    print(f"      loss {model.losses[0]:.4f} -> {model.losses[-1]:.4f} "
          f"in {train_seconds:.1f}s")
    print(f"      validation accuracy   MLP {accuracy:.4f}   "
          f"nearest-centroid {baseline_accuracy:.4f}")
    print("      (accuracy is a diagnostic; recall below is what decides)\n")

    # 4. End-to-end --------------------------------------------------------
    print(f"[4/4] end-to-end recall@{args.k} at ef={args.ef}, "
          f"against the exact index")
    truth = corpus.ground_truth
    if truth is None or truth.shape[1] < args.k:
        truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=args.k)

    graph = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    graph.reserve(corpus.n)
    graph.add_batch(corpus.base)

    descent = recall_of(graph, corpus, truth, args.k, args.ef, False)

    untrained = RouterMLP(dim=corpus.dim, clusters=km.centroids.shape[0],
                          hidden=args.hidden, seed=args.seed + 1)
    untrained.mean, untrained.scale = model.mean, model.scale
    graph.set_router(NeuralRouter.from_json(router_json(untrained, km.medoids)))
    random_weights = recall_of(graph, corpus, truth, args.k, args.ef, True)

    graph.set_router(NeuralRouter.from_json(router_json(model, km.medoids)))
    trained_recall = recall_of(graph, corpus, truth, args.k, args.ef, True)

    graph.set_router(NeuralRouter.from_json(router_json(model, km.medoids)))
    graph.router_top_p = 1
    trained_single = recall_of(graph, corpus, truth, args.k, args.ef, True)
    graph.router_top_p = 2

    print(f"      {'hierarchy descent (baseline)':<34}{descent:.4f}")
    print(f"      {'router, random weights, top_p=2':<34}{random_weights:.4f}")
    print(f"      {'router, trained, top_p=1':<34}{trained_single:.4f}")
    print(f"      {'router, trained, top_p=2':<34}{trained_recall:.4f}")

    # Splitting the effect matters: seeding the beam from two entry points
    # instead of one helps on its own, whatever the weights are. Quoting the
    # total against the descent as a result for "the neural router" would
    # silently credit the network with that too.
    entry_effect = random_weights - descent
    learning_effect = trained_recall - random_weights
    print()
    print("  Splitting the effect:")
    print(f"    2 entry points instead of 1, weights aside : {entry_effect:+.4f}")
    print(f"    what the trained network adds on top       : {learning_effect:+.4f}")
    print(f"    total against the descent                  : "
          f"{trained_recall - descent:+.4f}")
    print()

    if learning_effect <= 0.0:
        print("  The trained weights are no better than random ones, so any gain")
        print("  against the descent is the extra entry point rather than")
        print("  learning. Report it that way; check the loss curve above.")
    elif learning_effect < entry_effect - 1e-9:
        print("  Most of the gain comes from the extra entry point, not the")
        print("  network. Quoting the total as a result for the neural router")
        print("  alone would overstate it.")
    elif abs(learning_effect - entry_effect) <= 1e-9:
        print("  The network and the extra entry point contribute about equally.")
        print("  Half of the headline gain is therefore not attributable to")
        print("  learning at all, which is worth stating explicitly.")
    else:
        print("  The network contributes more than the extra entry point does,")
        print("  which is the claim worth making for the router.")

    if exact_index is None:
        print()
        print("  Caveat on the accuracy line above: with cheap labels the target")
        print("  is essentially k-means assignment, which is what")
        print("  nearest-centroid computes by definition -- so it is being")
        print("  graded on its own answer and beating it is not possible. Use")
        print("  exact labels (the default where affordable) for a fair test.")

    export_router(model, km.medoids, out_path, vectors=corpus.base)
    print(f"\n  weights written to {out_path}")
    print(f"  load them with: NeuralRouter.load({out_path!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
