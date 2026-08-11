#!/usr/bin/env python3
"""Interactive playground for the flat (brute-force) vector index.

    python scripts/try_vectors.py --demo            # scripted walkthrough
    python scripts/try_vectors.py --sift            # load SIFT10K and poke at it
    python scripts/try_vectors.py --random 5000x64
    python scripts/try_vectors.py --npy myvectors.npy --metric cosine

Type `help` at the prompt for the command list.

`check` re-runs any search through the numpy oracle in hylis.datasets and
diffs the two. Exhaustive search has one right answer, so a disagreement is
always a bug -- there is no tolerance to hide behind.
"""

from __future__ import annotations

import argparse
import os
import shlex
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
except ImportError as exc:  # pragma: no cover - depends on the build having run
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "Build the C++ extensions first:\n"
        "    cmake --preset default && cmake --build build"
    )


METRICS = {"l2": Metric.L2, "ip": Metric.InnerProduct, "cosine": Metric.Cosine}
METRIC_NAMES = {v: k for k, v in METRICS.items()}


def warn_if_unoptimised() -> None:
    """Timings from a Debug build are off by ~4x -- more than most of the
    speedups this project measures -- so say so loudly rather than let a
    meaningless number end up in a report."""
    import hylis._flat as _flat

    if not getattr(_flat, "__optimized__", True):
        print(f"  !! built as {getattr(_flat, '__build_type__', '?')} -- these")
        print("  !! timings are NOT representative. Rebuild for benchmarks:")
        print("  !!     cmake --preset release && cmake --build build")

HELP = """
  load data
    random <n> [dim]        n random clustered vectors
    sift [n]                SIFT10K corpus (needs scripts/fetch_data.py siftsmall)
    npy <path>              a .npy array of shape (n, dim)
    fvecs <path>            a .fvecs file (SIFT/TEXMEX format)
    csv <path>              CSV, one vector per row, all columns numeric
    add <v1> <v2> ...       type a single vector in by hand

  query
    knn <id> [k]            neighbours of a stored vector
    search <k> <v1> <v2>... neighbours of a typed query
    query <i> [k]           use query vector i from the loaded dataset
    filter <sel> [k] [i]    filtered search keeping a `sel` fraction (0-1)

  index selection  (all stay built; the exact scan is everyone's oracle)
    index flat|hnsw|hnswlib|routed
                            which index answers queries
    hnsw [M] [efC]          (re)build our graph
    hnswlib [M] [efC]       build the reference implementation
    router [path]           load a trained router, or train one now
    ef <n>                  beam width; 0 for the index default
    compare [k] [nq]        every index, same queries: recall vs speed

  verify and measure
    check [k]               diff every dataset query against the numpy oracle
    truth [k]               compare against the corpus's *published* ground truth
    bench [nq] [k]          time the scan, and pre- vs post-filter plans
    stats / metric <m> / clear / help / quit
"""


KINDS = ("flat", "hnsw", "hnswlib", "routed")


class Playground:
    def __init__(self, dim: int = 0, metric: Metric = Metric.L2) -> None:
        self.metric = metric
        # Every index is kept over the same vectors, not one or the other. The
        # exact scan is the oracle the approximate ones are graded against;
        # hnswlib is the external reference that says whether ours is
        # competitive; `routed` is the same graph as `hnsw` with the neural
        # router supplying entry points, so comparing those two isolates
        # routing and nothing else.
        self.flat: FlatIndex | None = None
        self.hnsw: HnswIndex | None = None
        self.hnswlib = None
        self.kind = "flat"
        self.ef = 0                              # 0 = the index default
        self.base: np.ndarray | None = None      # mirror, for the oracle
        self.queries: np.ndarray | None = None
        self.published_gt: np.ndarray | None = None
        self.source = "(empty)"
        self.hnsw_build_seconds = 0.0
        self.hnswlib_build_seconds = 0.0
        self.router_note = "none"
        if dim:
            self._reset(dim)

    @property
    def index(self):
        """Whichever index `kind` currently selects.

        `hnsw` and `routed` are the *same object* — they differ only in whether
        search is told to use the router, which is the whole point of the
        controlled comparison.
        """
        if self.kind == "hnswlib":
            return self.hnswlib
        if self.kind in ("hnsw", "routed"):
            return self.hnsw
        return self.flat

    @property
    def uses_router(self) -> bool:
        return self.kind == "routed"

    # -- loading ----------------------------------------------------------

    def _reset(self, dim: int) -> None:
        self.flat = FlatIndex(dim, self.metric)
        self.hnsw = None
        self.hnswlib = None
        self.hnsw_build_seconds = 0.0
        self.hnswlib_build_seconds = 0.0
        self.router_note = "none"
        self.base = np.empty((0, dim), dtype=np.float32)

    def _install(self, vectors: np.ndarray, source: str,
                 queries: np.ndarray | None = None,
                 published_gt: np.ndarray | None = None) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            print(f"  expected a 2-D (n, dim) array, got shape {vectors.shape}")
            return

        self._reset(vectors.shape[1])
        start = time.perf_counter()
        self.flat.reserve(len(vectors))
        self.flat.add_batch(vectors)
        elapsed = time.perf_counter() - start

        self.base = vectors
        self.queries = queries
        self.published_gt = published_gt
        self.source = source
        print(f"  loaded {source}: {len(vectors):,} x {vectors.shape[1]}-d "
              f"in {elapsed*1000:.0f} ms")
        if self.kind in ("hnsw", "routed"):
            self.build_hnsw()
        self.stats()

    def build_hnsw(self, m: int = 16, ef_construction: int = 200) -> None:
        """Build the graph over whatever is currently loaded.

        Kept lazy because it costs seconds where the flat index costs
        milliseconds — that asymmetry is itself part of the tradeoff.
        """
        if self.base is None or not len(self.base):
            print("  nothing loaded; try `sift` or `random 5000`")
            return
        start = time.perf_counter()
        graph = HnswIndex(self.base.shape[1], self.metric, M=m,
                          ef_construction=ef_construction)
        graph.reserve(len(self.base))
        graph.add_batch(self.base)
        self.hnsw_build_seconds = time.perf_counter() - start
        self.hnsw = graph
        s = graph.stats()
        print(f"  built HNSW (M={m}, efC={ef_construction}) in "
              f"{self.hnsw_build_seconds*1000:.0f} ms")
        print(f"    {s.levels} levels, population {s.layer_population}")
        print(f"    mean degree at layer 0 = {s.mean_degree_l0:.1f}, "
              f"{s.reachable:,}/{s.nodes:,} reachable")
        print(f"    graph adds {s.graph_bytes/1e6:.1f} MB on top of the vectors")

    def build_hnswlib(self, m: int = 16, ef_construction: int = 200) -> None:
        """Build the reference implementation over the same vectors."""
        if not hnswlib_module.available:
            print("  hnswlib was not available when hylis was built.")
            print("  Reconfigure with -DHYLIS_WITH_HNSWLIB=ON to use it.")
            return
        if self.base is None or not len(self.base):
            print("  nothing loaded; try `sift` or `random 5000`")
            return
        start = time.perf_counter()
        lib = hnswlib_module.HnswlibIndex(self.base.shape[1], self.metric,
                                          capacity=len(self.base), M=m,
                                          ef_construction=ef_construction)
        lib.add_batch(self.base)
        self.hnswlib_build_seconds = time.perf_counter() - start
        self.hnswlib = lib
        print(f"  built hnswlib (M={m}, efC={ef_construction}) in "
              f"{self.hnswlib_build_seconds*1000:.0f} ms")

    def _corpus_slug(self) -> str:
        """Cache key for a trained router.

        Includes the shape, not just the source name: two different `random`
        corpora would otherwise share a filename, and a router trained for one
        dimensionality would be offered to the other.
        """
        name = "".join(c if c.isalnum() else "-" for c in self.source.split()[0])
        return f"{name}-{len(self.base)}x{self.base.shape[1]}"

    def load_router(self, path: str | None = None) -> None:
        """Attach a trained router, training one now if none exists.

        Training is the expensive part — clustering plus a few dozen epochs —
        so a saved router for this corpus is reused when there is one.
        """
        if self.hnsw is None:
            self.build_hnsw()
            if self.hnsw is None:
                return

        candidate = Path(path) if path else (
            ds.default_data_dir() / f"router-{self._corpus_slug()}.json")
        if candidate.exists():
            try:
                self.hnsw.set_router(NeuralRouter.load(str(candidate)))
            except (RuntimeError, ValueError) as exc:
                # A router trained on a different corpus is useless but not
                # fatal: say so and train a fresh one rather than leaving the
                # routed index silently unavailable.
                print(f"  {candidate.name} does not fit this corpus ({exc})")
                print("  training a replacement")
            else:
                self.router_note = f"loaded from {candidate.name}"
                print(f"  router loaded from {candidate}")
                return
        else:
            print(f"  no saved router at {candidate}; training one now")
        print("  (k-means + MLP - this takes a minute on a large corpus)")
        km = kmeans(self.base, n_clusters=min(256, max(2, len(self.base) // 20)),
                    seed=0)
        exact = None
        if 40_000 * len(self.base) <= 2e9:
            exact = self.flat
        ts = build_training_set(self.base, km.assignment, n_samples=40_000,
                                seed=0, exact_index=exact)
        model = RouterMLP(dim=self.base.shape[1], clusters=km.centroids.shape[0],
                          hidden=64, seed=0)
        start = time.perf_counter()
        model.fit(ts.x_train, ts.y_train, epochs=25)
        accuracy = float((model.predict(ts.x_val) == ts.y_val).mean())

        blob = router_json(model, km.medoids, vectors=self.base)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(blob, encoding="utf-8")
        self.hnsw.set_router(NeuralRouter.from_json(blob))
        self.router_note = f"trained here, val acc {accuracy:.3f}"
        print(f"  trained {km.centroids.shape[0]} clusters in "
              f"{time.perf_counter()-start:.0f}s, validation accuracy "
              f"{accuracy:.3f}")
        print(f"  saved to {candidate}")

    def set_kind(self, kind: str) -> None:
        if kind not in KINDS:
            print(f"  unknown index {kind!r}; one of {', '.join(KINDS)}")
            return
        if kind in ("hnsw", "routed") and self.hnsw is None:
            self.build_hnsw()
            if self.hnsw is None:
                return
        if kind == "routed" and not self.hnsw.has_router:
            self.load_router()
            if not self.hnsw.has_router:
                return
        if kind == "hnswlib" and self.hnswlib is None:
            self.build_hnswlib()
            if self.hnswlib is None:
                return
        self.kind = kind
        print(f"  queries now use the {kind} index")

    def load_random(self, n: int, dim: int = 64) -> None:
        v = ds.random_vectors(n=n, dim=dim, n_queries=min(50, n), seed=0,
                              n_clusters=max(2, n // 100))
        self._install(v.base, f"random {n}x{dim} (clustered)", v.queries)

    def load_sift(self, limit: int | None = None) -> None:
        try:
            s = ds.load_sift("siftsmall", limit=limit)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            return
        self._install(s.base, f"SIFT10K{'' if limit is None else f' (first {limit})'}",
                      s.queries, s.ground_truth)

    def load_npy(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            print(f"  no such file: {p}")
            return
        try:
            arr = np.load(p)
        except (ValueError, OSError) as exc:
            print(f"  could not read {p}: {exc}")
            return
        self._install(arr, p.name)

    def load_fvecs(self, path: str) -> None:
        try:
            self._install(ds.read_fvecs(path), Path(path).name)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {exc}")

    def load_csv(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            print(f"  no such file: {p}")
            return
        try:
            arr = np.loadtxt(p, delimiter=",", dtype=np.float32, ndmin=2)
        except ValueError as exc:
            print(f"  could not parse {p} as numeric rows: {exc}")
            print("  (a header line will do this -- delete it, or use .npy)")
            return
        self._install(arr, p.name)

    def add_vector(self, values: list[float]) -> None:
        vec = np.array(values, dtype=np.float32)
        if self.index is None:
            self._reset(len(vec))
            self.source = "(typed in)"
        if len(vec) != self.index.dim:
            print(f"  index holds {self.index.dim}-d vectors, got {len(vec)}")
            return
        vid = self.index.add(vec)
        self.base = np.vstack([self.base, vec])
        print(f"  added id {vid}   (size {len(self.index)})")

    # -- queries ----------------------------------------------------------

    def _require(self) -> bool:
        if self.index is None or len(self.index) == 0:
            print("  index is empty; try `random 2000` or `sift`")
            return False
        return True

    def _search(self, index, query: np.ndarray, k: int,
                allowed: list[int] | None = None, routed: bool | None = None):
        """Run one search on a specific index, honouring ef where it applies.

        Only our HnswIndex takes a router flag; FlatIndex has no entry point to
        choose and hnswlib has its own hierarchy we do not reach into.
        """
        if routed is None:
            routed = self.uses_router
        if index is self.hnsw:
            if allowed is None:
                return index.search(query, k, self.ef, routed)
            return index.search_filtered(query, k, allowed, self.ef, routed)
        if index is self.hnswlib:
            if allowed is None:
                return index.search(query, k, self.ef)
            return index.search_filtered(query, k, allowed, self.ef)
        if allowed is None:
            return index.search(query, k)
        return index.search_filtered(query, k, allowed)

    def _report(self, label: str, query: np.ndarray, k: int,
                allowed: list[int] | None = None) -> None:
        start = time.perf_counter()
        found = self._search(self.index, query, k, allowed)
        elapsed = time.perf_counter() - start

        if self.kind in ("hnsw", "routed"):
            scanned = f"{self.hnsw.last_visited:,} visited"
        elif self.kind == "hnswlib":
            scanned = "graph traversal"
        else:
            scanned = f"{(len(self.index) if allowed is None else len(allowed)):,} scanned"
        print(f"  [{self.kind}] {label} -> {len(found)} result(s) in "
              f"{elapsed*1e6:.0f} us ({scanned})")
        unit = "distance" if self.metric == Metric.L2 else "similarity"
        for rank, n in enumerate(found):
            print(f"    {rank:>3}. id {n.id:<8} {unit} {n.score:.6f}")

    def knn(self, vid: int, k: int = 5) -> None:
        if not self._require():
            return
        if not 0 <= vid < len(self.index):
            print(f"  id {vid} out of range [0, {len(self.index)})")
            return
        self._report(f"knn(id={vid}, k={k})", self.base[vid], k)

    def search_typed(self, k: int, values: list[float]) -> None:
        if not self._require():
            return
        if len(values) != self.index.dim:
            print(f"  index holds {self.index.dim}-d vectors, got {len(values)}")
            return
        self._report(f"search(k={k})", np.array(values, dtype=np.float32), k)

    def dataset_query(self, i: int, k: int = 5) -> None:
        if not self._require():
            return
        if self.queries is None:
            print("  this corpus has no query set; use `knn` or `search`")
            return
        if not 0 <= i < len(self.queries):
            print(f"  query {i} out of range [0, {len(self.queries)})")
            return
        self._report(f"query[{i}] (k={k})", self.queries[i], k)

    def filtered(self, selectivity: float, k: int = 5, qi: int = 0) -> None:
        """Filtered search, and the two plans that can answer it.

        Shows the number that decides the plan: how many vectors each route
        has to touch. Pre-filtering scans only the survivors; post-filtering
        scans everything and throws most of it away.
        """
        if not self._require():
            return
        if not 0.0 < selectivity <= 1.0:
            print("  selectivity must be in (0, 1]")
            return

        n = len(self.index)
        rng = np.random.default_rng(0)
        price = rng.uniform(0, 1000, size=n)
        threshold = ds.threshold_for_selectivity(price, selectivity)
        allowed = np.flatnonzero(price < threshold)
        if len(allowed) == 0:
            print("  predicate matched nothing; try a larger selectivity")
            return

        query = self.queries[qi] if self.queries is not None else self.index.vector_at(0)
        print(f"  predicate `price < {threshold:.2f}` keeps {len(allowed):,} "
              f"of {n:,} ({len(allowed)/n*100:.2f}%)")

        start = time.perf_counter()
        pre = self.index.search_filtered(query, k, allowed.tolist())
        pre_t = time.perf_counter() - start

        start = time.perf_counter()
        full_ids, _ = self.index.search_batch(query[None, :], k=n)
        allowed_set = set(allowed.tolist())
        post = [i for i in full_ids[0].tolist() if i in allowed_set][:k]
        post_t = time.perf_counter() - start

        agree = [x.id for x in pre] == post
        print(f"    pre-filter  (scan {len(allowed):,})  {pre_t*1e6:8.0f} us")
        print(f"    post-filter (scan {n:,})  {post_t*1e6:8.0f} us")
        print(f"    same answer: {agree}"
              f"{'' if agree else '   <-- THIS IS A BUG'}")
        print(f"    ids: {[x.id for x in pre]}")

    # -- verification -----------------------------------------------------

    def check(self, k: int = 10) -> None:
        """Diff the engine against the numpy oracle over the query set."""
        if not self._require():
            return
        queries = self.queries
        if queries is None:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(self.base), size=min(20, len(self.base)),
                              replace=False)
            queries = self.base[pick]
            print(f"  no query set; using {len(queries)} random stored vectors")

        metric_name = "angular" if self.metric == Metric.Cosine else "euclidean"
        if self.metric == Metric.InnerProduct:
            print("  the oracle covers L2 and cosine only; skipping")
            return

        start = time.perf_counter()
        ids, _ = self.index.search_batch(queries, k=k)
        engine_t = time.perf_counter() - start

        start = time.perf_counter()
        expected = ds.compute_ground_truth(self.base, queries, k=k, metric=metric_name)
        oracle_t = time.perf_counter() - start

        recall = ds.recall_at_k(ids, expected, k=k)
        exact = np.array_equal(ids, expected)
        print(f"  {len(queries)} queries, k={k}")
        print(f"    engine  {engine_t*1000:7.1f} ms")
        print(f"    oracle  {oracle_t*1000:7.1f} ms  (numpy, matrix form)")
        print(f"    same neighbours: {recall == 1.0}   (recall {recall:.4f})")
        print(f"    identical order: {exact}")
        if recall != 1.0:
            print("    MISMATCH -- exact search has one right answer, so this "
                  "is a bug")
        elif not exact:
            print("    (ordering differs only where distances tie within "
                  "float32 -- expected)")

    def truth(self, k: int = 10) -> None:
        """Compare against the corpus's own published ground truth."""
        if not self._require():
            return
        if self.published_gt is None or self.queries is None:
            print("  this corpus ships no published ground truth (SIFT does)")
            return
        k = min(k, self.published_gt.shape[1])
        ids, _ = self.index.search_batch(self.queries, k=k)
        recall = ds.recall_at_k(ids, self.published_gt[:, :k], k=k)
        print(f"  vs published ground truth, {len(self.queries)} queries, k={k}")
        print(f"    recall {recall:.4f}"
              f"{'   (exact match)' if recall == 1.0 else '   MISMATCH'}")

    def bench(self, n_queries: int = 100, k: int = 10) -> None:
        if not self._require():
            return
        warn_if_unoptimised()
        n = len(self.index)
        rng = np.random.default_rng(0)
        probes = self.base[rng.choice(n, size=min(n_queries, n), replace=False)]

        start = time.perf_counter()
        self.index.search_batch(probes, k=k)
        elapsed = time.perf_counter() - start
        per = elapsed / len(probes)

        print(f"  {len(probes)} queries over {n:,} x {self.index.dim}-d, k={k}")
        print(f"    {per*1000:.3f} ms/query   ({1/per:,.0f} queries/sec)")
        print(f"    {n*self.index.dim/per/1e9:.2f} G distance-ops/sec")
        print(f"  every query touches all {n:,} vectors -- this is the number")
        print(f"  HNSW has to beat, and the cost the planner avoids by filtering")

    # -- misc -------------------------------------------------------------

    def stats(self) -> None:
        if self.flat is None:
            print("  no index yet")
            return
        n = len(self.flat)
        print(f"  source  {self.source}")
        print(f"  size    {n:,} vectors x {self.flat.dim} dims")
        print(f"  metric  {METRIC_NAMES[self.metric]}")
        print(f"  active  {self.kind}" + (f"  (ef={self.ef or 'default'})"
                                          if self.kind != "flat" else ""))
        print(f"  memory  {n * self.flat.dim * 4 / 1e6:.1f} MB "
              f"(float32, one contiguous block)")
        if self.hnsw is not None:
            s = self.hnsw.stats()
            print(f"  hnsw    {s.levels} levels, +{s.graph_bytes/1e6:.1f} MB graph, "
                  f"{s.reachable:,}/{s.nodes:,} reachable, "
                  f"built in {self.hnsw_build_seconds*1000:.0f} ms")
            print(f"  router  {self.router_note}")
        else:
            print("  hnsw    not built (`hnsw` or `index hnsw` to build it)")
        if self.hnswlib is not None:
            print(f"  hnswlib built in {self.hnswlib_build_seconds*1000:.0f} ms "
                  f"(external baseline)")
        if self.queries is not None:
            print(f"  queries {len(self.queries)}"
                  f"{'  (+ published ground truth)' if self.published_gt is not None else ''}")

    def compare(self, k: int = 10, n_queries: int = 50) -> None:
        """Run the same queries through every available index.

        The whole point of keeping all of them: the exact scan gives the
        answer, each approximate index gives an answer, and the interesting
        question is what the speed cost in accuracy. `hnsw` and `routed` share
        one graph, so the gap between those two rows is routing and nothing
        else.
        """
        if not self._require():
            return

        queries = self.queries
        if queries is None:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(self.base), size=min(n_queries, len(self.base)),
                              replace=False)
            queries = self.base[pick]
        queries = queries[:n_queries]

        start = time.perf_counter()
        exact = [self.flat.search(q, k) for q in queries]
        flat_us = (time.perf_counter() - start) / len(queries) * 1e6
        total_truth = max(sum(len(r) for r in exact), 1)

        print(f"  {len(queries)} queries, k={k}, over {len(self.flat):,} vectors")
        print(f"    {'index':<10}{'ef':>6}{'recall':>10}{'us/query':>11}"
              f"{'visited':>10}{'vs flat':>9}")

        def row(name, run, visited_of=None, efs=None):
            for ef in (efs or ([self.ef] if self.ef else [10, 20, 50, 100, 200])):
                start = time.perf_counter()
                got = [run(q, ef) for q in queries]
                per_query = (time.perf_counter() - start) / len(queries) * 1e6
                hits = sum(len({n.id for n in g} & {n.id for n in w})
                           for g, w in zip(got, exact))
                visited = "-"
                if visited_of is not None:
                    samples = []
                    for q in queries[:10]:
                        run(q, ef)
                        samples.append(visited_of())
                    visited = f"{np.mean(samples):,.0f}"
                print(f"    {name:<10}{ef:>6}{hits/total_truth:>10.4f}"
                      f"{per_query:>11.0f}{visited:>10}{flat_us/per_query:>8.1f}x")

        if self.hnswlib is not None:
            row("hnswlib", lambda q, ef: self.hnswlib.search(q, k, ef))
        if self.hnsw is not None:
            row("hnsw", lambda q, ef: self.hnsw.search(q, k, ef, False),
                lambda: self.hnsw.last_visited)
            if self.hnsw.has_router:
                row("routed", lambda q, ef: self.hnsw.search(q, k, ef, True),
                    lambda: self.hnsw.last_visited)
            else:
                print("    routed     -- no router loaded (`router` to train one)")

        print(f"    {'flat':<10}{'-':>6}{1.0:>10.4f}{flat_us:>11.0f}"
              f"{len(self.flat):>10,}{'1.0x':>8}   <- exact, by definition")

    def set_metric(self, name: str) -> None:
        if name not in METRICS:
            print(f"  unknown metric {name!r}; one of {', '.join(METRICS)}")
            return
        self.metric = METRICS[name]
        if self.base is not None and len(self.base):
            print(f"  rebuilding index with metric {name}")
            self._install(self.base, self.source, self.queries, self.published_gt)
        else:
            print(f"  metric set to {name}")

    def clear(self) -> None:
        self.flat = None
        self.hnsw = None
        self.base = None
        self.queries = None
        self.published_gt = None
        self.source = "(empty)"
        self.hnsw_build_seconds = 0.0
        print("  cleared")


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------


def split_command(line: str) -> list[str]:
    """Tokenise, keeping Windows paths intact (see scripts/try_btree.py)."""
    if os.name != "nt":
        return shlex.split(line)
    parts = shlex.split(line, posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p
            for p in parts]


def dispatch(pg: Playground, line: str) -> bool:
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
        elif cmd == "random":
            pg.load_random(int(args[0]), int(args[1]) if len(args) > 1 else 64)
        elif cmd == "sift":
            pg.load_sift(int(args[0]) if args else None)
        elif cmd == "npy":
            pg.load_npy(args[0])
        elif cmd == "fvecs":
            pg.load_fvecs(args[0])
        elif cmd == "csv":
            pg.load_csv(args[0])
        elif cmd == "add":
            pg.add_vector([float(a) for a in args])
        elif cmd == "knn":
            pg.knn(int(args[0]), int(args[1]) if len(args) > 1 else 5)
        elif cmd == "search":
            pg.search_typed(int(args[0]), [float(a) for a in args[1:]])
        elif cmd == "query":
            pg.dataset_query(int(args[0]), int(args[1]) if len(args) > 1 else 5)
        elif cmd == "filter":
            pg.filtered(float(args[0]),
                        int(args[1]) if len(args) > 1 else 5,
                        int(args[2]) if len(args) > 2 else 0)
        elif cmd in ("check", "c"):
            pg.check(int(args[0]) if args else 10)
        elif cmd == "truth":
            pg.truth(int(args[0]) if args else 10)
        elif cmd in ("bench", "b"):
            pg.bench(int(args[0]) if args else 100,
                     int(args[1]) if len(args) > 1 else 10)
        elif cmd in ("stats", "s"):
            pg.stats()
        elif cmd == "index":
            pg.set_kind(args[0].lower())
        elif cmd == "hnsw":
            pg.build_hnsw(int(args[0]) if args else 16,
                          int(args[1]) if len(args) > 1 else 200)
            pg.kind = "hnsw"
        elif cmd == "flat":
            pg.set_kind("flat")
        elif cmd == "hnswlib":
            pg.build_hnswlib(int(args[0]) if args else 16,
                             int(args[1]) if len(args) > 1 else 200)
            if pg.hnswlib is not None:
                pg.kind = "hnswlib"
        elif cmd == "routed":
            pg.set_kind("routed")
        elif cmd == "router":
            pg.load_router(args[0] if args else None)
        elif cmd == "ef":
            pg.ef = max(0, int(args[0]))
            print(f"  ef = {pg.ef or 'index default'}")
        elif cmd == "compare":
            pg.compare(int(args[0]) if args else 10,
                       int(args[1]) if len(args) > 1 else 50)
        elif cmd == "metric":
            pg.set_metric(args[0].lower())
        elif cmd == "clear":
            pg.clear()
        else:
            print(f"  unknown command {cmd!r}; type `help`")
    except (IndexError, ValueError) as exc:
        print(f"  bad arguments for {cmd!r}: {exc}")
        print("  type `help` for usage")
    return True


DEMO = [
    "random 2000 32",
    "hnswlib",
    "router",
    "compare 10 50",
    "stats",
    "knn 0 5",
    "search 3 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
    "query 0 5",
    "check 10",
    "filter 0.01 5",
    "filter 0.5 5",
    "bench 100 10",
    "metric cosine",
    "check 10",
]


def run_demo() -> None:
    pg = Playground()
    for line in DEMO:
        print(f"> {line}")
        dispatch(pg, line)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metric", choices=sorted(METRICS), default="l2")
    parser.add_argument("--sift", action="store_true", help="preload SIFT10K")
    parser.add_argument("--random", metavar="NxDIM", help="preload random vectors")
    parser.add_argument("--npy", metavar="PATH", help="preload a .npy array")
    parser.add_argument("--fvecs", metavar="PATH", help="preload an .fvecs file")
    parser.add_argument("--demo", action="store_true", help="scripted walkthrough")
    args = parser.parse_args(argv)

    if args.demo:
        run_demo()
        return 0

    pg = Playground(metric=METRICS[args.metric])
    if args.sift:
        pg.load_sift()
    if args.random:
        n, _, dim = args.random.partition("x")
        pg.load_random(int(n), int(dim or 64))
    if args.npy:
        pg.load_npy(args.npy)
    if args.fvecs:
        pg.load_fvecs(args.fvecs)

    print("flat vector index playground. `help` for commands, `quit` to exit.")
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
