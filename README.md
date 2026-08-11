# hylis — Hybrid Learned Index System

A from-scratch hybrid database indexing engine combining:

1. **B+ Tree** — classical ordered index (point + range lookup)
2. **Learned Index** — two-stage Recursive Model Index (RMI) approximating the CDF
3. **Neural-Routed HNSW** — HNSW with the upper graph layers replaced by a small
   neural router (the novel contribution)
4. **Hybrid Query Planner** — routes each query to the cheapest applicable index

Built one module at a time, fully tested, with no core-logic dependencies on
FAISS / hnswlib / etc. (those are benchmark baselines only).

## Status

| # | Module              | Status |
|---|---------------------|--------|
| 1 | Storage + WAL       | ✅ |
| 2 | B+ Tree             | ✅ |
| 3 | Brute-force vector  | ✅ |
| 4 | Learned Index (RMI) | ✅ |
| 5 | HNSW baseline       | ✅ |
| 6 | Neural Router       | ☐ |
| 7 | Incremental retrain | ☐ |
| 8 | Query planner       | ☐ |
| 9 | Benchmarks          | ☐ |
| 10| Demo CLI            | ☐ |

## Layout

```
cpp/
  hylis/
    storage/    # record store + write-ahead log (header-only)
    index/      # B+ tree, RMI, per-column selection, flat + HNSW vector search
  bindings/     # pybind11 -> hylis._storage, _btree, _flat, _rmi, _hnsw
  tests/        # GoogleTest suites for the C++ core
python/hylis/   # user-facing Python API; built extensions land here
  datasets.py   # data generators + loaders + the brute-force oracle
  learned.py    # numpy stage-1 models, for the RMI ablation only
scripts/        # fetch_data.py, benchmarks, experiments, playgrounds
tests/          # pytest, exercising the C++ core through the bindings
data/           # downloaded corpora (git-ignored)
```

The C++ core is header-only and built with CMake; the ML layers (learned
index, neural router, planner) will be pure Python calling into that core
in-process via pybind11.

## Build and test

Requires a C++17 compiler, CMake 3.18+, Ninja and Python 3.8+ on `PATH`
(pybind11 and GoogleTest are fetched automatically).

```bash
cmake --preset release      # use this for anything you intend to time
cmake --build build

ctest --test-dir build      # C++ suites
pytest tests/               # Python bridge suites
```

`--preset default` is a Debug build; a bare `cmake -B build` now defaults to
Release, because an unoptimised scan measures ~5.7x slower than a real one
and would silently corrupt every benchmark drawn from it. The extensions
expose `hylis._flat.__optimized__` so the benchmark scripts can refuse to
report timings from a Debug build.

The compiled extensions are written straight into `python/hylis/`, so
`pytest` works from a clean checkout with no install step.

## The learned index

`RMIndex` is a two-stage Recursive Model Index: for sorted keys the CDF *is*
the map from key to position, so it is approximated by a model and corrected
with a short local search.

It is **exact, not approximate**. Each second-stage model records the worst
prediction error it made at build time, so a lookup searches only a provably
sufficient window. A bad model costs time; it can never cost correctness —
`validate()` proves that by replaying every key.

There is **no neural network and no training loop**. Both stages are
closed-form least-squares fits; a million-key index builds in ~20 ms.

```bash
python scripts/bench_index.py            # B+ tree vs RMI, to 10M keys
python scripts/experiment_curvature.py   # error vs model count, per distribution
python scripts/experiment_stage1.py      # linear vs a from-scratch MLP stage 1
```

Three findings the scripts produce, all reproducible:

**Curvature is cheap; discontinuity is not.** On `lognormal` (curved but
continuous) mean error falls 926× as models are added, and keeps falling. On
`clustered` (SOSD `fb`-shaped, cliffs rather than curves) it stops moving
*entirely* past 4096 models — 16× more models change nothing, because 99% of
them are routed no keys at all. That floor is set by gaps in the data, not by
the model budget.

**The RMI wins every read-only row.** 3–8× faster lookups than the B+ tree at
every size and distribution tested, in less memory. The tree's real claim is
mutability: the RMI has no `insert` at all and must be rebuilt wholesale.

**A neural stage 1 helps only where routing is the bottleneck.** It is
indistinguishable from a line on near-linear keys and hits the same floor on
stepped ones, but on a skewed CDF it genuinely routes better (mean error
2.8 → 1.2). It costs ~1000× the fit time to get there.

`ColumnIndex` picks per column by **building every candidate and timing real
lookups**, rather than guessing from an analytic cost model. `IndexCatalog`
persists those decisions — plans, not fitted models, since models rebuild in
milliseconds while producing a plan means re-measuring everything. Each plan
carries a fingerprint of the data it was chosen for, so a stale plan is
detected and re-tuned; it can only ever cost speed, never correctness.

## Vector search: exact and approximate

Two indexes over the same vectors, both kept in the final build.

`FlatIndex` scans everything, so it is exact by construction — the oracle
HNSW's recall is measured against, and the baseline its speedups are quoted
from. `HnswIndex` is a hierarchical proximity graph (Malkov & Yashunin, 2018):
a random layer assignment builds sparse upper layers that act as a highway
over the data, and the search descends through them before doing fine-grained
work at layer 0.

```bash
python scripts/bench_vector.py             # recall/QPS curve + filtered crossover
python scripts/try_vectors.py --sift       # then: hnsw, compare, index flat
```

Measured on SIFT10K (M=16, efConstruction=200):

```
index      ef   recall@10        QPS   visited   speedup
flat        -      1.0000      2,531    10,000     1.00x
hnsw       10      0.8840     47,128       252    18.6x
hnsw       20      0.9630     32,508       363    12.8x
hnsw       40      0.9980     19,214       560     7.6x
hnsw       80      1.0000     13,845       868     5.5x
```

**ef=20 clears 0.95 recall at 12.8× the throughput of an exhaustive scan**,
touching 363 of 10,000 vectors. Layer populations come out
`[10000, 598, 38, 5, 1]` — the ~1/M decay the algorithm depends on — for 27%
memory overhead on top of the raw vectors.

Three things worth knowing:

**The selection heuristic is the whole algorithm.** Keeping a node's M
*nearest* candidates fills its links with mutual near-duplicates and leaves
no way out of its own cluster. The paper's Algorithm 4 instead drops a
candidate that is closer to an already-chosen neighbour than to the node
itself, preserving long-range edges. `use_heuristic = False` switches it off
so the difference is measurable rather than asserted.

**Full reachability is not guaranteed.** Links are added bidirectionally, but
pruning a full neighbour list can drop the back-link — and a node whose every
in-edge is pruned can never be returned, whatever ef is. At M=16 reachability
is 100%; at M=2 it strands ~5% of nodes. So it is reported by `reachable()`
as a statistic, not asserted by `validate()`.

**Filtered search crosses over at ~50% selectivity.** Filtered brute force is
`O(|allowed|)` and exact, so it gets cheaper as a predicate tightens. The
graph must traverse non-matching nodes to stay connected, so it gets more
expensive — at 0.1% selectivity it visits all 10,004 nodes and takes 1502 µs
against brute force's 3 µs. Below ~50% selectivity the exhaustive scan wins
outright. That crossover is precisely what the query planner has to predict.

## Trying the B+ tree

```bash
python scripts/try_btree.py --demo            # scripted walkthrough
python scripts/try_btree.py --order 4         # interactive; small order splits fast
python scripts/try_btree.py --load clustered:20000
python scripts/try_btree.py --csv mydata.csv
```

The REPL keeps a shadow Python dict beside the tree, so `check` diffs every
key against it and re-runs `validate()` on demand — the same differential
idea as the C++ fuzz test, exposed so you can try to break the tree by hand.

## Data

`hylis.datasets` supplies three things, matching the three things the engine
has to index:

| Helper | For | Source |
|---|---|---|
| `synthetic_keys(dist, n, seed)` | B+ tree, learned index | generated offline |
| `load_sift(variant)` | vector search, HNSW, router | downloaded |
| `make_hybrid(vectors)` | query planner | vectors + generated predicates |

Key distributions are generated rather than downloaded because for a learned
index the *shape of the key CDF is the experiment*: `sequential_gaps` is
near-linear and easy, `clustered` emulates the SOSD `fb` dataset and is the
adversarial case where a B+ tree should win. `KeyDataset.linearity()` scores
that shape, giving an axis to plot learned-index error against.

Vectors come from SIFT because it ships **published ground-truth neighbours**,
so recall can be checked against an oracle this project had no hand in
producing. `compute_ground_truth` is verified against those published lists in
`tests/test_sift.py`.

```bash
pip install -r requirements.txt
python scripts/fetch_data.py --list
python scripts/fetch_data.py siftsmall     # ~5 MB
```

Nothing is blocked on the download — everything except `load_sift` works
offline, and the SIFT tests skip themselves when `data/` is empty. `data/` is
git-ignored; benchmark corpora do not belong in a repository.
