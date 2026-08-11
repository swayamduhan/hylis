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
| 3 | Brute-force vector  | ☐ |
| 4 | Learned Index (RMI) | ☐ |
| 5 | HNSW baseline       | ☐ |
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
    index/      # B+ tree, and later the learned index / HNSW
  bindings/     # pybind11 modules -> hylis._storage, hylis._btree
  tests/        # GoogleTest suites for the C++ core
python/hylis/   # user-facing Python API; built extensions land here
  datasets.py   # data generators + loaders + the brute-force oracle
scripts/        # fetch_data.py, benchmark drivers
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
cmake --preset default      # or: cmake -B build -G Ninja
cmake --build build

ctest --test-dir build      # C++ suites
pytest tests/               # Python bridge suites
```

The compiled extensions are written straight into `python/hylis/`, so
`pytest` works from a clean checkout with no install step.

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
