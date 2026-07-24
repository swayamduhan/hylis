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
hylis/
  storage/      # record store + write-ahead log
  index/        # B+ tree, learned index, HNSW, neural router
  planner/      # hybrid query planner
  bench/        # benchmark suite
  cli/          # demo UI
tests/          # pytest, one file per module
```

Python 3.11+. Core: NumPy only. Router uses PyTorch.
