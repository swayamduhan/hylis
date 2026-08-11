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
| 6 | Neural Router       | ✅ |
| 7 | Incremental retrain | ✅ |
| 8 | Query planner       | ✅ |
| 9 | Benchmarks          | ☐ |
| 10| Demo CLI            | ☐ |

## What is left, and why in this order

A review of the finished modules found one **structural** gap rather than a
list of small ones: the three index families were each validated, and nothing
composed them. `ColumnIndex.range_query` returns row ids; `search_filtered`
takes row ids; no code path connected the two. Every caller of
`search_filtered` built its `allowed` set by hand, and `make_hybrid` — written
for exactly this path — was used by nothing but its own tests.

That was module 8, and it is now built. The rest, ranked by value:

**1. ~~The hybrid query planner (module 8).~~** ✅ Done — see below.

**2. Wire `RecordStore` to `ColumnIndex`.** Module 1 is complete, tested, and
connected to nothing — no index reads from the store it built.

**3. Vector deletion.** Module 7 made the structured side fully mutable, but
`HnswIndex` and `FlatIndex` stay append-only, so `DynamicRMIndex::erase` has no
vector counterpart. The design is already owned: tombstone, filter at search,
compact at rebuild — identical to `dynamic_rmi.hpp`.

**4. The router during insertion.** The real research item, and the known
limitation this README already states below: insertion still descends the
hierarchy, which is why the layer-0-only build does not scale. Highest research
value, highest risk.

**5. Modules 9 and 10 are largely done** — nine benchmark scripts and two
interactive REPLs exist. What remains is consolidation, not new work.

### The framing these add up to

Three levels of a database, each with a learned component measured against the
non-learned baseline it replaces:

| level | learned thing | baseline | measured result |
|---|---|---|---|
| data | key → position (RMI) | B+ tree | 3–8× faster, wins at every cliff density |
| index | query → graph entry (neural router) | hierarchical descent | +31% at low ef, −6% at high |
| query | predicate → plan (planner) | fixed-threshold rule | 100% plan agreement once calibrated |

A *learned* cost model at the third level is the natural next step, and is
deliberately not built yet: it needs the fixed-threshold planner as the
baseline it has to beat, or there would be no way to show it was worth having.

**Out of scope, deliberately:** concurrency. The HNSW visited set is
epoch-stamped rather than cleared per query — clearing would make every search
O(n) however few nodes it touched — which makes search non-thread-safe. Stated
rather than papered over.

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

## The neural router

HNSW's upper layers do exactly one job: hand the layer-0 beam search a
starting node near the query. The router replaces that walk with a forward
pass — k-means partitions the corpus, each cluster's *medoid* becomes a
candidate entry point, and a small MLP classifies query → cluster.

Training needs **nothing collected and nothing labelled**. Inputs are base
vectors with jitter; labels are the cluster of each sample's true nearest
neighbour, computed by `FlatIndex`. The exact index built in module 3 to
*grade* approximate search turns out to also be what *teaches* the router.

```bash
python scripts/train_router.py --sift     # cluster, label, train, evaluate
python scripts/bench_router.py            # all four indexes, same corpus
```

Training is Python; inference is C++. A query costs tens of microseconds, so a
Python callback per query would cost more than the query does. Weights cross
as JSON, and a cross-language test asserts both forward passes predict the
same cluster on every query — a transposed matrix would otherwise give a
router that loads cleanly, runs fast, routes badly, and corrupts every
downstream number silently.

### Measured on SIFT10K

```
index        ef   recall@10        QPS   visited  routing
flat          -      1.0000        951    10,000        -
hnswlib      20      0.9820     43,258         -        -
hnsw         20      0.9630     17,052       363        4
routed       20      0.9770     16,355       273        0
```

**The router improves recall and cuts nodes visited by ~25%, but barely moves
throughput** — MLP inference costs about what the saved traversal did.

Three honest caveats, all of which correct assumptions made while planning
this module:

**The descent was never expensive.** It costs *4 nodes*, not the ~64 estimated
— the upper layers hold only 598, 38, 5, 1 nodes, so walking them is nearly
free. The router's value is therefore not eliminating that cost; it is
supplying *better* entry points, so the layer-0 beam converges in fewer steps.

**Half the gain is not the network.** Seeding the beam from two entry points
instead of one is worth +0.007 recall on its own, with random weights. The
trained network adds a further +0.007. Quoting the +0.014 total as a result
for "the neural router" would credit it with both.

**Our HNSW is competitive, not a strawman** — hnswlib reaches 0.9820 at ef=20
where ours reaches 0.9630, and ours is ~2× slower in raw throughput, which is
the expected gap between a hand-written implementation and a heavily optimised
one. That comparison is why hnswlib is vendored as a benchmark baseline
(`-DHYLIS_WITH_HNSWLIB=OFF` to build without it); a router that beat a slow
baseline would have proved nothing.

### At 1M vectors, the router earns its place

The 10K result was near-parity on throughput. Scaling to SIFT1M — where the
graph is 100× larger — is where the idea is actually testable:

```
index        ef   recall@10        QPS   visited  routing
flat          -      1.0000         20 1,000,000        -
hnswlib      20      0.8012     19,410         -        -
hnsw         20      0.7846      7,567       650        4
routed       20      0.7934      9,350       493        0
hnsw        160      0.9914      1,766     2,787        4
routed      160      0.9914      1,667     2,625        0
```

**+24% throughput at ef=20 and +31% at ef=10**, with recall unchanged — up
from parity at 10K. The mechanism is visible in `visited`: the router cuts
nodes scanned by 34% at ef=10, 24% at ef=20, and progressively less as the
beam widens. **The advantage shrinks as ef rises and reverses by ef=160**,
where a wide beam finds good neighbours regardless of where it started and
the router's inference is pure overhead.

So the honest claim is narrow and real: *a learned router pays off at low ef
and large n — exactly the regime where the entry point matters and there is
no beam width to recover from a bad one.*

Two costs to state alongside it:

**Our build is ~2.3× slower than hnswlib's** (12.1 min vs 5.3 min at 1M) and
superlinear in n — 4× the data at 80k costs 14× the time. At matched ef the
recall tracks hnswlib closely (0.9914 vs 0.9920 at ef=160), so the gap is
engineering, not algorithm, but it is real.

**The layer-0-only configuration does not scale.** The router replaces the
hierarchy at *query* time, but insertion still descends it — so with no upper
layers every insert walks layer 0 from a fixed entry point, and the build cost
grows worse than the normal one (1.5× at 80k, widening). `bench_router.py`
skips it above 200k. Making the router usable *during construction* is the
obvious next step and is not in this module.

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

### A prediction this project got wrong

`rmi.hpp` and `datasets.py` both used to claim that a **discontinuous** CDF is
where a distribution-free B+ tree beats the learned index. The reasoning was
that a model straddling a cliff eats the full error however large M gets —
which is true, and the conclusion still does not follow.

`scripts/experiment_discontinuity.py` holds n at 500,000 and sweeps the number
of cliffs from 64 to 250,000, where half of all keys start a new cluster:

```
   cliffs    btree  best RMI  speedup   max_err  probes  winner
       64    195.5      47.5    4.12x     3,299      64     RMI
    1,000    150.0      34.7    4.32x     2,073      53     RMI
   10,000    152.6      39.8    3.83x       229      52     RMI
  100,000    146.6      29.3    5.00x        91      64     RMI
  250,000    156.2      20.9    7.46x        67      64     RMI
```

**The B+ tree won 0 of 5 rows, and the RMI's margin grows with cliff density.**
Two things the reasoning missed: only cliff-many models are hurt, so cost
follows the typical model rather than the worst (`max_err` 3,299 beside 47
ns/lookup is that gap); and a hurt model falls back to *binary search*, capped
at O(log n), not to a scan.

One precision, which a test caught after a first draft of the correction
overstated it: a cliff genuinely does cost more **comparisons** — 64 against 4.
What it does not cost is more **time**, because those 64 are a linear pass over
one or two cache lines of a contiguous array while a B+ tree's 4 are
pointer-chased and miss cache every time. *Counting comparisons flatters the
tree.*

So no shape of data hands the tree a lookup win. Its claim is mutability —
which is what module 7 answers.

## Incremental retraining

`RMIndex` is build-only: every model and error bound derives from the whole key
set at once, so one insert invalidates all of them. `DynamicRMIndex` pays that
down without giving up exactness — an immutable RMI base, a **B+ tree delta
buffer** (module 2 reused), and a tombstone bitmap.

```bash
python scripts/experiment_merge_threshold.py   # rho and tau_e, oracle-checked
python scripts/bench_sosd.py --synthetic       # B+ tree vs RMI vs dynamic RMI
```

Following **DynaMind** (Cheng et al., *Knowledge-Based Systems* 348, 2026) for
the incremental-learning / machine-unlearning formulation, with two deviations,
both consequences of this index being *exact* where theirs need not be:

**Moments accumulate over the shifted coordinate** `x = key − origin`, not the
raw key. The paper's `S_kk = Σk²` reaches ~10³¹ at our key sizes, which a
double holds to a few figures before the subtraction that follows cancels the
rest.

**Unlearning updates the statistics only, never the installed model.** Moving
the model would invalidate the error bounds measured against it, and a lookup
could then miss a key that exists.

**Deletions are tombstoned, never compacted**, and that is load-bearing twice:
base positions never shift, so every surviving key's bound stays valid, and
withdrawing a key from its model's moments becomes exact and O(1).

Two findings worth stating:

**The paper's Cook's-distance trigger buys nothing here.** Every finite `τ_e`
from 0.1 to 100 gave identical merge counts and mean error, at ~20× the cost of
no trigger at all. The cause is architectural — inserts land in the delta
buffer, so base models stay exactly right until the merge and there is no drift
to catch. (Separately, the score is not scale-free in segment length: median
0.58 at 1562 keys/model against 6.5e6 at 6, so `τ_e = 1.0` could not have
transferred regardless.) The default is therefore the trigger **off**.

**The incremental merge's saving is real but conditional.** A model that saw no
write needs only its intercept shifted, which provably leaves its residuals —
and so its bounds — untouched. An append leaves 255 of 256 models needing only
that; a *scattered* batch disturbs 192 of them and rescans 75% of the keys.
Both halves are asserted in tests, so the saving is never quoted as
unconditional.

The router goes stale the same way and is repaired in two tiers — see
`router_health()` / `repair_router_medoids()`. Drift is measured against the
**medoids** rather than centroids, because the medoid is the node the beam
actually starts from, and because the router format deliberately does not
persist centroids.

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

## The hybrid query planner

The module the project is named for: one query, two indexes.

```sql
SELECT id FROM t
WHERE  price < $t                     -- structured predicate
ORDER BY distance(embedding, $q)      -- vector similarity
LIMIT  10
```

```bash
python scripts/bench_planner.py --calibrate
```

Everything needed already existed — `ColumnIndex.range_query` returns row ids,
and both vector indexes take row ids as a filter. `HybridPlanner` is that join
plus the decision of how to execute it, across three plans:

| plan | cost | wins when |
|---|---|---|
| `PreFilter` | `O(\|allowed\|)`, exact | tight predicates |
| `FilteredGraph` | grows as the predicate tightens | loose predicates |
| `PostFilter` | search unfiltered, then drop non-matches | **never — it is the trap** |

**`PostFilter` is implemented because it is what a system without a planner
does**, and measuring it is what makes choosing against it a result rather than
an assumption. At 20k×32-d, k=10, across 50 queries (500 rows wanted):

```
selectivity   0.1%   0.5%   1.0%   5.0%  10.0%  25%+
rows missing   498    484    463    262     75     0
```

At 0.1% selectivity it returns **2 rows out of 500**. Not slow — *wrong*, and
silently so.

**Selectivity is executed, not estimated.** The planner runs the predicate
through `ColumnIndex` first, so it knows `|allowed|` exactly. A real optimiser
estimates from histograms because it must plan before executing; here the
structured lookup is nanoseconds and the row ids are work the query needs
anyway, so executing the predicate *is* the estimate. The cost — a predicate
matching nearly everything is paid for in full before the planner can discover
it — is stated rather than hidden.

**The 50% crossover did not transfer, and calibration fixes it.** The threshold
was inherited from module 5's SIFT10K measurement; on 20,000 32-d vectors at
ef=64 it sits nearer 20%:

| | agreement with the measured winner | mean regret | worst regret |
|---|---|---|---|
| inherited 50% | 8/10 | 1.24× | **3.25×** |
| `calibrate()` → 20.8% | **10/10** | **1.00×** | **1.00×** |

The crossover is not a constant of the algorithm — it moves with n,
dimensionality, ef and cache speed. `calibrate()` times both plans on real
queries and adopts the measured midpoint, which is the same answer
`choose_index()` gives one level down.

*Regret* is reported next to agreement deliberately: a planner wrong 20% of the
time but only ever by 3% is better than one wrong 5% of the time and
catastrophically so.

### A bug every test missed and the benchmark caught

`matching_rows` returns row ids ordered by the column's **key** — the attribute
— not by row id. `PostFilter` tested membership with a binary search, which is
only valid on a list sorted by row id.

Every C++ fixture happened to use a column whose values were already ascending,
so the search always succeeded and the bug was invisible to the entire suite.
The first real attribute column exposed it immediately: ~2 of 500 rows returned
*at every selectivity, including 100%*. **Every test was correct; they all
shared one accidental property.** Now pinned by a test that first asserts its
own fixture lacks that property.

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
hardest shape for a piecewise-linear fit. `KeyDataset.position_error()` scores
that shape in records, giving an axis to plot learned-index error against.

> This paragraph used to end "…and is the adversarial case where a B+ tree
> should win." **That was measured and retracted** — see *A prediction this
> project got wrong* below.

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
