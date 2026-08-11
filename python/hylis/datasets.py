"""Datasets for building, testing and benchmarking the hylis engine.

Deliberately not imported by ``hylis/__init__.py``: this is the only part of
the package that needs numpy, and keeping it a submodule means ``import
hylis`` stays dependency-free for anyone who only wants the C++ core.

    from hylis import datasets

Three families, matching the three things the engine has to index:

``synthetic_keys``     sorted int64 keys with a controllable CDF shape, for the
                       B+ tree and the learned index. The shape *is* the
                       experiment for an RMI, so these are generated rather
                       than downloaded.
``load_sift``          real 128-dim vectors with published ground-truth
                       neighbours, for brute-force search / HNSW / the router.
``make_hybrid``        vectors plus scalar attributes at *exact* selectivity,
                       for the query planner. This is the only one that
                       exercises structured and vector search together.

Everything is seeded and reproducible, and everything except ``load_sift``
works offline, so no module is ever blocked on a download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

__all__ = [
    "KeyDataset",
    "VectorDataset",
    "HybridDataset",
    "KeyDistribution",
    "Metric",
    "synthetic_keys",
    "random_vectors",
    "read_fvecs",
    "read_ivecs",
    "load_sift",
    "compute_ground_truth",
    "recall_at_k",
    "make_hybrid",
    "threshold_for_selectivity",
    "default_data_dir",
]

KeyDistribution = Literal["uniform", "lognormal", "sequential_gaps", "clustered"]
Metric = Literal["euclidean", "angular"]


def default_data_dir() -> Path:
    """Repo-root ``data/``. Git-ignored; populated by scripts/fetch_data.py."""
    return Path(__file__).resolve().parents[2] / "data"


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyDataset:
    """Sorted, unique int64 keys plus the payload values they map to.

    Sorted and unique because that is what both indexes assume: the B+ tree
    enforces uniqueness anyway, and a learned index models the key CDF, which
    is only defined over a sorted domain.
    """

    keys: np.ndarray
    values: np.ndarray
    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.keys.ndim != 1:
            raise ValueError(f"keys must be 1-D, got shape {self.keys.shape}")
        if self.keys.shape != self.values.shape:
            raise ValueError(
                f"keys/values length mismatch: {self.keys.shape} vs {self.values.shape}"
            )

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def items(self) -> Iterator[tuple[int, int]]:
        """(key, value) pairs as Python ints, ready to feed BPlusTree.insert."""
        return zip(self.keys.tolist(), self.values.tolist())

    def _linear_cdf_fit(self) -> np.ndarray:
        """Absolute position error of a single linear fit to the key CDF."""
        n = len(self)
        if n < 2:
            return np.zeros(n, dtype=np.float64)
        x = self.keys.astype(np.float64)
        y = np.arange(n, dtype=np.float64)
        if np.ptp(x) == 0:
            return np.zeros(n, dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        return np.abs((slope * x + intercept) - y)

    def position_error(self) -> tuple[float, float]:
        """(mean, max) position error of one linear model over the whole CDF.

        This is the headline difficulty metric, because it is not a proxy for
        anything -- it *is* what a learned index costs. An RMI predicts a
        key's position and then searches locally to correct itself, so the
        max error is exactly the window it must search, and halving it halves
        the work. Reported in records, so it is directly comparable to the
        ~log2(n) comparisons a B+ tree would spend.
        """
        err = self._linear_cdf_fit()
        return float(err.mean()), float(err.max())

    def linearity(self) -> float:
        """R^2 of a straight line fit to the key CDF, in [0, 1].

        Kept because it is the conventional summary, but prefer
        ``position_error`` -- R^2 is dominated by the overall spread and is
        badly insensitive to exactly the structure that hurts a learned
        index. The ``clustered`` generator scores ~0.995 here, looking almost
        perfectly linear, while carrying ~45x the position error of
        ``uniform``: the damage is local, and a global R^2 cannot see it.
        """
        n = len(self)
        if n < 2:
            return 1.0
        x = self.keys.astype(np.float64)
        y = np.arange(n, dtype=np.float64)
        # Guard the degenerate all-identical-keys case, where var(x) == 0.
        if np.ptp(x) == 0:
            return 1.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr * corr)


@dataclass(frozen=True)
class VectorDataset:
    """Base vectors, query vectors, and (optionally) true nearest neighbours.

    ``ground_truth[i]`` holds the indices into ``base`` of the true nearest
    neighbours of ``queries[i]``, nearest first. When it comes from the
    dataset's own published file it is an independent oracle -- the engine's
    results can be checked against numbers hylis had no part in producing.
    """

    base: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray | None
    name: str
    metric: Metric = "euclidean"

    def __post_init__(self) -> None:
        if self.base.ndim != 2 or self.queries.ndim != 2:
            raise ValueError("base and queries must both be 2-D (n, dim)")
        if self.base.shape[1] != self.queries.shape[1]:
            raise ValueError(
                f"dimension mismatch: base is {self.base.shape[1]}-d, "
                f"queries are {self.queries.shape[1]}-d"
            )
        if self.ground_truth is not None:
            if self.ground_truth.shape[0] != self.queries.shape[0]:
                raise ValueError(
                    f"ground truth has {self.ground_truth.shape[0]} rows but "
                    f"there are {self.queries.shape[0]} queries"
                )

    @property
    def n(self) -> int:
        return int(self.base.shape[0])

    @property
    def dim(self) -> int:
        return int(self.base.shape[1])

    @property
    def n_queries(self) -> int:
        return int(self.queries.shape[0])

    def with_ground_truth(self, k: int = 100) -> "VectorDataset":
        """Return a copy with ground truth computed if it is missing."""
        if self.ground_truth is not None and self.ground_truth.shape[1] >= k:
            return self
        gt = compute_ground_truth(self.base, self.queries, k, self.metric)
        return VectorDataset(self.base, self.queries, gt, self.name, self.metric)


@dataclass(frozen=True)
class HybridDataset:
    """Vectors with scalar attributes attached -- the query planner's input.

    ``attributes`` maps column name to a length-n array aligned with
    ``vectors.base``, so row i of the vector set and ``attributes[c][i]``
    describe the same record.
    """

    vectors: VectorDataset
    keys: np.ndarray
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "hybrid"

    def __post_init__(self) -> None:
        n = self.vectors.n
        if self.keys.shape[0] != n:
            raise ValueError(f"expected {n} keys, got {self.keys.shape[0]}")
        for col, values in self.attributes.items():
            if values.shape[0] != n:
                raise ValueError(
                    f"attribute {col!r} has {values.shape[0]} rows, expected {n}"
                )

    def __len__(self) -> int:
        return self.vectors.n

    def selectivity(self, mask: np.ndarray) -> float:
        """Fraction of records a boolean mask keeps."""
        return float(np.count_nonzero(mask) / len(self))


# --------------------------------------------------------------------------
# Synthetic keys
# --------------------------------------------------------------------------


def synthetic_keys(
    distribution: KeyDistribution = "uniform",
    n: int = 100_000,
    seed: int = 0,
    n_clusters: int = 100,
) -> KeyDataset:
    """Generate ``n`` sorted, unique int64 keys with a given CDF shape.

    The four shapes span the range a learned index actually cares about, from
    the case where one linear model suffices to the case where it badly fails:

    ``sequential_gaps``  near-linear CDF (dense ids with small holes). The easy
                         case -- an RMI should approach O(1).
    ``uniform``          linear in expectation but locally noisy.
    ``lognormal``        heavy right tail; the distribution the original RMI
                         paper used.
    ``clustered``        dense clumps separated by wide empty gaps, emulating
                         the SOSD ``fb`` dataset. The adversarial case, where a
                         B+ tree is expected to *beat* the learned index.

    Values are the row ids 0..n-1, which is what the engine stores against a
    key anyway.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = np.random.default_rng(seed)

    # Over-generate, dedupe, and top up: uniqueness is required, and dedupe
    # shrinks the array by an amount that depends on the distribution.
    keys = np.empty(0, dtype=np.int64)
    attempts = 0
    while keys.shape[0] < n:
        attempts += 1
        if attempts > 20:
            raise RuntimeError(
                f"could not draw {n} unique keys from {distribution!r}; the "
                "distribution is too concentrated for this n"
            )
        draw = _draw_keys(distribution, int(n * 1.3) + 1024, rng, n_clusters)
        keys = np.unique(np.concatenate([keys, draw]))

    keys = keys[:n]
    values = np.arange(n, dtype=np.int64)
    return KeyDataset(
        keys=keys,
        values=values,
        name=f"{distribution}-{n}-s{seed}",
        description=_KEY_DESCRIPTIONS[distribution],
    )


_KEY_DESCRIPTIONS: dict[str, str] = {
    "uniform": "uniform over a wide int64 range; linear CDF with local noise",
    "lognormal": "lognormal, heavy right tail (as used in the original RMI paper)",
    "sequential_gaps": "dense ascending ids with small random gaps; near-linear CDF",
    "clustered": "dense clusters separated by wide gaps; emulates SOSD 'fb'",
}


def _draw_keys(
    distribution: str, size: int, rng: np.random.Generator, n_clusters: int
) -> np.ndarray:
    if distribution == "uniform":
        return rng.integers(0, 2**40, size=size, dtype=np.int64)

    if distribution == "lognormal":
        # sigma=2 gives a pronounced tail without overflowing int64 once
        # scaled; clip defensively since lognormal is unbounded above.
        raw = rng.lognormal(mean=0.0, sigma=2.0, size=size)
        scaled = np.clip(raw * 1e6, 0, float(2**52))
        return scaled.astype(np.int64)

    if distribution == "sequential_gaps":
        gaps = rng.integers(1, 8, size=size, dtype=np.int64)
        return np.cumsum(gaps)

    if distribution == "clustered":
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        centres = rng.integers(0, 2**40, size=n_clusters, dtype=np.int64)
        # Spread within a cluster is tiny relative to the gaps between them,
        # which is what makes a single linear model fit so poorly.
        pick = rng.integers(0, n_clusters, size=size)
        offset = rng.integers(0, 10_000, size=size, dtype=np.int64)
        return centres[pick] + offset

    raise ValueError(
        f"unknown distribution {distribution!r}; expected one of "
        f"{sorted(_KEY_DESCRIPTIONS)}"
    )


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------


def random_vectors(
    n: int = 10_000,
    dim: int = 128,
    n_queries: int = 100,
    seed: int = 0,
    n_clusters: int = 0,
    metric: Metric = "euclidean",
) -> VectorDataset:
    """Synthetic vectors, for development before SIFT is downloaded.

    With ``n_clusters > 0`` the points are drawn around that many centres,
    which is much more like real embedding data than isotropic noise: in high
    dimensions uniform random points are all roughly equidistant, so an ANN
    index has no structure to exploit and recall numbers measured on them are
    meaningless. Use clusters for anything you intend to quote.

    Ground truth is computed exactly, so recall is measurable immediately.
    """
    rng = np.random.default_rng(seed)
    if n_clusters > 0:
        centres = rng.normal(0.0, 1.0, size=(n_clusters, dim)).astype(np.float32)
        pick = rng.integers(0, n_clusters, size=n)
        base = centres[pick] + rng.normal(0.0, 0.15, size=(n, dim)).astype(np.float32)
        qpick = rng.integers(0, n_clusters, size=n_queries)
        queries = centres[qpick] + rng.normal(
            0.0, 0.15, size=(n_queries, dim)
        ).astype(np.float32)
    else:
        base = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
        queries = rng.normal(0.0, 1.0, size=(n_queries, dim)).astype(np.float32)

    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    gt = compute_ground_truth(base, queries, min(100, n), metric)
    return VectorDataset(base, queries, gt, f"random-{n}x{dim}-s{seed}", metric)


def read_fvecs(path: str | Path, limit: int | None = None) -> np.ndarray:
    """Read a ``.fvecs`` file into an (n, dim) float32 array.

    Format (TEXMEX/SIFT): each vector is a little-endian int32 dimension
    followed by that many float32s. The dimension is repeated on every record,
    so the file is read as int32, reshaped to (n, dim+1), and the count column
    dropped -- no per-vector Python loop.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"{path}: bad leading dimension {dim}; not an fvecs file")
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(
            f"{path}: size {raw.size} is not a multiple of {stride} "
            f"(dim {dim} + 1); file is truncated or not fvecs"
        )

    rows = raw.reshape(-1, stride)
    if limit is not None:
        rows = rows[:limit]
    # Same buffer reinterpreted as float32; copy so the result owns its memory.
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def read_ivecs(path: str | Path, limit: int | None = None) -> np.ndarray:
    """Read an ``.ivecs`` file into an (n, dim) int32 array.

    Identical layout to fvecs but with int32 payloads; this is how SIFT ships
    its ground-truth neighbour indices.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"{path}: bad leading dimension {dim}; not an ivecs file")
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(
            f"{path}: size {raw.size} is not a multiple of {stride} "
            f"(dim {dim} + 1); file is truncated or not ivecs"
        )

    rows = raw.reshape(-1, stride)
    if limit is not None:
        rows = rows[:limit]
    return np.ascontiguousarray(rows[:, 1:])


def write_fvecs(path: str | Path, vectors: np.ndarray) -> None:
    """Write an (n, dim) float32 array as ``.fvecs``. Used by the tests."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {vectors.shape}")
    n, dim = vectors.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = vectors.view(np.int32)
    with open(path, "wb") as fh:
        fh.write(out.tobytes())


def load_sift(
    variant: Literal["siftsmall", "sift"] = "siftsmall",
    data_dir: str | Path | None = None,
    limit: int | None = None,
) -> VectorDataset:
    """Load SIFT10K (``siftsmall``) or SIFT1M (``sift``) from ``data/``.

    Both ship published ground truth, which is the reason to prefer them over
    synthetic data: recall@k can be checked against neighbour lists computed
    by someone else, so a bug in hylis cannot quietly define its own oracle.

    Run ``python scripts/fetch_data.py siftsmall`` if this raises.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    base_dir = root / variant
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"{base_dir} not found. Download it with:\n"
            f"    python scripts/fetch_data.py {variant}"
        )

    base = read_fvecs(base_dir / f"{variant}_base.fvecs", limit=limit)
    queries = read_fvecs(base_dir / f"{variant}_query.fvecs")
    gt_path = base_dir / f"{variant}_groundtruth.ivecs"
    ground_truth = read_ivecs(gt_path) if gt_path.exists() else None

    # Published ground truth indexes the *full* base set, so truncating the
    # base invalidates it rather than merely shortening it.
    if limit is not None and ground_truth is not None:
        ground_truth = compute_ground_truth(base, queries, 100, "euclidean")

    return VectorDataset(base, queries, ground_truth, variant, "euclidean")


# --------------------------------------------------------------------------
# Ground truth and recall
# --------------------------------------------------------------------------


def compute_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    metric: Metric = "euclidean",
    chunk: int = 4096,
) -> np.ndarray:
    """Exact k nearest neighbours by brute force. The test oracle.

    This is the same role ``std::map`` plays for the B+ tree: an obviously
    correct reference the fast implementation is diffed against. It is
    deliberately numpy rather than the C++ core, so a bug in the engine cannot
    also corrupt the thing checking the engine.

    Distances are accumulated over chunks of ``base`` to bound peak memory at
    O(chunk * n_queries) rather than O(n * n_queries).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = base.shape[0]
    k = min(k, n)
    nq = queries.shape[0]

    if metric == "angular":
        base = _l2_normalise(base)
        queries = _l2_normalise(queries)

    best_d = np.full((nq, 0), np.inf, dtype=np.float32)
    best_i = np.zeros((nq, 0), dtype=np.int32)

    for start in range(0, n, chunk):
        block = base[start : start + chunk]
        if metric == "euclidean":
            # ||q - b||^2 = ||q||^2 - 2 q.b + ||b||^2; the ||q||^2 term is
            # constant per query so it cannot change the ranking and is
            # dropped. These are therefore not true distances -- only the
            # order and the returned indices are meaningful.
            d = (block * block).sum(axis=1)[None, :] - 2.0 * (queries @ block.T)
        else:
            d = -(queries @ block.T)

        idx = np.arange(start, start + block.shape[0], dtype=np.int32)
        idx = np.broadcast_to(idx, (nq, block.shape[0]))

        cand_d = np.concatenate([best_d, d.astype(np.float32)], axis=1)
        cand_i = np.concatenate([best_i, idx], axis=1)

        keep = min(k, cand_d.shape[1])
        part = np.argpartition(cand_d, keep - 1, axis=1)[:, :keep]
        rows = np.arange(nq)[:, None]
        best_d = cand_d[rows, part]
        best_i = cand_i[rows, part]

    order = np.argsort(best_d, axis=1, kind="stable")
    rows = np.arange(nq)[:, None]
    return np.ascontiguousarray(best_i[rows, order][:, :k])


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Zero vectors have no direction; leaving them at zero keeps every angular
    # distance to them equal rather than producing NaN.
    np.maximum(norms, 1e-12, out=norms)
    return (x / norms).astype(np.float32)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    """Mean fraction of the true k nearest neighbours that were retrieved.

    Compared as *sets*, not sequences: an ANN index that returns the right
    neighbours in a different order has still found them, and ties at equal
    distance make the ordering arbitrary anyway.
    """
    found = np.asarray(found)
    truth = np.asarray(truth)
    if found.shape[0] != truth.shape[0]:
        raise ValueError(
            f"{found.shape[0]} result rows vs {truth.shape[0]} ground-truth rows"
        )
    if k is None:
        k = found.shape[1]

    hits = 0
    for row_found, row_truth in zip(found[:, :k], truth[:, :k]):
        hits += len(set(row_found.tolist()) & set(row_truth.tolist()))
    return hits / (found.shape[0] * k)


# --------------------------------------------------------------------------
# Hybrid: vectors + scalar attributes
# --------------------------------------------------------------------------


def threshold_for_selectivity(values: np.ndarray, selectivity: float) -> float:
    """Value ``t`` such that ``values < t`` matches ``selectivity`` of rows.

    Returned as a threshold rather than a mask so the same number can be
    pushed into a real query predicate. Exact for continuous attributes; with
    duplicates the achieved fraction can only land on a tie boundary, so check
    with ``(values < t).mean()`` if that matters.
    """
    if not 0.0 <= selectivity <= 1.0:
        raise ValueError(f"selectivity must be in [0, 1], got {selectivity}")
    n = values.shape[0]
    k = int(round(selectivity * n))
    if k <= 0:
        return float(np.min(values))
    if k >= n:
        return float(np.nextafter(np.max(values), np.inf))
    return float(np.sort(values)[k])


def make_hybrid(
    vectors: VectorDataset | None = None,
    seed: int = 0,
    n_categories: int = 16,
) -> HybridDataset:
    """Attach scalar attributes to a vector dataset, for the query planner.

    No small public dataset carries both embeddings and rich scalar metadata,
    so the standard move in the filtered-ANN literature (and what the NeurIPS
    BigANN filtered track does) is to synthesise attributes over real vectors.
    That is what this does: the vectors stay real, the predicates become
    controllable.

    Attributes:
      ``price``      continuous, uniform in [0, 1000) -- pair with
                     ``threshold_for_selectivity`` to hit an exact selectivity
      ``timestamp``  continuous, ascending-ish epoch seconds
      ``category``   discrete, ``n_categories`` levels, roughly equal sized

    Selectivity is the axis the planner lives or dies on: a very selective
    predicate should make it filter first and scan the survivors, while a
    permissive one should make it search the vector index and filter after.
    Sweeping it is the experiment that shows the cost model works.
    """
    if vectors is None:
        vectors = random_vectors(n=10_000, dim=128, seed=seed, n_clusters=50)

    n = vectors.n
    rng = np.random.default_rng(seed)

    price = rng.uniform(0.0, 1000.0, size=n).astype(np.float64)
    timestamp = np.sort(rng.uniform(1.6e9, 1.7e9, size=n)).astype(np.float64)
    rng.shuffle(timestamp)  # ascending draw, shuffled so row order carries no signal
    category = rng.integers(0, n_categories, size=n, dtype=np.int64)

    return HybridDataset(
        vectors=vectors,
        keys=np.arange(n, dtype=np.int64),
        attributes={"price": price, "timestamp": timestamp, "category": category},
        name=f"hybrid-{vectors.name}",
    )
