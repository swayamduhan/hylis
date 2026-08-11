"""Training for the neural router — the learned replacement for HNSW's descent.

What the router has to do
-------------------------
HNSW's upper layers exist for one purpose: hand the layer-0 beam search a
starting node near the query. The router replaces that walk with a forward
pass.

It cannot regress a node id directly — ids are arbitrary labels with no
geometry, so there is no continuous target to fit. Instead the corpus is
partitioned by k-means into C clusters, each with a precomputed *medoid* (the
real node nearest its centroid), and the router is a classifier from query
vector to cluster. The medoids of its top-p clusters become entry points.

The baseline the network has to beat
------------------------------------
Assigning a query to its nearest centroid is already a router, needs no
training at all, and costs C distance computations — comparable to a small
MLP's forward pass. `NearestCentroidRouter` implements it precisely so the
neural router has an honest opponent rather than only the descent.

The MLP's opportunity is that nearest-centroid and *cluster-of-true-nearest-
neighbour* are not the same function: Voronoi cells drawn around centroids do
not line up exactly with nearest-neighbour structure. Learning the second
while paying the cost of the first is the entire thesis. If the MLP merely
reproduces nearest-centroid, it has learned nothing worth its weights, and
`scripts/train_router.py` reports exactly that comparison.

Everything here is written on numpy by hand, matching the project's
from-scratch constraint; the forward/backward/Adam pattern follows
`learned.py`, whose backward pass is gradient-checked against finite
differences.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KMeansResult",
    "RouterMLP",
    "NearestCentroidRouter",
    "TrainingSet",
    "kmeans",
    "medoids_for",
    "build_training_set",
    "export_router",
    "router_json",
]


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KMeansResult:
    centroids: np.ndarray      # (C, dim) float32
    assignment: np.ndarray     # (n,) int32 — cluster of every base vector
    medoids: np.ndarray        # (C,) int32 — the node standing in for each cluster
    iterations: int
    seconds: float


def _assign(vectors: np.ndarray, centroids: np.ndarray,
            chunk: int = 8192) -> np.ndarray:
    """Nearest centroid for every row, chunked to bound peak memory.

    Uses the expanded ||c||^2 - 2 v.c form so the work becomes one matmul that
    BLAS can drive; the dropped ||v||^2 term is constant per row and cannot
    change which centroid wins.
    """
    out = np.empty(vectors.shape[0], dtype=np.int32)
    centroid_sq = (centroids * centroids).sum(axis=1)
    for start in range(0, vectors.shape[0], chunk):
        block = vectors[start : start + chunk]
        scores = centroid_sq[None, :] - 2.0 * (block @ centroids.T)
        out[start : start + block.shape[0]] = np.argmin(scores, axis=1)
    return out


def kmeans(vectors: np.ndarray, n_clusters: int, seed: int = 0,
           iterations: int = 25, sample_limit: int = 100_000) -> KMeansResult:
    """k-means with k-means++ initialisation.

    Centroids are fitted on a subsample when the corpus is large — the extra
    accuracy from using every point is not worth minutes of wall clock, and
    every point is still assigned to a final centroid afterwards.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if n == 0:
        raise ValueError("kmeans: no vectors")
    n_clusters = int(min(n_clusters, n))
    if n_clusters < 1:
        raise ValueError(f"kmeans: n_clusters must be >= 1, got {n_clusters}")

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    fit_on = vectors
    if n > sample_limit:
        fit_on = vectors[rng.choice(n, size=sample_limit, replace=False)]

    # k-means++: each new centre is drawn with probability proportional to its
    # squared distance from the nearest existing one. Costs one extra pass per
    # centre and removes the dead clusters random init routinely produces.
    centroids = np.empty((n_clusters, vectors.shape[1]), dtype=np.float32)
    centroids[0] = fit_on[rng.integers(fit_on.shape[0])]
    closest_sq = ((fit_on - centroids[0]) ** 2).sum(axis=1)
    for c in range(1, n_clusters):
        total = float(closest_sq.sum())
        if total <= 0.0:  # every point already coincides with a centre
            centroids[c] = fit_on[rng.integers(fit_on.shape[0])]
        else:
            pick = int(rng.choice(fit_on.shape[0], p=closest_sq / total))
            centroids[c] = fit_on[pick]
        closest_sq = np.minimum(closest_sq, ((fit_on - centroids[c]) ** 2).sum(axis=1))

    ran = 0
    for ran in range(1, iterations + 1):
        assignment = _assign(fit_on, centroids)
        moved = 0.0
        for c in range(n_clusters):
            members = fit_on[assignment == c]
            if members.shape[0] == 0:
                # An empty cluster is a wasted entry point. Re-seed it on the
                # point currently worst served, which is where an extra centre
                # helps most.
                worst = int(np.argmax(((fit_on - centroids[assignment]) ** 2).sum(axis=1)))
                centroids[c] = fit_on[worst]
                continue
            new_centre = members.mean(axis=0)
            moved += float(np.abs(new_centre - centroids[c]).sum())
            centroids[c] = new_centre
        if moved == 0.0:
            break

    assignment = _assign(vectors, centroids)
    return KMeansResult(
        centroids=centroids,
        assignment=assignment,
        medoids=medoids_for(vectors, centroids, assignment),
        iterations=ran,
        seconds=time.perf_counter() - start_time,
    )


def medoids_for(vectors: np.ndarray, centroids: np.ndarray,
                assignment: np.ndarray) -> np.ndarray:
    """The real node nearest each centroid.

    A centroid is an average and usually not a stored vector, so it cannot be
    an entry point — the graph search has to start at an actual node. The
    medoid is the closest thing the corpus has to the centroid.
    """
    medoids = np.zeros(centroids.shape[0], dtype=np.int32)
    for c in range(centroids.shape[0]):
        members = np.flatnonzero(assignment == c)
        if members.size == 0:
            # Empty cluster: fall back to whichever node is globally nearest,
            # so the medoid is always a valid id even in degenerate cases.
            medoids[c] = int(np.argmin(((vectors - centroids[c]) ** 2).sum(axis=1)))
            continue
        distances = ((vectors[members] - centroids[c]) ** 2).sum(axis=1)
        medoids[c] = int(members[int(np.argmin(distances))])
    return medoids


# --------------------------------------------------------------------------
# Training data — generated, never collected
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSet:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


def build_training_set(vectors: np.ndarray, assignment: np.ndarray,
                       n_samples: int = 40_000, jitter: float = 0.05,
                       seed: int = 0, val_fraction: float = 0.2,
                       extra_queries: np.ndarray | None = None,
                       extra_labels: np.ndarray | None = None,
                       exact_index=None) -> TrainingSet:
    """Build (query, cluster) pairs from the corpus itself.

    There is nothing to collect and nothing to label by hand. Inputs are base
    vectors with Gaussian jitter — real queries are never exactly on the data
    manifold, and a router trained only on exact stored vectors would not have
    seen that.

    Labels come one of two ways, and the difference matters more than it looks:

    ``exact_index`` given (a ``FlatIndex``)
        The label is the cluster of the sample's **true nearest neighbour**.
        This is the target worth learning, because it is *not* the same
        function as "nearest centroid" — Voronoi cells drawn around centroids
        do not line up exactly with nearest-neighbour structure. A network
        that learns this has learned something `NearestCentroidRouter` cannot
        do for free.

    ``exact_index`` omitted
        The label is the cluster of the vector the sample was derived from.
        Free, but for small jitter it is very nearly k-means assignment — so
        the network is being asked to imitate nearest-centroid, and can at
        best tie with it. Use this only when an exact search is unaffordable
        (it costs n_samples x len(corpus) distance computations).

    `tests/test_router.py` pins how closely the two labellings agree on a
    corpus small enough to compute both.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if n == 0:
        raise ValueError("build_training_set: no vectors")
    rng = np.random.default_rng(seed)

    picks = rng.integers(0, n, size=n_samples)
    scale = jitter * float(np.std(vectors)) if jitter > 0 else 0.0
    x = vectors[picks]
    if scale > 0:
        x = x + rng.normal(0.0, scale, size=x.shape).astype(np.float32)

    if exact_index is not None:
        nearest, _ = exact_index.search_batch(x, k=1)
        y = assignment[nearest[:, 0]].astype(np.int64)
    else:
        y = assignment[picks].astype(np.int64)

    if extra_queries is not None and extra_labels is not None and len(extra_queries):
        x = np.vstack([x, np.ascontiguousarray(extra_queries, dtype=np.float32)])
        y = np.concatenate([y, np.asarray(extra_labels, dtype=np.int64)])

    order = rng.permutation(x.shape[0])
    x, y = x[order], y[order]
    split = int(x.shape[0] * (1.0 - val_fraction))
    return TrainingSet(x[:split], y[:split], x[split:], y[split:])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class NearestCentroidRouter:
    """Routing with no network at all: pick the nearest centroid.

    The honest opponent. It needs no training, and its C distance computations
    cost about what a small MLP's forward pass does — so a neural router that
    does not beat it has not earned its complexity.
    """

    name = "nearest-centroid"

    def __init__(self, centroids: np.ndarray) -> None:
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        # Negated distance, so "larger is better" matches the MLP's logits.
        return -((self.centroids * self.centroids).sum(axis=1)[None, :]
                 - 2.0 * (x @ self.centroids.T))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x), axis=1)


class RouterMLP:
    """dim -> hidden -> clusters, ReLU, softmax, cross-entropy, Adam.

    Written out by hand. The layout is chosen to match
    cpp/hylis/index/router.hpp exactly: weights are row-major, so exporting is
    a plain ravel() with no transpose, and the C++ forward pass indexes
    w1[i*hidden + h] over the same values.
    """

    name = "mlp"

    def __init__(self, dim: int, clusters: int, hidden: int = 64,
                 lr: float = 0.01, seed: int = 0) -> None:
        if dim < 1 or clusters < 1 or hidden < 1:
            raise ValueError("RouterMLP: dim, clusters and hidden must all be >= 1")
        self.dim, self.clusters, self.hidden = dim, clusters, hidden
        self.lr = lr
        self.seed = seed
        self.losses: list[float] = []
        self.mean = np.zeros(dim, dtype=np.float32)
        self.scale = 1.0
        self._init_parameters()

    def _init_parameters(self) -> None:
        rng = np.random.default_rng(self.seed)
        # He initialisation: variance 2/fan_in is what keeps ReLU activations
        # from collapsing toward zero as they pass through the layer.
        self.w1 = rng.normal(0, np.sqrt(2.0 / self.dim),
                             size=(self.dim, self.hidden)).astype(np.float32)
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / self.hidden),
                             size=(self.hidden, self.clusters)).astype(np.float32)
        self.b2 = np.zeros(self.clusters, dtype=np.float32)

    # -- forward / backward, separable so gradients can be checked ---------

    def _forward(self, x: np.ndarray):
        pre = x @ self.w1 + self.b1
        act = np.maximum(pre, 0.0)
        return act @ self.w2 + self.b2, (x, pre, act)

    def _backward(self, cache, dlogits: np.ndarray):
        x, pre, act = cache
        gw2 = act.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dact = dlogits @ self.w2.T
        dpre = dact * (pre > 0.0)
        gw1 = x.T @ dpre
        gb1 = dpre.sum(axis=0)
        return gw1, gb1, gw2, gb2

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # Shifted by the row max before exponentiating: logits reach the tens
        # here, and exp() of that overflows float32 into inf/nan.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        out, _ = self._forward(self._standardise(x))
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x), axis=1)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 30,
            batch_size: int = 256, verbose: bool = False) -> "RouterMLP":
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if x.shape[0] == 0:
            return self

        # Standardisation is folded into the model rather than asked of the
        # caller, because the C++ side has to apply the identical transform and
        # a mismatch would be invisible until recall quietly dropped.
        self.mean = x.mean(axis=0).astype(np.float32)
        self.scale = float(np.std(x)) or 1.0
        xs = self._standardise(x)

        rng = np.random.default_rng(self.seed)
        params = ["w1", "b1", "w2", "b2"]
        moment1 = {p: np.zeros_like(getattr(self, p)) for p in params}
        moment2 = {p: np.zeros_like(getattr(self, p)) for p in params}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        self.losses = []

        for epoch in range(epochs):
            order = rng.permutation(xs.shape[0])
            epoch_loss = 0.0
            batches = 0
            for start in range(0, xs.shape[0], batch_size):
                idx = order[start : start + batch_size]
                xb, yb = xs[idx], y[idx]
                logits, cache = self._forward(xb)
                probs = self._softmax(logits)

                # Cross-entropy, and the gradient it produces through softmax
                # collapses to (probs - onehot)/n -- the reason this pairing is
                # standard is that the two Jacobians cancel.
                n = float(xb.shape[0])
                epoch_loss += float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean())
                batches += 1
                dlogits = probs
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= n

                grads = dict(zip(params, self._backward(cache, dlogits)))
                step += 1
                for p in params:
                    g = grads[p].reshape(getattr(self, p).shape)
                    moment1[p] = beta1 * moment1[p] + (1 - beta1) * g
                    moment2[p] = beta2 * moment2[p] + (1 - beta2) * (g * g)
                    m_hat = moment1[p] / (1 - beta1 ** step)
                    v_hat = moment2[p] / (1 - beta2 ** step)
                    setattr(self, p, (getattr(self, p) -
                                      self.lr * m_hat / (np.sqrt(v_hat) + eps)
                                      ).astype(np.float32))

            self.losses.append(epoch_loss / max(batches, 1))
            if verbose:
                print(f"    epoch {epoch + 1:3d}/{epochs}  loss {self.losses[-1]:.4f}")
        return self


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def router_json(model: RouterMLP, medoids: np.ndarray) -> str:
    """Serialise weights in the exact layout cpp/hylis/index/router.hpp reads.

    Standardisation is folded into the first layer here rather than exported
    as separate fields:  (x - mean)/scale @ w1 + b1  ==  x @ (w1/scale) +
    (b1 - mean/scale @ w1). Doing the algebra once at export means the C++
    forward pass is a plain matmul with nothing to get subtly wrong, and one
    fewer thing that can silently disagree across the language boundary.
    """
    w1 = (model.w1 / model.scale).astype(np.float32)
    b1 = (model.b1 - (model.mean / model.scale) @ model.w1).astype(np.float32)

    return json.dumps({
        "version": 1,
        "dim": int(model.dim),
        "hidden": int(model.hidden),
        "clusters": int(model.clusters),
        "w1": [float(v) for v in w1.ravel()],
        "b1": [float(v) for v in b1.ravel()],
        "w2": [float(v) for v in model.w2.astype(np.float32).ravel()],
        "b2": [float(v) for v in model.b2.astype(np.float32).ravel()],
        "medoids": [int(v) for v in np.asarray(medoids).ravel()],
    })


def export_router(model: RouterMLP, medoids: np.ndarray, path: str) -> str:
    blob = router_json(model, medoids)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(blob)
    return blob
