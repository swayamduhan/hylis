"""Tests for the neural router: clustering, training, and the language boundary.

The load-bearing test here is `TestCrossLanguage`. Training happens in numpy
and inference happens in C++, so a transposed weight matrix or a mismatched
standardisation would produce a router that loads cleanly, runs at full speed,
routes badly, and quietly degrades every downstream number. Nothing else in
the suite would catch it — so the two forward passes are compared directly.
"""

import numpy as np
import pytest

from hylis import FlatIndex, HnswIndex, Metric
from hylis import datasets as ds
from hylis._hnsw import NeuralRouter
from hylis.router import (
    NearestCentroidRouter,
    RouterMLP,
    build_training_set,
    kmeans,
    medoids_for,
    router_json,
)


@pytest.fixture(scope="module")
def corpus():
    return ds.random_vectors(n=4000, dim=24, n_queries=100, seed=0, n_clusters=40)


@pytest.fixture(scope="module")
def clustering(corpus):
    return kmeans(corpus.base, n_clusters=48, seed=0)


@pytest.fixture(scope="module")
def trained(corpus, clustering):
    ts = build_training_set(corpus.base, clustering.assignment,
                            n_samples=20000, seed=0)
    model = RouterMLP(dim=corpus.dim, clusters=48, hidden=48, seed=0)
    model.fit(ts.x_train, ts.y_train, epochs=25)
    return model, ts


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def test_kmeans_shapes_and_coverage(corpus, clustering):
    assert clustering.centroids.shape == (48, corpus.dim)
    assert clustering.assignment.shape == (corpus.n,)
    assert clustering.medoids.shape == (48,)
    assert clustering.assignment.min() >= 0
    assert clustering.assignment.max() < 48
    assert clustering.medoids.min() >= 0
    assert clustering.medoids.max() < corpus.n


def test_kmeans_is_deterministic(corpus):
    a = kmeans(corpus.base, n_clusters=16, seed=3)
    b = kmeans(corpus.base, n_clusters=16, seed=3)
    c = kmeans(corpus.base, n_clusters=16, seed=4)
    assert np.array_equal(a.assignment, b.assignment)
    assert not np.array_equal(a.assignment, c.assignment)


def test_kmeans_leaves_no_empty_clusters(corpus, clustering):
    """An empty cluster is a wasted entry point — its medoid would be a node
    nothing routes to."""
    counts = np.bincount(clustering.assignment, minlength=48)
    assert (counts > 0).all()


def test_medoids_are_real_nodes_near_their_centroid(corpus, clustering):
    for c in range(clustering.centroids.shape[0]):
        medoid = clustering.medoids[c]
        members = np.flatnonzero(clustering.assignment == c)
        if members.size == 0:
            continue
        distances = ((corpus.base[members] - clustering.centroids[c]) ** 2).sum(axis=1)
        assert distances.min() == pytest.approx(
            float(((corpus.base[medoid] - clustering.centroids[c]) ** 2).sum()), rel=1e-5
        )


def test_kmeans_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        kmeans(np.zeros((0, 4), dtype=np.float32), n_clusters=2)


def test_more_clusters_than_vectors_is_clamped():
    small = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    result = kmeans(small, n_clusters=50, seed=0)
    assert result.centroids.shape[0] == 5


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------


def test_training_set_splits_and_labels(corpus, clustering):
    ts = build_training_set(corpus.base, clustering.assignment,
                            n_samples=5000, seed=0, val_fraction=0.2)
    assert ts.x_train.shape[1] == corpus.dim
    assert len(ts.x_train) + len(ts.x_val) == 5000
    assert len(ts.x_val) == pytest.approx(1000, abs=2)
    assert ts.y_train.min() >= 0 and ts.y_train.max() < 48


def test_training_inputs_are_jittered_not_copied(corpus, clustering):
    """A router trained only on exact stored vectors would never have seen a
    point that is not already in the corpus — which is every real query."""
    ts = build_training_set(corpus.base, clustering.assignment,
                            n_samples=500, jitter=0.05, seed=0)
    exact_matches = sum(
        1 for row in ts.x_train if np.any(np.all(corpus.base == row, axis=1))
    )
    assert exact_matches < len(ts.x_train) * 0.05


def test_cheap_labels_agree_with_exact_nearest_neighbour_labels(corpus, clustering):
    """Pins the cost trade documented in build_training_set.

    Labels use the cluster of the vector a sample was derived from, rather than
    re-deriving the true nearest neighbour with FlatIndex — which at 1M vectors
    would cost more than the training does. That is only sound if the two agree,
    which is checkable here where the corpus is small enough to do both.
    """
    ts = build_training_set(corpus.base, clustering.assignment,
                            n_samples=1000, jitter=0.05, seed=0, val_fraction=0.0)

    exact = FlatIndex(corpus.dim)
    exact.add_batch(corpus.base)
    nn_ids, _ = exact.search_batch(ts.x_train, k=1)
    exact_labels = clustering.assignment[nn_ids[:, 0]]

    agreement = float((exact_labels == ts.y_train).mean())
    assert agreement > 0.9, f"cheap labels agreed with exact ones only {agreement:.1%}"


def test_training_set_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        build_training_set(np.zeros((0, 4), dtype=np.float32),
                           np.zeros(0, dtype=np.int32))


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def test_training_reduces_loss_and_generalises(trained):
    model, ts = trained
    assert model.losses[-1] < model.losses[0] / 5, "training should actually train"
    accuracy = float((model.predict(ts.x_val) == ts.y_val).mean())
    assert accuracy > 0.7, f"validation accuracy only {accuracy:.3f}"


def test_untrained_router_is_much_worse(corpus, clustering, trained):
    """The minimum bar for 'training did something'."""
    model, ts = trained
    untrained = RouterMLP(dim=corpus.dim, clusters=48, hidden=48, seed=1)
    untrained.mean = model.mean
    untrained.scale = model.scale

    trained_acc = float((model.predict(ts.x_val) == ts.y_val).mean())
    random_acc = float((untrained.predict(ts.x_val) == ts.y_val).mean())
    assert trained_acc > random_acc * 3


def test_backward_pass_matches_finite_differences():
    """The from-scratch gradient, checked rather than trusted.

    A subtly wrong hand-derived gradient still trains, just badly — which would
    make the entire linear-vs-neural comparison measure a bug.
    """
    rng = np.random.default_rng(0)
    model = RouterMLP(dim=4, clusters=3, hidden=5, seed=1)
    x = rng.normal(size=(7, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=7)

    def loss():
        logits, _ = model._forward(x)
        probs = model._softmax(logits)
        return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

    logits, cache = model._forward(x)
    probs = model._softmax(logits)
    dlogits = probs.copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= float(len(y))
    grads = dict(zip(["w1", "b1", "w2", "b2"], model._backward(cache, dlogits)))

    eps = 1e-3
    for name in ["w1", "b1", "w2", "b2"]:
        param = getattr(model, name)
        flat = param.ravel()
        analytic = grads[name].reshape(param.shape).ravel()
        for i in range(flat.size):
            original = float(flat[i])
            flat[i] = original + eps
            up = loss()
            flat[i] = original - eps
            down = loss()
            flat[i] = original
            numeric = (up - down) / (2 * eps)
            assert numeric == pytest.approx(analytic[i], abs=2e-3), f"{name}[{i}]"


def test_softmax_does_not_overflow():
    model = RouterMLP(dim=2, clusters=3, hidden=2, seed=0)
    probs = model._softmax(np.array([[1000.0, 999.0, -1000.0]], dtype=np.float32))
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0)


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        RouterMLP(dim=0, clusters=4)
    with pytest.raises(ValueError):
        RouterMLP(dim=4, clusters=0)
    with pytest.raises(ValueError):
        RouterMLP(dim=4, clusters=4, hidden=0)


def test_nearest_centroid_router_needs_no_training(corpus, clustering):
    """The honest opponent: routing with no network at all."""
    baseline = NearestCentroidRouter(clustering.centroids)
    predicted = baseline.predict(corpus.base[:500])
    assert np.array_equal(predicted, clustering.assignment[:500])


# --------------------------------------------------------------------------
# The language boundary
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair(clustering, trained):
    model, _ = trained
    return model, NeuralRouter.from_json(router_json(model, clustering.medoids))


class TestCrossLanguage:
    """Python trains, C++ infers. They must compute the same function."""

    def test_shape_survives_export(self, pair, corpus):
        model, cpp = pair
        assert cpp.dim == corpus.dim
        assert cpp.hidden == model.hidden
        assert cpp.clusters == model.clusters

    def test_logits_match(self, pair, corpus):
        model, cpp = pair
        queries = corpus.queries.astype(np.float32)
        py_logits = model.logits(queries)
        for i in range(len(queries)):
            got = cpp.logits(queries[i])
            assert np.allclose(got, py_logits[i], atol=1e-3), f"query {i}"

    def test_predicted_cluster_matches_on_every_query(self, pair, corpus):
        """The assertion that actually matters — a transposed matrix survives a
        loose tolerance on logits but not this."""
        model, cpp = pair
        queries = corpus.queries.astype(np.float32)
        py_pred = model.predict(queries)
        mismatches = sum(
            1 for i in range(len(queries))
            if int(np.argmax(cpp.logits(queries[i]))) != int(py_pred[i])
        )
        assert mismatches == 0, f"{mismatches}/{len(queries)} disagreed"

    def test_entry_points_are_the_medoids_of_the_predicted_clusters(self, pair,
                                                                    clustering, corpus):
        model, cpp = pair
        query = corpus.queries[0].astype(np.float32)
        clusters = cpp.predict(query, 2)
        entries = cpp.entry_points(query, 2)
        assert set(entries) <= {int(clustering.medoids[c]) for c in clusters}


# --------------------------------------------------------------------------
# Inside a search
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph(corpus, clustering, trained):
    model, _ = trained
    index = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    index.add_batch(corpus.base)
    index.set_router(NeuralRouter.from_json(router_json(model, clustering.medoids)))
    return index


def test_router_attaches(graph):
    assert graph.has_router
    assert graph.router().clusters == 48


def test_incompatible_router_rejected(corpus, clustering, trained):
    model, _ = trained
    other = HnswIndex(corpus.dim + 1)
    other.add_batch(np.zeros((10, corpus.dim + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        other.set_router(NeuralRouter.from_json(router_json(model, clustering.medoids)))


def test_routed_search_is_correct(graph, corpus):
    for query in corpus.queries[:20]:
        found = graph.search(query, k=10, ef=50, use_router=True)
        assert len(found) == 10
        ids = [n.id for n in found]
        assert len(set(ids)) == 10
        assert all(0 <= i < corpus.n for i in ids)
        scores = [n.score for n in found]
        assert scores == sorted(scores)


def test_routing_costs_no_graph_visits(graph, corpus):
    """The measurement that makes the comparison interpretable: the descent
    spends graph traversal to reach layer 0, the router spends arithmetic."""
    query = corpus.queries[0]
    graph.search(query, k=10, ef=50, use_router=False)
    descent = graph.last_routing_visited
    graph.search(query, k=10, ef=50, use_router=True)
    assert graph.last_routing_visited == 0
    assert descent > 0


def test_router_does_not_hurt_recall_at_low_ef(graph, corpus):
    """Recorded as a measurement, not demanded as a requirement.

    The entry point matters most when the beam is narrow, because a narrow beam
    has little room to recover from a bad start. If the router ever falls well
    behind the descent here, the idea is not working and the number should say
    so.
    """
    truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=10)

    descent_ids, _ = graph.search_batch(corpus.queries, k=10, ef=20, use_router=False)
    routed_ids, _ = graph.search_batch(corpus.queries, k=10, ef=20, use_router=True)

    descent = ds.recall_at_k(descent_ids, truth, k=10)
    routed = ds.recall_at_k(routed_ids, truth, k=10)
    assert routed > descent - 0.05, (
        f"router recall {routed:.4f} vs descent {descent:.4f} — the router is "
        "materially worse, which is a finding rather than a flake"
    )


def test_top_p_and_global_entry_are_settable(graph, corpus):
    graph.router_top_p = 4
    assert graph.router_top_p == 4
    graph.router_keeps_global_entry = True
    assert graph.router_keeps_global_entry
    assert len(graph.search(corpus.queries[0], k=10, ef=50, use_router=True)) == 10
    graph.router_top_p = 2
    graph.router_keeps_global_entry = False


def test_flat_only_graph_has_no_upper_layers(corpus):
    flat_graph = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200,
                           flat_only=True)
    flat_graph.add_batch(corpus.base)
    stats = flat_graph.stats()
    assert stats.levels == 1
    assert stats.layer_population == [corpus.n]
    assert stats.edges == stats.layer0_edges
    flat_graph.validate()


def test_flat_only_saves_graph_memory(corpus):
    full = HnswIndex(corpus.dim, Metric.L2, 16, 200, 100, False)
    flat_graph = HnswIndex(corpus.dim, Metric.L2, 16, 200, 100, True)
    full.add_batch(corpus.base)
    flat_graph.add_batch(corpus.base)
    assert flat_graph.stats().graph_bytes < full.stats().graph_bytes
