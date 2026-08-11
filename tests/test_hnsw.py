"""Python-side tests for the HNSW graph index.

The C++ suite carries the structural coverage. What matters here is recall
measured through the same helpers the benchmarks use — `compute_ground_truth`
and `recall_at_k` in hylis.datasets — and, where SIFT is available, against
neighbour lists this project had no hand in producing.

Note the shape of these assertions. HNSW is approximate, so there is nothing
to assert equality against: they check that recall clears a floor and that it
never falls when ef rises. That is the strongest honest claim available for
an index allowed to miss neighbours.
"""

import numpy as np
import pytest

from hylis import FlatIndex, HnswIndex, Metric
from hylis import datasets as ds


@pytest.fixture(scope="module")
def corpus():
    # Clustered: uniform random points in high dimensions are all roughly
    # equidistant, so a graph index has no structure to exploit and recall
    # measured on them means nothing.
    return ds.random_vectors(n=4000, dim=32, n_queries=100, seed=0, n_clusters=40)


@pytest.fixture(scope="module")
def graph(corpus):
    index = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    index.reserve(corpus.n)
    index.add_batch(corpus.base)
    return index


# --------------------------------------------------------------------------
# Bridge
# --------------------------------------------------------------------------


def test_construction_and_repr(graph, corpus):
    assert len(graph) == corpus.n
    assert graph.dim == 32
    assert graph.M == 16
    assert graph.ef_construction == 200
    assert graph.metric == Metric.L2
    assert "HnswIndex(" in repr(graph)


def test_add_returns_sequential_ids():
    index = HnswIndex(3)
    assert index.add(np.array([1, 2, 3], dtype=np.float32)) == 0
    assert index.add(np.array([4, 5, 6], dtype=np.float32)) == 1
    assert len(index) == 2


def test_float64_and_list_input_accepted():
    index = HnswIndex(2)
    index.add_batch(np.array([[0, 0], [3, 4]], dtype=np.float64))
    index.add([1.0, 1.0])
    assert len(index) == 3


def test_search_returns_neighbours_best_first(graph, corpus):
    found = graph.search(corpus.queries[0], k=10, ef=50)
    assert len(found) == 10
    scores = [n.score for n in found]
    assert scores == sorted(scores)


def test_search_batch_shapes(graph, corpus):
    ids, scores = graph.search_batch(corpus.queries[:5], k=10, ef=50)
    assert ids.shape == (5, 10)
    assert scores.shape == (5, 10)
    assert ids.dtype == np.int64
    assert scores.dtype == np.float32


def test_vector_at_round_trips():
    index = HnswIndex(3)
    v = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    index.add(v)
    assert np.array_equal(index.vector_at(0), v)


def test_empty_index():
    index = HnswIndex(4)
    assert len(index) == 0
    assert index.entry_point == -1
    assert index.search(np.zeros(4, dtype=np.float32), 5) == []
    index.validate()


def test_clear_resets(graph_free=None):
    index = HnswIndex(8)
    v = ds.random_vectors(n=200, dim=8, n_queries=1, seed=1, n_clusters=5)
    index.add_batch(v.base)
    index.clear()
    assert len(index) == 0
    assert index.entry_point == -1
    index.validate()


def test_stats_are_exposed(graph, corpus):
    stats = graph.stats()
    assert stats.nodes == corpus.n
    assert stats.levels >= 2
    assert stats.layer_population[0] == corpus.n
    assert stats.max_degree_l0 <= 32, "layer 0 cap is 2*M"
    assert stats.total_bytes > stats.graph_bytes
    assert "HnswStats(" in repr(stats)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        HnswIndex(0)
    with pytest.raises(ValueError):
        HnswIndex(4, Metric.L2, M=1)
    with pytest.raises(ValueError):
        HnswIndex(4, Metric.L2, 16, 0)


def test_wrong_shapes_rejected(graph):
    with pytest.raises(ValueError, match="expected 32 values"):
        graph.search(np.zeros(5, dtype=np.float32), 1)
    with pytest.raises(ValueError, match="1-D"):
        graph.search(np.zeros((2, 32), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="2-D"):
        graph.search_batch(np.zeros(32, dtype=np.float32), 1)


def test_out_of_range_ids_rejected(graph):
    with pytest.raises(IndexError):
        graph.vector_at(999999)
    with pytest.raises(IndexError):
        graph.search_filtered(np.zeros(32, dtype=np.float32), 1, [999999])


# --------------------------------------------------------------------------
# Recall — bounds and monotonicity, never equality
# --------------------------------------------------------------------------


def test_recall_clears_the_floor_at_default_parameters(graph, corpus):
    """M=16, efC=200, ef=100 is the configuration the literature reports
    ~0.95+ recall@10 for."""
    ids, _ = graph.search_batch(corpus.queries, k=10, ef=100)
    truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=10)
    assert ds.recall_at_k(ids, truth, k=10) >= 0.95


def test_recall_never_falls_as_ef_rises(graph, corpus):
    """The one property a user actually relies on: ef is the quality knob."""
    truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=10)

    previous = 0.0
    for ef in (10, 20, 50, 100, 200):
        ids, _ = graph.search_batch(corpus.queries, k=10, ef=ef)
        recall = ds.recall_at_k(ids, truth, k=10)
        assert recall >= previous - 1e-9, f"recall fell at ef={ef}"
        previous = recall
    assert previous > 0.99


def test_visits_far_fewer_nodes_than_a_full_scan(graph, corpus):
    """The entire justification for the structure."""
    visited = []
    for query in corpus.queries[:20]:
        graph.search(query, k=10, ef=50)
        visited.append(graph.last_visited)

    mean = float(np.mean(visited))
    assert mean < corpus.n / 5, f"visited {mean:.0f} of {corpus.n}"


def test_reachability_is_complete_at_sensible_parameters(graph, corpus):
    assert graph.reachable() == corpus.n
    assert graph.stats().reachable == corpus.n


def test_self_retrieval(graph, corpus):
    hits = 0
    probes = range(0, corpus.n, 41)
    for i in probes:
        found = graph.search(graph.vector_at(i), k=1, ef=50)
        if found and found[0].id == i:
            hits += 1
    assert hits >= 0.99 * len(list(probes))


def test_matches_flat_index_closely(graph, corpus):
    """The exact oracle from module 3 is what makes recall meaningful."""
    exact = FlatIndex(corpus.dim)
    exact.add_batch(corpus.base)

    exact_ids, _ = exact.search_batch(corpus.queries, k=10)
    graph_ids, _ = graph.search_batch(corpus.queries, k=10, ef=200)
    assert ds.recall_at_k(graph_ids, exact_ids, k=10) >= 0.99


def test_heuristic_selection_is_at_least_as_good_as_naive(corpus):
    truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=10)

    def recall_with(use_heuristic):
        index = HnswIndex(corpus.dim, Metric.L2, M=8, ef_construction=100)
        index.use_heuristic = use_heuristic
        index.add_batch(corpus.base)
        ids, _ = index.search_batch(corpus.queries, k=10, ef=20)
        return ds.recall_at_k(ids, truth, k=10)

    assert recall_with(True) >= recall_with(False)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_cosine_is_scale_invariant(corpus):
    index = HnswIndex(corpus.dim, Metric.Cosine)
    index.add_batch(corpus.base)

    query = corpus.queries[0]
    a = [n.id for n in index.search(query, k=10, ef=100)]
    b = [n.id for n in index.search((query * 31.0).astype(np.float32), k=10, ef=100)]
    assert a == b


def test_cosine_matches_the_angular_oracle(corpus):
    index = HnswIndex(corpus.dim, Metric.Cosine)
    index.add_batch(corpus.base)

    ids, _ = index.search_batch(corpus.queries, k=10, ef=200)
    truth = ds.compute_ground_truth(corpus.base, corpus.queries, k=10, metric="angular")
    assert ds.recall_at_k(ids, truth, k=10) >= 0.9


def test_inner_product_prefers_larger_dots():
    index = HnswIndex(2, Metric.InnerProduct)
    index.add_batch(np.array([[1, 0], [5, 0], [2, 0]], dtype=np.float32))
    found = index.search(np.array([1, 0], dtype=np.float32), k=3, ef=10)
    assert found[0].id == 1
    assert found[0].score == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Filtered search
# --------------------------------------------------------------------------


def test_filtered_search_returns_only_allowed(graph, corpus):
    allowed = list(range(0, corpus.n, 3))
    allowed_set = set(allowed)

    for query in corpus.queries[:10]:
        found = graph.search_filtered(query, 10, allowed, ef=100)
        assert found
        assert all(n.id in allowed_set for n in found)
        scores = [n.score for n in found]
        assert scores == sorted(scores)


def test_empty_filter_returns_nothing(graph, corpus):
    assert graph.search_filtered(corpus.queries[0], 5, []) == []


def test_filtering_everything_matches_an_unfiltered_search(graph, corpus):
    everything = list(range(corpus.n))
    for query in corpus.queries[:5]:
        a = [n.id for n in graph.search(query, 10, ef=50)]
        b = [n.id for n in graph.search_filtered(query, 10, everything, ef=50)]
        assert a == b


def test_tighter_filters_cost_more_traversal(graph, corpus):
    """Not a defect — the mechanism the query planner exists to exploit.

    The graph must step *through* non-matching nodes to stay connected, so a
    tighter predicate means visiting more of the corpus to collect the same k,
    while a filtered exhaustive scan gets cheaper in proportion. Those two
    curves cross somewhere, and that crossover is what module 8 must predict.
    """
    query = corpus.queries[0]
    costs = {}
    for step in (2, 10, 100):
        allowed = list(range(0, corpus.n, step))
        graph.search_filtered(query, 10, allowed, ef=50)
        costs[step] = graph.last_visited

    assert costs[100] > costs[10] > costs[2]


def test_filtered_results_agree_with_a_filtered_exact_scan(graph, corpus):
    """Recall, not equality — but over the surviving subset."""
    exact = FlatIndex(corpus.dim)
    exact.add_batch(corpus.base)

    allowed = list(range(0, corpus.n, 4))
    hits = 0
    for query in corpus.queries[:20]:
        approx = {n.id for n in graph.search_filtered(query, 10, allowed, ef=200)}
        truth = {n.id for n in exact.search_filtered(query, 10, allowed)}
        hits += len(approx & truth)
    assert hits >= 0.9 * 20 * 10


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sift():
    try:
        return ds.load_sift("siftsmall")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.mark.sift
class TestSift:
    def test_clears_the_recall_floor_on_real_vectors(self, sift):
        """The headline claim, on data with genuine structure and against
        neighbour lists TEXMEX computed years before this project existed."""
        index = HnswIndex(sift.dim, Metric.L2, M=16, ef_construction=200)
        index.reserve(sift.n)
        index.add_batch(sift.base)
        index.validate()
        assert index.reachable() == sift.n

        ids, _ = index.search_batch(sift.queries, k=10, ef=100)
        recall = ds.recall_at_k(ids, sift.ground_truth[:, :10], k=10)
        assert recall >= 0.95, f"recall@10 was {recall:.4f}"

    def test_visits_a_small_fraction_of_the_corpus(self, sift):
        index = HnswIndex(sift.dim, Metric.L2, M=16, ef_construction=200)
        index.add_batch(sift.base)

        visited = []
        for query in sift.queries[:50]:
            index.search(query, k=10, ef=100)
            visited.append(index.last_visited)
        assert float(np.mean(visited)) < sift.n / 5
