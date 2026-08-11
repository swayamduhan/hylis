"""Python-side tests for the flat vector index.

Two jobs. The first is the usual bridge check: that numpy arrays cross into
C++ correctly. The second is the one that matters -- confirming the C++
engine's answers are identical to an independent implementation's, first
numpy's oracle and then, on real SIFT data, neighbour lists published by
someone else entirely.

Exhaustive search has exactly one right answer, so these are equality
assertions with no tolerance. HNSW will not get to be tested this way; it
will be graded against this.
"""

import numpy as np
import pytest

from hylis import FlatIndex, Metric
from hylis import datasets as ds


@pytest.fixture
def small():
    idx = FlatIndex(2)
    idx.add_batch(np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=np.float32))
    return idx


# --------------------------------------------------------------------------
# Bridge
# --------------------------------------------------------------------------


def test_construction_and_repr(small):
    assert small.dim == 2
    assert small.metric == Metric.L2
    assert len(small) == 4
    assert "FlatIndex(dim=2" in repr(small)


def test_add_returns_sequential_ids():
    idx = FlatIndex(2)
    assert idx.add(np.array([1, 2], dtype=np.float32)) == 0
    assert idx.add(np.array([3, 4], dtype=np.float32)) == 1
    assert len(idx) == 2


def test_search_returns_neighbours_best_first(small):
    got = small.search(np.array([0.9, 0.0], dtype=np.float32), k=2)
    assert [n.id for n in got] == [1, 0]
    assert got[0].score == pytest.approx(0.1)
    assert got[1].score == pytest.approx(0.9)


def test_neighbor_unpacks_as_a_pair(small):
    for nid, score in small.search(np.array([0, 0], dtype=np.float32), k=1):
        assert nid == 0
        assert score == pytest.approx(0.0)


def test_float64_input_is_accepted():
    """Python's default float is float64; forcing the user to cast every
    array would be a papercut on every single call."""
    idx = FlatIndex(2)
    idx.add_batch(np.array([[0, 0], [3, 4]], dtype=np.float64))
    got = idx.search(np.array([0, 0], dtype=np.float64), k=2)
    assert [n.id for n in got] == [0, 1]
    assert got[1].score == pytest.approx(5.0)


def test_list_input_is_accepted():
    idx = FlatIndex(2)
    idx.add([1.0, 2.0])
    assert len(idx) == 1


def test_search_batch_shapes_and_agreement(small):
    queries = np.array([[0, 0], [10, 10]], dtype=np.float32)
    ids, scores = small.search_batch(queries, k=3)
    assert ids.shape == (2, 3)
    assert scores.shape == (2, 3)
    assert ids.dtype == np.int64
    assert scores.dtype == np.float32
    assert ids[0].tolist() == [n.id for n in small.search(queries[0], 3)]
    assert ids[1][0] == 3


def test_search_batch_width_is_clamped_to_index_size(small):
    ids, scores = small.search_batch(np.zeros((2, 2), dtype=np.float32), k=99)
    assert ids.shape == (2, 4)


def test_vector_at_round_trips():
    idx = FlatIndex(3)
    v = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    idx.add(v)
    assert np.array_equal(idx.vector_at(0), v)


def test_clear_keeps_dim_and_metric():
    idx = FlatIndex(4, Metric.Cosine)
    idx.add(np.ones(4, dtype=np.float32))
    idx.clear()
    assert len(idx) == 0
    assert idx.dim == 4 and idx.metric == Metric.Cosine


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_wrong_dimension_rejected(small):
    with pytest.raises(ValueError, match="expected 2 values"):
        small.search(np.array([1, 2, 3], dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="expected 2 values"):
        small.add(np.array([1, 2, 3], dtype=np.float32))


def test_wrong_rank_rejected(small):
    with pytest.raises(ValueError, match="1-D"):
        small.search(np.zeros((2, 2), dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="2-D"):
        small.add_batch(np.zeros(4, dtype=np.float32))


def test_add_batch_column_mismatch_rejected(small):
    with pytest.raises(ValueError, match="expected 2 columns"):
        small.add_batch(np.zeros((3, 5), dtype=np.float32))


def test_zero_dim_rejected():
    with pytest.raises(ValueError):
        FlatIndex(0)


def test_out_of_range_id_rejected(small):
    with pytest.raises(IndexError):
        small.vector_at(99)
    with pytest.raises(IndexError):
        small.search_filtered(np.zeros(2, dtype=np.float32), 1, [99])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_inner_product_prefers_larger_dots():
    idx = FlatIndex(2, Metric.InnerProduct)
    idx.add_batch(np.array([[1, 0], [5, 0], [2, 0]], dtype=np.float32))
    got = idx.search(np.array([1, 0], dtype=np.float32), k=3)
    assert [n.id for n in got] == [1, 2, 0]
    assert got[0].score == pytest.approx(5.0)


def test_cosine_is_scale_invariant():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 16)).astype(np.float32)
    idx = FlatIndex(16, Metric.Cosine)
    idx.add_batch(data)

    q = rng.normal(size=16).astype(np.float32)
    a = [n.id for n in idx.search(q, k=10)]
    b = [n.id for n in idx.search((q * 37.5).astype(np.float32), k=10)]
    assert a == b


def test_cosine_matches_numpy_oracle():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(300, 12)).astype(np.float32)
    queries = rng.normal(size=(10, 12)).astype(np.float32)

    idx = FlatIndex(12, Metric.Cosine)
    idx.add_batch(data)
    ids, _ = idx.search_batch(queries, k=10)

    expected = ds.compute_ground_truth(data, queries, k=10, metric="angular")
    assert ds.recall_at_k(ids, expected, k=10) == 1.0


# --------------------------------------------------------------------------
# Agreement with the numpy oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim,n,k", [(2, 50, 1), (8, 500, 10), (64, 300, 50)])
def test_matches_numpy_oracle(dim, n, k):
    """Engine and oracle must return the same neighbours.

    Asserted as sets rather than position-for-position, because the two
    compute L2 by different routes on purpose: the engine sums (q-b)^2
    directly, while the oracle uses the expanded |b|^2 - 2q.b form so numpy
    can run it as a matrix product. On continuous data those round
    differently in the last bit, and two candidates whose true distances
    differ by less than float32 can resolve will swap adjacent slots.

    Demanding identical ordering there would not be a stronger test, it would
    be a flaky one -- so ordering is checked where it is meaningful (the
    scores below, and SIFT's exactly-representable integers further down),
    and membership is checked here.
    """
    rng = np.random.default_rng(dim * 31 + n)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    queries = rng.normal(size=(20, dim)).astype(np.float32)

    idx = FlatIndex(dim)
    idx.add_batch(data)
    ids, scores = idx.search_batch(queries, k=k)

    expected = ds.compute_ground_truth(data, queries, k=k)
    assert ds.recall_at_k(ids, expected, k=k) == 1.0

    # Scores are ascending and are the true distances, whichever id holds a
    # given slot -- so a swapped near-tie is invisible here, as it should be.
    for row, query in zip(range(len(queries)), queries):
        assert np.all(np.diff(scores[row]) >= -1e-6), "results must be sorted"
        for rank, vid in enumerate(ids[row]):
            true_d = float(np.linalg.norm(data[vid] - query))
            assert scores[row][rank] == pytest.approx(true_d, rel=1e-5)


def test_disagreements_with_the_oracle_are_only_ties():
    """Pins the claim made above: where the orderings differ, the distances
    are equal to within what float32 can represent. A real ranking bug would
    show a gap far larger than this."""
    rng = np.random.default_rng(64 * 31 + 300)
    data = rng.normal(size=(300, 64)).astype(np.float32)
    queries = rng.normal(size=(20, 64)).astype(np.float32)

    idx = FlatIndex(64)
    idx.add_batch(data)
    ids, _ = idx.search_batch(queries, k=50)
    expected = ds.compute_ground_truth(data, queries, k=50)

    for row, col in zip(*np.nonzero(ids != expected)):
        query = queries[row].astype(np.float64)
        got = np.linalg.norm(data[ids[row, col]].astype(np.float64) - query)
        want = np.linalg.norm(data[expected[row, col]].astype(np.float64) - query)
        # One float32 ulp at these magnitudes is ~1e-6; require the gap to be
        # comfortably inside that.
        assert abs(got - want) < 1e-5, (
            f"row {row} rank {col}: distances {got} vs {want} differ by more "
            f"than float32 rounding -- this is a real ordering bug, not a tie"
        )


def test_l2_scores_are_true_distances():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(100, 5)).astype(np.float32)
    query = rng.normal(size=5).astype(np.float32)

    idx = FlatIndex(5)
    idx.add_batch(data)
    got = idx.search(query, k=5)

    for n in got:
        expected = float(np.linalg.norm(data[n.id] - query))
        assert n.score == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------
# Filtered search -- the pre-filter plan
# --------------------------------------------------------------------------


def test_filtered_search_only_returns_allowed(small):
    got = small.search_filtered(np.array([0, 0], dtype=np.float32), k=4, allowed=[2, 3])
    assert [n.id for n in got] == [2, 3]


def test_empty_filter_returns_nothing(small):
    assert small.search_filtered(np.zeros(2, dtype=np.float32), 3, []) == []


def test_pre_filter_equals_post_filter():
    """The identity the planner depends on: both routes, same answer."""
    rng = np.random.default_rng(5)
    data = rng.normal(size=(800, 10)).astype(np.float32)
    queries = rng.normal(size=(15, 10)).astype(np.float32)

    idx = FlatIndex(10)
    idx.add_batch(data)

    allowed = [i for i in range(800) if i % 7 == 0]
    allowed_set = set(allowed)

    for q in queries:
        pre = [n.id for n in idx.search_filtered(q, 5, allowed)]

        full, _ = idx.search_batch(q[None, :], k=800)
        post = [i for i in full[0].tolist() if i in allowed_set][:5]

        assert pre == post


def test_filtered_search_matches_the_oracle_over_the_subset():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(600, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)

    idx = FlatIndex(8)
    idx.add_batch(data)

    allowed = np.arange(0, 600, 3, dtype=np.int64)
    got = [n.id for n in idx.search_filtered(query, 10, allowed.tolist())]

    sub = ds.compute_ground_truth(data[allowed], query[None, :], k=10)
    assert got == allowed[sub[0]].tolist()


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sift():
    try:
        return ds.load_sift("siftsmall")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def index(sift):
    idx = FlatIndex(sift.dim)
    idx.reserve(sift.n)
    idx.add_batch(sift.base)
    return idx


@pytest.mark.sift
class TestSift:
    def test_index_holds_the_whole_corpus(self, index, sift):
        assert len(index) == sift.n == 10_000
        assert index.dim == 128

    def test_matches_the_published_ground_truth(self, index, sift):
        """The result this module exists to produce.

        TEXMEX computed these neighbour lists in 2010. If the engine
        reproduces them exactly on all 100 queries, its answers are correct
        by an authority entirely outside this project -- which is what makes
        it usable as the yardstick for HNSW later.

        Compared as sets per query: SIFT descriptors are quantised, so
        genuine distance ties exist and their ordering is arbitrary.
        """
        ids, _ = index.search_batch(sift.queries, k=100)
        assert ds.recall_at_k(ids, sift.ground_truth, k=100) == 1.0

    def test_top_1_agrees_exactly(self, index, sift):
        ids, _ = index.search_batch(sift.queries, k=1)
        assert np.array_equal(ids[:, 0], sift.ground_truth[:, 0])

    def test_matches_the_numpy_oracle_exactly(self, index, sift):
        """C++ and numpy must agree element-for-element, not merely as sets.

        They compute L2 differently on purpose -- the engine sums (q-b)^2
        directly, the oracle uses the expanded |b|^2 - 2q.b form so it can
        run as a matrix product. Agreeing anyway means neither is relying on
        a quirk of how the other rounds.
        """
        ids, _ = index.search_batch(sift.queries, k=10)
        expected = ds.compute_ground_truth(sift.base, sift.queries, k=10)
        assert np.array_equal(ids, expected)

    def test_self_retrieval(self, index, sift):
        probes = sift.base[:200]
        ids, scores = index.search_batch(probes, k=1)
        assert ids[:, 0].tolist() == list(range(200))
        assert np.allclose(scores[:, 0], 0.0, atol=1e-3)

    def test_filtered_search_on_real_vectors(self, index, sift):
        """A selective predicate over real data, both plans agreeing."""
        h = ds.make_hybrid(sift, seed=0)
        price = h.attributes["price"]
        threshold = ds.threshold_for_selectivity(price, 0.05)
        allowed = np.flatnonzero(price < threshold)
        assert len(allowed) == 500

        query = sift.queries[0]
        pre = [n.id for n in index.search_filtered(query, 10, allowed.tolist())]

        sub = ds.compute_ground_truth(sift.base[allowed], query[None, :], k=10)
        assert pre == allowed[sub[0]].tolist()
