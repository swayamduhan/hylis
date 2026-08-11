"""Tests for the dataset helpers.

These matter more than usual: every later benchmark number is only as
trustworthy as the data and the ground-truth oracle underneath it. A silent
bug here would not fail loudly, it would just produce plausible-looking
recall figures that mean nothing.
"""

import numpy as np
import pytest

from hylis import BPlusTree
from hylis import datasets as ds


# --------------------------------------------------------------------------
# Synthetic keys
# --------------------------------------------------------------------------


ALL_DISTRIBUTIONS = ["uniform", "lognormal", "sequential_gaps", "clustered"]


@pytest.mark.parametrize("dist", ALL_DISTRIBUTIONS)
def test_keys_are_sorted_unique_and_the_right_length(dist):
    d = ds.synthetic_keys(dist, n=5000, seed=1)
    assert len(d) == 5000
    assert d.keys.dtype == np.int64
    assert np.all(np.diff(d.keys) > 0), "keys must be strictly ascending"
    assert len(np.unique(d.keys)) == 5000


@pytest.mark.parametrize("dist", ALL_DISTRIBUTIONS)
def test_key_generation_is_deterministic(dist):
    a = ds.synthetic_keys(dist, n=1000, seed=7)
    b = ds.synthetic_keys(dist, n=1000, seed=7)
    c = ds.synthetic_keys(dist, n=1000, seed=8)
    assert np.array_equal(a.keys, b.keys)
    assert not np.array_equal(a.keys, c.keys)


def test_unknown_distribution_rejected():
    with pytest.raises(ValueError, match="unknown distribution"):
        ds.synthetic_keys("gaussian", n=10)


def test_zero_or_negative_n_rejected():
    with pytest.raises(ValueError):
        ds.synthetic_keys("uniform", n=0)


def test_linearity_ranks_distributions_as_expected():
    """The whole point of having four shapes: they must actually differ.

    A learned index interpolates the key CDF, so if these all scored the same
    there would be no experiment to run. sequential_gaps is near-linear;
    clustered and lognormal are the ones an RMI should struggle with.
    """
    scores = {
        dist: ds.synthetic_keys(dist, n=20_000, seed=3).linearity()
        for dist in ALL_DISTRIBUTIONS
    }
    assert scores["sequential_gaps"] > 0.99
    assert scores["uniform"] > 0.99
    assert scores["lognormal"] < scores["uniform"]
    assert scores["clustered"] < scores["sequential_gaps"]


def test_keys_load_into_the_btree():
    """The generated data is directly consumable by the engine."""
    d = ds.synthetic_keys("clustered", n=2000, seed=2)
    tree = BPlusTree(order=32)
    for key, value in d.items():
        assert tree.insert(key, value) is True

    tree.validate()
    assert len(tree) == 2000
    assert tree.keys() == d.keys.tolist()
    assert tree.find(int(d.keys[100])) == 100


# --------------------------------------------------------------------------
# fvecs / ivecs round-trip
# --------------------------------------------------------------------------


def test_fvecs_round_trip(tmp_path):
    original = np.random.default_rng(0).normal(size=(37, 8)).astype(np.float32)
    path = tmp_path / "t.fvecs"
    ds.write_fvecs(path, original)
    assert np.array_equal(ds.read_fvecs(path), original)


def test_fvecs_limit_reads_a_prefix(tmp_path):
    original = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    path = tmp_path / "t.fvecs"
    ds.write_fvecs(path, original)
    assert np.array_equal(ds.read_fvecs(path, limit=10), original[:10])


def test_truncated_fvecs_is_rejected(tmp_path):
    """A half-finished download must fail loudly, not yield short vectors."""
    original = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    path = tmp_path / "t.fvecs"
    ds.write_fvecs(path, original)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 7])

    with pytest.raises(ValueError, match="truncated or not fvecs"):
        ds.read_fvecs(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.fvecs"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        ds.read_fvecs(path)


def test_missing_sift_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        ds.load_sift("siftsmall", data_dir=tmp_path)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def test_ground_truth_matches_a_naive_loop():
    """Differential test of the vectorised oracle against the obvious one.

    compute_ground_truth is chunked, drops the constant ||q||^2 term and
    juggles argpartition indices -- none of that is self-evidently correct, so
    it is checked against a plain double loop the way the B+ tree is checked
    against std::map.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 16)).astype(np.float32)
    queries = rng.normal(size=(15, 16)).astype(np.float32)

    got = ds.compute_ground_truth(base, queries, k=10, chunk=32)

    for qi, query in enumerate(queries):
        distances = [float(np.sum((b - query) ** 2)) for b in base]
        expected = sorted(range(len(base)), key=lambda i: distances[i])[:10]
        assert got[qi].tolist() == expected, f"query {qi}"


def test_ground_truth_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 12)).astype(np.float32)
    queries = rng.normal(size=(10, 12)).astype(np.float32)

    whole = ds.compute_ground_truth(base, queries, k=20, chunk=10_000)
    for chunk in (7, 64, 499):
        assert np.array_equal(ds.compute_ground_truth(base, queries, 20, chunk=chunk),
                              whole), f"chunk={chunk}"


def test_nearest_neighbour_of_a_base_point_is_itself():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(300, 10)).astype(np.float32)
    gt = ds.compute_ground_truth(base, base[:20], k=1)
    assert gt[:, 0].tolist() == list(range(20))


def test_angular_metric_ignores_magnitude():
    """Cosine distance is scale-invariant, so scaling a query must not move it."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(100, 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)

    normal = ds.compute_ground_truth(base, queries, k=5, metric="angular")
    scaled = ds.compute_ground_truth(base, queries * 17.0, k=5, metric="angular")
    assert np.array_equal(normal, scaled)


def test_k_larger_than_the_dataset_is_clamped():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(6, 4)).astype(np.float32)
    gt = ds.compute_ground_truth(base, base[:2], k=100)
    assert gt.shape == (2, 6)


def test_recall_is_order_insensitive():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    assert ds.recall_at_k(truth, truth) == 1.0
    assert ds.recall_at_k(np.array([[3, 2, 1], [6, 5, 4]]), truth) == 1.0
    assert ds.recall_at_k(np.array([[1, 2, 9], [4, 9, 9]]), truth) == pytest.approx(0.5)
    assert ds.recall_at_k(np.array([[7, 8, 9], [7, 8, 9]]), truth) == 0.0


def test_random_vectors_carry_usable_ground_truth():
    v = ds.random_vectors(n=500, dim=16, n_queries=25, seed=5, n_clusters=10)
    assert v.n == 500 and v.dim == 16 and v.n_queries == 25
    assert v.ground_truth is not None
    assert ds.recall_at_k(v.ground_truth[:, :10], v.ground_truth[:, :10]) == 1.0


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="dimension mismatch"):
        ds.VectorDataset(np.zeros((4, 8), np.float32),
                         np.zeros((2, 5), np.float32), None, "bad")


# --------------------------------------------------------------------------
# Hybrid data and selectivity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.001, 0.01, 0.1, 0.5, 0.9])
def test_selectivity_threshold_is_exact_for_continuous_attributes(target):
    """The planner's cost model is a function of selectivity, so the knob
    that sets it has to be accurate, not approximately accurate."""
    values = np.random.default_rng(0).uniform(0, 1000, size=10_000)
    t = ds.threshold_for_selectivity(values, target)
    achieved = float(np.mean(values < t))
    assert achieved == pytest.approx(target, abs=1e-4)


def test_selectivity_edges():
    values = np.random.default_rng(0).uniform(0, 1000, size=1000)
    assert float(np.mean(values < ds.threshold_for_selectivity(values, 0.0))) == 0.0
    assert float(np.mean(values < ds.threshold_for_selectivity(values, 1.0))) == 1.0
    with pytest.raises(ValueError, match="selectivity"):
        ds.threshold_for_selectivity(values, 1.5)


def test_hybrid_dataset_is_row_aligned():
    h = ds.make_hybrid(ds.random_vectors(n=400, dim=8, n_queries=5, seed=6), seed=6)
    assert len(h) == 400
    assert h.keys.shape == (400,)
    for name in ("price", "timestamp", "category"):
        assert h.attributes[name].shape == (400,)


def test_hybrid_predicate_selects_the_intended_fraction():
    h = ds.make_hybrid(ds.random_vectors(n=2000, dim=8, n_queries=5, seed=7), seed=7)
    price = h.attributes["price"]
    t = ds.threshold_for_selectivity(price, 0.05)
    assert h.selectivity(price < t) == pytest.approx(0.05, abs=1e-3)


def test_filtered_search_agrees_with_filter_then_search():
    """The identity the query planner depends on.

    Whichever route the planner picks -- restrict the candidate set first, or
    search everything and filter after -- the answer must be the same. If this
    did not hold, the planner would be choosing between different results
    rather than between different costs, and comparing plans would be
    meaningless.
    """
    h = ds.make_hybrid(ds.random_vectors(n=1500, dim=12, n_queries=8, seed=8), seed=8)
    price = h.attributes["price"]
    mask = price < ds.threshold_for_selectivity(price, 0.25)
    kept = np.flatnonzero(mask)

    # Route A: filter first, then exact search over survivors only.
    filtered = ds.compute_ground_truth(h.vectors.base[kept], h.vectors.queries, k=5)
    route_a = kept[filtered]

    # Route B: exact search over everything, then keep the first 5 survivors.
    full = ds.compute_ground_truth(h.vectors.base, h.vectors.queries, k=1500)
    route_b = np.array([[i for i in row if mask[i]][:5] for row in full])

    assert np.array_equal(route_a, route_b)


def test_hybrid_generation_is_deterministic():
    a = ds.make_hybrid(seed=9)
    b = ds.make_hybrid(seed=9)
    assert np.array_equal(a.attributes["price"], b.attributes["price"])
    assert np.array_equal(a.vectors.base, b.vectors.base)
