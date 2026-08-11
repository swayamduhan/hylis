"""Python-side tests for the learned index and per-column selection.

The C++ suite carries the invariant coverage. What matters here is the
differential sweep against the B+ tree over the real generated
distributions in hylis.datasets: the two structures share the CompareOp
interface precisely so a query planner can substitute one for the other, and
that is only safe if they are indistinguishable through it.
"""

import numpy as np
import pytest

from hylis import (
    BPlusTree,
    ColumnIndex,
    CompareOp,
    IndexCatalog,
    IndexKind,
    IndexPlan,
    RMIndex,
    choose_index,
    measure_plan,
)
from hylis import datasets as ds

DISTRIBUTIONS = ["sequential_gaps", "uniform", "lognormal", "clustered"]

OPS = [CompareOp.Lt, CompareOp.Le, CompareOp.Gt, CompareOp.Ge, CompareOp.Eq]


def keys_and_values(dist, n=5000, seed=0):
    data = ds.synthetic_keys(dist, n=n, seed=seed)
    return data.keys.tolist(), list(range(len(data.keys)))


# --------------------------------------------------------------------------
# Bridge
# --------------------------------------------------------------------------


@pytest.fixture
def index():
    idx = RMIndex(models=64)
    idx.build(list(range(0, 1000, 5)), list(range(200)))
    return idx


def test_build_and_find(index):
    assert len(index) == 200
    assert index.find(0) == 0
    assert index.find(500) == 100
    assert index.find(501) is None
    assert 500 in index
    assert 501 not in index


def test_repr_and_properties(index):
    assert index.models == 64
    assert index.search_threshold == 64
    assert "RMIndex(" in repr(index)


def test_unsorted_keys_rejected():
    idx = RMIndex()
    with pytest.raises(ValueError, match="ascending"):
        idx.build([3, 1, 2], [1, 2, 3])


def test_duplicate_keys_rejected():
    idx = RMIndex()
    with pytest.raises(ValueError, match="ascending"):
        idx.build([1, 2, 2], [1, 2, 3])


def test_mismatched_lengths_rejected():
    idx = RMIndex()
    with pytest.raises(ValueError):
        idx.build([1, 2, 3], [1, 2])


def test_zero_models_rejected():
    with pytest.raises(ValueError):
        RMIndex(models=0)


def test_empty_index():
    idx = RMIndex()
    idx.build([], [])
    assert len(idx) == 0
    assert idx.find(1) is None
    assert idx.range_query(CompareOp.Ge, 0) == []
    idx.validate()


def test_range_and_range_query(index):
    assert index.range(0, 20) == [0, 1, 2, 3, 4]
    assert index.range_query(CompareOp.Lt, 20) == [0, 1, 2, 3]
    assert index.range_query(CompareOp.Le, 20) == [0, 1, 2, 3, 4]
    assert index.range_query(CompareOp.Eq, 20) == [4]
    assert index.range_query(CompareOp.Eq, 21) == []


def test_stats_are_exposed(index):
    stats = index.stats()
    assert stats.size == 200
    assert stats.models == 64
    assert stats.max_error < 200
    assert stats.mean_error <= stats.max_error
    assert stats.total_bytes > stats.model_bytes


def test_clear(index):
    index.clear()
    assert len(index) == 0
    index.validate()


# --------------------------------------------------------------------------
# The differential sweep against the B+ tree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_agrees_with_btree_on_every_distribution(dist):
    keys, values = keys_and_values(dist)

    rmi = RMIndex(models=256)
    rmi.build(keys, values)
    rmi.validate()

    tree = BPlusTree(order=32)
    for k, v in zip(keys, values):
        tree.insert(k, v)

    assert rmi.keys() == tree.keys()
    assert rmi.items() == tree.items()

    rng = np.random.default_rng(0)
    for _ in range(200):
        present = int(keys[rng.integers(len(keys))])
        for probe in (present, present + 1):
            assert rmi.find(probe) == tree.find(probe), f"find({probe})"
            for op in OPS:
                assert rmi.range_query(op, probe) == tree.range_query(op, probe), (
                    f"{dist} {op} {probe}"
                )


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_validate_holds_on_every_distribution(dist):
    """The exactness proof: every key lies inside its predicted window."""
    keys, values = keys_and_values(dist, n=20000)
    idx = RMIndex(models=1024)
    idx.build(keys, values)
    idx.validate()

    for i in range(0, len(keys), 97):
        assert idx.find(keys[i]) == values[i]


@pytest.mark.parametrize("models", [1, 16, 1024])
def test_correct_at_any_model_count(models):
    """Correctness must not depend on model quality — only speed does."""
    keys, values = keys_and_values("clustered", n=5000)
    idx = RMIndex(models=models)
    idx.build(keys, values)
    idx.validate()
    assert all(idx.find(k) == v for k, v in zip(keys, values))


# --------------------------------------------------------------------------
# Curvature
# --------------------------------------------------------------------------


def test_more_models_reduce_error_on_a_curved_distribution():
    keys, values = keys_and_values("lognormal", n=50000)

    def mean_error(models):
        idx = RMIndex(models=models)
        idx.build(keys, values)
        return idx.stats().mean_error

    assert mean_error(64) < mean_error(1) / 100
    assert mean_error(4096) < mean_error(64)


def test_more_models_do_not_rescue_a_stepped_distribution():
    """clustered has cliffs, not curves: past a point nothing changes."""
    keys, values = keys_and_values("clustered", n=50000)

    def stats(models):
        idx = RMIndex(models=models)
        idx.build(keys, values)
        return idx.stats()

    a, b = stats(4096), stats(65536)
    assert b.max_error == a.max_error
    assert b.mean_error == pytest.approx(a.mean_error)
    assert b.empty_models > b.models * 0.99


def test_a_useless_model_stays_logarithmic():
    keys, values = keys_and_values("clustered", n=20000)
    idx = RMIndex(models=1)
    idx.build(keys, values)
    idx.validate()

    budget = np.log2(len(keys)) + 8
    for i in range(0, len(keys), 211):
        assert idx.find(keys[i]) == values[i]
        assert idx.probes(keys[i]) <= budget


# --------------------------------------------------------------------------
# Per-column selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_column_index_matches_a_dict_whatever_it_chose(dist):
    keys, values = keys_and_values(dist, n=3000)
    column = ColumnIndex.build(keys, values)
    column.validate()

    assert len(column) == len(keys)
    for k, v in zip(keys, values):
        assert column.find(k) == v
    assert column.find(keys[-1] + 1) is None


def test_forcing_either_structure_gives_identical_answers():
    keys, values = keys_and_values("lognormal", n=3000)

    tree_plan = IndexPlan()
    tree_plan.kind = IndexKind.BPlusTree
    rmi_plan = IndexPlan()
    rmi_plan.kind = IndexKind.RMI
    rmi_plan.rmi_models = 128

    tree = ColumnIndex.build_with(keys, values, tree_plan)
    rmi = ColumnIndex.build_with(keys, values, rmi_plan)
    assert tree.kind == IndexKind.BPlusTree
    assert rmi.kind == IndexKind.RMI

    rng = np.random.default_rng(1)
    for _ in range(100):
        probe = int(keys[rng.integers(len(keys))]) + int(rng.integers(2))
        assert tree.find(probe) == rmi.find(probe)
        for op in OPS:
            assert tree.range_query(op, probe) == rmi.range_query(op, probe)


def test_choose_index_records_its_evidence():
    keys, values = keys_and_values("sequential_gaps", n=20000)
    plan = choose_index(keys, values)

    assert plan.ns_per_lookup > 0, "the decision must be measured, not assumed"
    assert plan.index_bytes > 0
    assert plan.n_keys == len(keys)
    assert plan.key_min == keys[0]
    assert plan.key_max == keys[-1]
    assert plan.matches(keys)


def test_measure_plan_times_a_specific_structure():
    keys, values = keys_and_values("uniform", n=10000)

    tree_plan = IndexPlan()
    tree_plan.kind = IndexKind.BPlusTree
    measured = measure_plan(keys, values, tree_plan)

    assert measured.kind == IndexKind.BPlusTree
    assert measured.ns_per_lookup > 0
    assert measured.index_bytes > 0


def test_plan_fingerprint_detects_changed_data():
    keys, values = keys_and_values("uniform", n=1000)
    plan = choose_index(keys, values)

    assert plan.matches(keys)
    assert not plan.matches(keys[:-1])
    assert not plan.matches(keys[:-1] + [keys[-1] + 1])
    assert not plan.matches([])


def test_a_stale_plan_still_builds_a_correct_index():
    other_keys, other_values = keys_and_values("clustered", n=1000, seed=9)
    wrong = choose_index(other_keys, other_values)

    keys, values = keys_and_values("sequential_gaps", n=4000)
    assert not wrong.matches(keys)

    column = ColumnIndex.build_with(keys, values, wrong)
    column.validate()
    assert all(column.find(k) == v for k, v in zip(keys, values))


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def test_catalog_round_trips_through_json():
    catalog = IndexCatalog()
    plan = IndexPlan()
    plan.kind = IndexKind.RMI
    plan.rmi_models = 16384
    plan.n_keys = 500
    plan.key_min = -7
    plan.key_max = 12345
    catalog.set("price", plan)

    back = IndexCatalog.parse(catalog.serialize())
    assert len(back) == 1
    assert back.get("price").rmi_models == 16384
    assert back.get("price").key_min == -7
    assert back.get("missing") is None


def test_catalog_saves_and_loads(tmp_path):
    path = str(tmp_path / "catalog.json")
    keys, values = keys_and_values("uniform", n=2000)

    catalog = IndexCatalog()
    catalog.build_column("price", keys, values)
    catalog.save(path)

    loaded = IndexCatalog.load(path)
    assert loaded.columns() == ["price"]
    assert loaded.freshness("price", keys) == IndexCatalog.Freshness.Fresh


def test_catalog_load_of_a_missing_file_is_empty(tmp_path):
    assert len(IndexCatalog.load(str(tmp_path / "nope.json"))) == 0


def test_catalog_load_of_a_corrupt_file_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"plans":[{"column":"x","kind":', encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        IndexCatalog.load(str(path))


def test_catalog_reports_freshness():
    keys, values = keys_and_values("uniform", n=1000)
    catalog = IndexCatalog()

    assert catalog.freshness("price", keys) == IndexCatalog.Freshness.Missing
    catalog.build_column("price", keys, values)
    assert catalog.freshness("price", keys) == IndexCatalog.Freshness.Fresh
    assert catalog.freshness("price", keys[:-1]) == IndexCatalog.Freshness.Stale


def test_catalog_retunes_when_the_column_changes():
    catalog = IndexCatalog()
    keys, values = keys_and_values("sequential_gaps", n=3000)
    catalog.build_column("c", keys, values)

    other_keys, other_values = keys_and_values("clustered", n=3000)
    column = catalog.build_column("c", other_keys, other_values)
    column.validate()

    assert catalog.freshness("c", other_keys) == IndexCatalog.Freshness.Fresh
    assert catalog.get("c").key_max == other_keys[-1]


def test_heterogeneous_columns_are_tracked_independently():
    """What the catalog is actually for: several columns, each with its own
    decision, recorded together."""
    catalog = IndexCatalog()
    columns = {}
    for dist in DISTRIBUTIONS:
        keys, values = keys_and_values(dist, n=5000)
        columns[dist] = keys
        catalog.build_column(dist, keys, values)

    assert sorted(catalog.columns()) == sorted(DISTRIBUTIONS)
    for dist, keys in columns.items():
        assert catalog.freshness(dist, keys) == IndexCatalog.Freshness.Fresh
        assert catalog.get(dist).n_keys == len(keys)

    # And the evidence genuinely differs between an easy and a hard column.
    assert catalog.get("sequential_gaps").max_error < catalog.get("clustered").max_error
