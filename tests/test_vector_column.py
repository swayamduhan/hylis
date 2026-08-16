"""Vector columns through the Python bridge: embeddings that belong to rows.

The C++ suite covers the invariants. What is worth testing again *here* is the
bridge itself, because every one of these calls crosses it with something the
scalar surface never had to carry: numpy arrays in, record keys out, and a
plan object that lives in a different extension module than the table does.

The one structural claim under test is the id translation. A vector index
numbers rows 0..n-1 into a float buffer; the table numbers them by record key.
Every assertion phrased in keys is really an assertion about that join, because
getting it wrong returns entirely plausible neighbours belonging to the wrong
rows -- a failure no amount of eyeballing a distance would catch.
"""

import numpy as np
import pytest

from hylis import (
    ColumnDef,
    IndexKind,
    LogicalType,
    PlanKind,
    PredOp,
    Record,
    RecordStore,
    Schema,
    Table,
    VectorPlan,
    VectorStructure,
)

DIM = 8


def shop_schema():
    return Schema(
        [
            ColumnDef("price", LogicalType.Int64),
            ColumnDef("band", LogicalType.Int64),
            ColumnDef("category", LogicalType.String),
            ColumnDef("image", LogicalType.Vector, DIM),
        ]
    )


def clustered(n, seed=7):
    """Clustered rather than uniform.

    Uniform vectors in 8 dimensions have nearly equal pairwise distances, so a
    wrong answer looks exactly like a right one and every ranking assertion
    becomes a coin flip.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-1.0, 1.0, size=(8, DIM)).astype(np.float32)
    pick = rng.integers(0, 8, size=n)
    jitter = rng.normal(0.0, 0.05, size=(n, DIM)).astype(np.float32)
    return np.ascontiguousarray(centres[pick] + jitter, dtype=np.float32)


def shop_rows(n, start=0):
    rows = []
    for i in range(start, start + n):
        price = (i * 37) % 500
        rows.append(
            Record(
                i,
                {
                    "price": str(price),
                    "band": str(price // 100),
                    "category": ["bags", "hats", "shoes"][i % 3],
                },
            )
        )
    return rows


def loaded(tmp_path, n=300, structure=VectorStructure.Graph, name="s"):
    store = RecordStore(str(tmp_path / name))
    table = Table(store, shop_schema())
    table.put_batch(shop_rows(n))
    table.create_vector_index("image", VectorPlan(structure=structure))
    data = clustered(n)
    table.put_vectors("image", list(range(n)), data)
    # The store is returned too: it is borrowed, and the binding's keep_alive
    # is the only thing stopping Python collecting it under a live Table.
    return store, table, data


def keys_of(matches):
    return [m.key for m in matches]


# --------------------------------------------------------------------------
# The key <-> row join
# --------------------------------------------------------------------------


def test_answers_in_record_keys_not_row_ids(tmp_path):
    # Keys deliberately not 0..n-1: with contiguous keys the translation is the
    # identity, and a missing translation would pass every other test here.
    store = RecordStore(str(tmp_path / "s"))
    table = Table(store, shop_schema())
    keys = [1000 + i * 7 for i in range(40)]
    table.put_batch([Record(k, {"price": "1", "band": "0"}) for k in keys])
    table.create_vector_index("image", VectorPlan(structure=VectorStructure.Exact))

    data = clustered(40)
    table.put_vectors("image", keys, data)

    hits = table.knn("image", data[5], k=3)
    assert hits[0].key == keys[5]
    assert hits[0].row == 5
    for hit in hits:
        assert table.get(hit.key) is not None, f"knn returned {hit.key}, not a record"
    assert table.vector_info("image").rows_are_keys is False


def test_an_embedding_needs_a_record(tmp_path):
    _store, table, data = loaded(tmp_path, 20, VectorStructure.Exact)
    with pytest.raises(ValueError, match="no record with key 9999"):
        table.put_vector(9999, "image", data[0])


def test_a_vector_in_the_record_payload_is_refused_naming_put_vector(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    table = Table(store, shop_schema())
    with pytest.raises(ValueError, match="put_vector"):
        table.put(Record(1, {"image": "0.1 0.2"}))


def test_a_query_of_the_wrong_shape_is_refused(tmp_path):
    _store, table, _data = loaded(tmp_path, 30, VectorStructure.Exact)
    with pytest.raises(ValueError):
        table.knn("image", np.zeros((2, DIM), dtype=np.float32), k=3)
    with pytest.raises(ValueError):
        table.put_vectors("image", [0, 1], np.zeros((3, DIM), dtype=np.float32))


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_the_graph_agrees_with_the_exact_scan(tmp_path):
    _store, table, data = loaded(tmp_path, 500)
    assert table.vector_info("image").has_graph

    agreed = 0
    for q in range(0, 500, 25):
        approx = table.knn("image", data[q], k=5, ef=64)
        truth = table.knn("image", data[q], k=5, exact=True)
        assert keys_of(approx)[0] == keys_of(truth)[0]
        agreed += keys_of(approx) == keys_of(truth)
    # Not 20/20: HNSW is approximate by construction, so demanding exactness
    # would be asserting the wrong property. This asserts it is not *broken*.
    assert agreed >= 16


def test_more_like_this_excludes_its_own_seed(tmp_path):
    _store, table, data = loaded(tmp_path, 200, VectorStructure.Exact)
    hits = table.knn_by_key("image", 42, k=5)
    assert len(hits) == 5
    assert 42 not in keys_of(hits)
    # And it is the plain answer with the seed removed -- a seed dropped by
    # post-filtering would have come back four rows short.
    plain = [k for k in keys_of(table.knn("image", data[42], k=6)) if k != 42]
    assert keys_of(hits) == plain


def test_more_like_this_says_so_when_the_row_has_no_embedding(tmp_path):
    _store, table, _data = loaded(tmp_path, 20, VectorStructure.Exact)
    table.put(Record(9001, {"price": "1", "band": "0"}))
    with pytest.raises(ValueError, match="no embedding"):
        table.knn_by_key("image", 9001, k=3)


def test_get_vector_round_trips_and_is_none_when_absent(tmp_path):
    _store, table, data = loaded(tmp_path, 50, VectorStructure.Exact)
    np.testing.assert_allclose(table.get_vector(7, "image"), data[7])
    table.put(Record(9001, {"price": "1", "band": "0"}))
    assert table.get_vector(9001, "image") is None


# --------------------------------------------------------------------------
# Deletion and compaction
# --------------------------------------------------------------------------


def test_a_deleted_row_leaves_every_answer_at_once(tmp_path):
    _store, table, data = loaded(tmp_path, 300)
    assert keys_of(table.knn("image", data[100], k=1)) == [100]

    table.erase(100)
    assert table.vector_info("image").orphans == 1
    for exact in (True, False):
        hits = table.knn("image", data[100], k=5, ef=64, exact=exact)
        assert len(hits) == 5, "the mask cost rows rather than excluding one"
        assert 100 not in keys_of(hits)
    table.validate()


def test_compaction_reclaims_orphans_without_changing_an_answer(tmp_path):
    _store, table, data = loaded(tmp_path, 400)
    for key in (5, 17, 200, 201, 202):
        table.erase(key)
    assert table.vector_info("image").orphans == 5

    before = keys_of(table.knn("image", data[33], k=8, exact=True))
    assert table.compact_vectors("image") == 5
    assert table.vector_info("image").orphans == 0
    after = keys_of(table.knn("image", data[33], k=8, exact=True))
    assert before == after, "compaction renumbered rows and changed the answer"
    table.validate()


def test_replacing_an_embedding_orphans_the_old_one(tmp_path):
    _store, table, data = loaded(tmp_path, 200, VectorStructure.Exact)
    table.put_vector(7, "image", data[150])
    assert table.vector_info("image").orphans == 1
    assert 7 in keys_of(table.knn("image", data[150], k=3))
    # The orphan may not come back as a second copy of row 7.
    hits = keys_of(table.knn("image", data[7], k=5))
    assert len(set(hits)) == len(hits)
    table.validate()


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_reopen_reproduces_the_index_exactly(tmp_path):
    store, table, data = loaded(tmp_path, 400)
    before = [table.knn("image", data[q], k=10, ef=64) for q in range(0, 400, 40)]
    table.checkpoint()
    del table, store

    reopened = Table.open(RecordStore(str(tmp_path / "s")))
    assert reopened.vector_info("image").rows == 400
    assert reopened.vector_info("image").has_graph

    for i, q in enumerate(range(0, 400, 40)):
        after = reopened.knn("image", data[q], k=10, ef=64)
        # Neighbour for neighbour, not recall-within-a-threshold. The graph is
        # not stored, it is replayed, and the seed is fixed -- so "the same" is
        # the correct claim, and anything weaker would hide a reload that
        # quietly built a different graph.
        assert keys_of(after) == keys_of(before[i]), f"query {q}"
    reopened.validate()


def test_the_sidecar_is_readable_by_the_dataset_loader(tmp_path):
    """It is written in TEXMEX .fvecs layout on purpose.

    A private binary format would have been simpler to write and impossible to
    inspect; this one is read by a function the project already had, which is
    the whole argument for choosing it.
    """
    from hylis import datasets as ds

    _store, table, data = loaded(tmp_path, 60, VectorStructure.Exact)
    table.save_vectors()

    stored = ds.read_fvecs(tmp_path / "s" / "image.fvecs")
    assert stored.shape == (60, DIM)
    np.testing.assert_allclose(stored, data, atol=1e-6)


def test_the_sidecar_holds_live_rows_only(tmp_path):
    store, table, _data = loaded(tmp_path, 100, VectorStructure.Exact)
    for key in (1, 2, 3):
        table.erase(key)
    assert table.vector_info("image").orphans == 3
    table.checkpoint()
    del table, store

    reopened = Table.open(RecordStore(str(tmp_path / "s")))
    assert reopened.vector_info("image").rows == 97
    assert reopened.vector_info("image").orphans == 0
    for key in (1, 2, 3):
        assert not reopened.has_vector(key, "image")


def test_embeddings_since_the_last_save_do_not_survive(tmp_path):
    store, table, data = loaded(tmp_path, 50, VectorStructure.Exact)
    table.checkpoint()
    table.put(Record(50, {"price": "1", "band": "0"}))
    table.put_vector(50, "image", data[0])
    del table, store  # no checkpoint, no save_vectors: the process ends here

    reopened = Table.open(RecordStore(str(tmp_path / "s")))
    # The record survived, because it went through the write-ahead log.
    assert reopened.get(50) is not None
    # The embedding did not, because vectors never do. Asserted rather than
    # left implicit: it is the stated cost of keeping a 128-float payload out
    # of a JSON WAL, and a test is the only thing that keeps it stated.
    assert reopened.vector_info("image").rows == 50
    assert not reopened.has_vector(50, "image")


def test_retuning_keeps_the_embeddings_and_refuses_a_metric_change(tmp_path):
    from hylis import Metric

    _store, table, data = loaded(tmp_path, 150, VectorStructure.Exact)
    before = keys_of(table.knn("image", data[20], k=5))

    table.create_vector_index("image", VectorPlan(structure=VectorStructure.Graph, M=8))
    assert table.vector_info("image").has_graph
    assert table.vector_info("image").rows == 150
    assert keys_of(table.knn("image", data[20], k=5, exact=True)) == before

    with pytest.raises(ValueError, match="metric"):
        table.create_vector_index("image", VectorPlan(metric=Metric.Cosine))


# --------------------------------------------------------------------------
# Hybrid queries
# --------------------------------------------------------------------------


def test_every_plan_returns_the_same_rows(tmp_path):
    _store, table, data = loaded(tmp_path, 600)
    table.create_index_as("band", IndexKind.Bitmap)
    predicates = [("band", PredOp.Lt, 3)]
    # ef high enough that the graph is not losing recall to beam width: the
    # question is whether the plans agree, not how well HNSW is tuned.
    planned, trace = table.hybrid(predicates, "image", data[77], k=10, ef=400)
    assert len(planned) == 10

    for kind in (PlanKind.PreFilter, PlanKind.FilteredGraph,
                 PlanKind.BitmapFilteredGraph):
        assert table.plan_available(kind, predicates, "image"), kind
        forced, _ = table.hybrid_with(kind, predicates, "image", data[77], k=10, ef=400)
        assert keys_of(forced) == keys_of(planned), f"{kind} disagreed"
    assert trace.plan.kind == PlanKind.BitmapFilteredGraph


def test_every_returned_row_satisfies_the_predicate(tmp_path):
    _store, table, data = loaded(tmp_path, 500)
    table.create_index("price")
    for q in range(0, 500, 60):
        hits, _ = table.hybrid([("price", PredOp.Lt, 200)], "image", data[q], k=10, ef=64)
        for hit in hits:
            assert int(table.get(hit.key).columns["price"]) < 200


def test_the_bitmap_plan_costs_no_row_ids(tmp_path):
    _store, table, data = loaded(tmp_path, 600)
    table.create_index_as("band", IndexKind.Bitmap)
    predicates = [("band", PredOp.Ge, 1)]

    hits, trace = table.hybrid(predicates, "image", data[5], k=10, ef=64)
    assert trace.mask_used
    assert trace.plan.selectivity_was_free
    assert trace.plan.kind == PlanKind.BitmapFilteredGraph
    assert len(hits) == 10

    # The plan stops being reachable the moment the two row spaces part, which
    # is the precondition being enforced rather than assumed.
    table.erase(0)
    assert not table.plan_available(PlanKind.BitmapFilteredGraph, predicates, "image")
    with pytest.raises(ValueError, match="row ids"):
        table.hybrid_with(PlanKind.BitmapFilteredGraph, predicates, "image",
                          data[5], k=5, ef=64)
    # The query still runs; it just takes the other route.
    _hits, after = table.hybrid(predicates, "image", data[5], k=5, ef=64)
    assert not after.mask_used


def test_rows_without_an_embedding_are_counted_not_hidden(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    table = Table(store, shop_schema())
    table.put_batch(shop_rows(200))
    table.create_index("price")
    table.create_vector_index("image", VectorPlan(structure=VectorStructure.Exact))

    data = clustered(100)
    table.put_vectors("image", list(range(0, 200, 2)), data)

    hits, trace = table.hybrid([("price", PredOp.Lt, 500)], "image", data[3], k=5)
    assert trace.structured.matched == 200
    assert trace.without_vector == 100
    assert trace.plan.matched_rows == 100
    assert all(hit.key % 2 == 0 for hit in hits)


def test_no_predicate_degenerates_to_a_similarity_search(tmp_path):
    _store, table, data = loaded(tmp_path, 200, VectorStructure.Exact)
    hits, trace = table.hybrid([], "image", data[12], k=5)
    assert trace.plan.kind == PlanKind.NoPredicate
    assert keys_of(hits) == keys_of(table.knn("image", data[12], k=5))


def test_a_predicate_no_index_can_serve_still_reaches_the_vector_search(tmp_path):
    _store, table, data = loaded(tmp_path, 300, VectorStructure.Exact)
    # Contains has no ordering to exploit, so Table answers it by scanning --
    # and those row ids are still the vector search's filter.
    hits, trace = table.hybrid([("category", PredOp.Contains, "ag")], "image",
                               data[4], k=5)
    assert not trace.structured.used_index
    assert trace.structured.scanned > 0
    assert hits
    assert all(table.get(h.key).columns["category"] == "bags" for h in hits)


def test_a_column_with_no_vector_index_says_which_mistake_it_was(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    table = Table(store, shop_schema())
    table.put_batch(shop_rows(20))
    query = np.zeros(DIM, dtype=np.float32)

    with pytest.raises(ValueError, match="create_vector_index"):
        table.knn("image", query, k=3)
    with pytest.raises(ValueError, match="not a vector"):
        table.knn("price", query, k=3)
