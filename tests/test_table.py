"""Table through the Python bridge: the store/index join.

The claim under test is that an index answers exactly what a full scan of the
store would answer. Everything else about this layer is worthless without it:
a predicate served by a B+ tree, by a learned index, or by no index at all must
return the same rows, or choosing between them stops being a performance
decision and becomes a semantic one.
"""

import pytest

from hylis import (
    ColumnDef,
    IndexKind,
    KeyEncoding,
    LogicalType,
    PredOp,
    Record,
    RecordStore,
    Schema,
    Table,
    op_is_indexable,
)

CATEGORIES = ["bags", "hats", "shoes"]
TITLES = ["nike air", "nike zoom", "adidas run", "puma go", "nikon lens"]


def shop_schema():
    return Schema(
        [
            ColumnDef("price", LogicalType.Int64),
            ColumnDef("category", LogicalType.String),
            ColumnDef("title", LogicalType.String),
            ColumnDef("created_at", LogicalType.Timestamp),
        ]
    )


def shop_rows(n):
    """Every tenth row has no timestamp, so the absent-value path is exercised
    by the ordinary fixture rather than only by its own test."""
    rows = []
    for i in range(n):
        cols = {
            "price": str((i * 37) % 500),
            "category": CATEGORIES[i % 3],
            "title": TITLES[i % 5],
        }
        if i % 10 != 0:
            cols["created_at"] = f"2026-01-{i % 28 + 1:02d}T12:00:00Z"
        rows.append(Record(i, cols))
    return rows


@pytest.fixture
def table(tmp_path):
    # The store is borrowed, not owned, and is dropped here on purpose: the
    # binding's keep_alive is what stops Python collecting it out from under a
    # live Table. If that ever regresses, every test using this fixture reads
    # freed memory rather than failing somewhere legible.
    store = RecordStore(str(tmp_path / "shop"))
    t = Table(store, shop_schema())
    t.put_batch(shop_rows(300))
    return t


def scan(table, column, keep, absent=False):
    """The oracle: a full pass over the store with no index involved."""
    out = []
    for record in table.scan():
        text = record.columns.get(column)
        if text is None:
            if absent:
                out.append(record.key)
            continue
        if not absent and keep(text):
            out.append(record.key)
    return sorted(out)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def test_indexes_every_scalar_type(table):
    for name in ("price", "category", "title", "created_at"):
        table.create_index(name)
        assert table.has_index(name), name

    info = table.info("category")
    assert info.indexed
    assert info.type == LogicalType.String
    assert info.distinct == 3
    assert not info.unique
    # Three distinct values over 300 rows: duplicated, so composite.
    assert info.encoding == KeyEncoding.Composite
    assert info.rows == 300
    assert info.skipped == 0

    stamped = table.info("created_at")
    assert stamped.rows == 270
    assert stamped.skipped == 30

    table.validate()


def test_describe_has_a_row_per_declared_column(table):
    table.create_index("price")
    described = table.describe()
    assert len(described) == 4
    assert sum(1 for c in described if c.indexed) == 1


def test_bool_and_vector_columns_are_refused_with_the_reason(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    schema = shop_schema()
    schema.add(ColumnDef("in_stock", LogicalType.Bool))
    schema.add(ColumnDef("embedding", LogicalType.Vector, 8))
    t = Table(store, schema)

    with pytest.raises(ValueError, match="bitmap"):
        t.create_index("in_stock")
    with pytest.raises(ValueError, match="put_vector"):
        t.create_index("embedding")


def test_an_ill_typed_record_is_refused_before_anything_is_written(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    t = Table(store, shop_schema())

    with pytest.raises(ValueError):
        t.put(Record(1, {"price": "abc"}))
    with pytest.raises(ValueError):
        t.put(Record(2, {"pirce": "40"}))
    # Neither reached the store: a record that would half-load is refused.
    assert len(t) == 0
    assert t.get(1) is None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", [-1, 0, 37, 250, 499, 500, 9999])
def test_numeric_predicates_agree_with_a_full_scan(table, probe):
    table.create_index("price")
    checks = {
        PredOp.Lt: lambda s: int(s) < probe,
        PredOp.Le: lambda s: int(s) <= probe,
        PredOp.Gt: lambda s: int(s) > probe,
        PredOp.Ge: lambda s: int(s) >= probe,
        PredOp.Eq: lambda s: int(s) == probe,
    }
    for op, keep in checks.items():
        keys, trace = table.select_keys("price", op, probe)
        assert keys == scan(table, "price", keep), (op, probe)
        assert trace.used_index


def test_string_equality_agrees_with_a_full_scan(table):
    table.create_index("category")
    for value in CATEGORIES + ["socks"]:
        keys, _ = table.select_keys("category", PredOp.Eq, value)
        assert keys == scan(table, "category", lambda s, v=value: s == v), value


def test_between_is_inclusive_at_both_ends(table):
    table.create_index("price")
    keys, _ = table.select_keys("price", PredOp.Between, 100, 200)
    assert keys == scan(table, "price", lambda s: 100 <= int(s) <= 200)


def test_prefix_is_served_by_the_index(table):
    table.create_index("title")
    for prefix in ("nike", "nik", "ni", "adidas", "z", ""):
        keys, trace = table.select_keys("title", PredOp.Prefix, prefix)
        assert keys == scan(table, "title",
                            lambda s, p=prefix: s.startswith(p)), prefix
        assert trace.used_index, prefix


def test_contains_and_is_null_scan_and_say_so(table):
    table.create_index("title")
    table.create_index("created_at")

    # No index here can serve an infix match. Answering it anyway, and saying
    # it was a scan, beats refusing.
    keys, trace = table.select_keys("title", PredOp.Contains, "ike")
    assert keys == scan(table, "title", lambda s: "ike" in s)
    assert not trace.used_index
    assert trace.scanned == len(table)
    assert "scanning" in trace.reason

    keys, trace = table.select_keys("created_at", PredOp.IsNull)
    assert keys == scan(table, "created_at", None, absent=True)
    assert len(keys) == 30
    assert not trace.used_index

    assert not op_is_indexable(PredOp.Contains)
    assert not op_is_indexable(PredOp.IsNull)
    assert op_is_indexable(PredOp.Lt)


def test_building_an_index_changes_the_cost_not_the_answer(table):
    want = scan(table, "price", lambda s: int(s) < 250)

    keys, trace = table.select_keys("price", PredOp.Lt, 250)
    assert keys == want
    assert not trace.used_index
    assert trace.scanned == 300

    table.create_index("price")
    keys, trace = table.select_keys("price", PredOp.Lt, 250)
    assert keys == want
    assert trace.used_index
    assert trace.scanned == 0


def test_results_come_back_sorted_by_record_key(table):
    table.create_index("category")
    indexed, _ = table.select_keys("category", PredOp.Eq, "shoes")
    assert indexed == sorted(indexed)
    scanned, _ = table.select_keys("title", PredOp.Contains, "nike")
    assert scanned == sorted(scanned)


def test_select_returns_the_records_the_store_holds(table):
    table.create_index("category")
    rows, _ = table.select("category", PredOp.Eq, "hats")
    assert rows
    for r in rows:
        assert r.columns["category"] == "hats"
        assert table.get(r.key).columns == r.columns


def test_count_matches_select(table):
    table.create_index("price")
    keys, _ = table.select_keys("price", PredOp.Lt, 250)
    assert table.count("price", PredOp.Lt, 250) == len(keys)


def test_a_string_operator_on_a_numeric_column_is_refused(table):
    table.create_index("price")
    with pytest.raises(ValueError):
        table.select_keys("price", PredOp.Prefix, "1")


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_an_update_moves_the_row_in_every_index(table):
    # Declared writable, so the selector may not hand these the static RMI.
    # Asking for a read-only structure and then writing to it is legal -- it
    # costs a rebuild -- but that is a different test.
    table.create_index("price", write_fraction=0.3)
    table.create_index("category", write_fraction=0.3)

    result = table.put(Record(7, {"price": "9999", "category": "shoes",
                                  "title": "nike air"}))
    assert not result.created
    assert result.indexes_touched > 0
    # A composite key takes a mid-range write in O(log n); nothing renumbers.
    assert result.rebuilds_triggered == 0

    keys, _ = table.select_keys("price", PredOp.Eq, 9999)
    assert keys == [7]
    old, _ = table.select_keys("price", PredOp.Eq, 7 * 37)
    assert 7 not in old
    table.validate()


def test_an_unchanged_column_is_not_touched(table):
    table.create_index("price", write_fraction=0.3)
    table.create_index("category", write_fraction=0.3)

    same = table.get(5)
    same.columns["title"] = "puma go"
    result = table.put(same)
    assert result.indexes_touched == 0
    table.validate()


def test_erasing_removes_the_row_from_every_index(table):
    table.create_index("price", write_fraction=0.3)
    table.create_index("category", write_fraction=0.3)

    result = table.erase(3)
    assert result.indexes_touched > 0
    # A delete is exact in every structure here and never rebuilds.
    assert result.rebuilds_triggered == 0
    assert table.get(3) is None

    keys, _ = table.select_keys("price", PredOp.Ge, 0)
    assert 3 not in keys
    table.validate()


def test_update_changes_only_the_named_columns(table):
    table.create_index("price", write_fraction=0.3)
    before = table.get(4).columns["title"]
    table.update(4, {"price": 777})

    after = table.get(4)
    assert after.columns["price"] == "777"
    assert after.columns["title"] == before
    keys, _ = table.select_keys("price", PredOp.Eq, 777)
    assert keys == [4]


def test_a_natively_keyed_column_rebuilds_when_a_write_makes_it_non_unique(tmp_path):
    # A native key was chosen because the values were unique; an insert that
    # collides makes them not, and the tree overwrote the colliding row's entry
    # rather than storing both. Without the rebuild the index would silently be
    # missing a row.
    store = RecordStore(str(tmp_path / "s"))
    t = Table(store, shop_schema())
    t.put_batch([Record(i, {"category": f"c{i}"}) for i in range(40)])
    t.create_index("category", write_fraction=0.5)
    assert t.info("category").encoding == KeyEncoding.Native

    result = t.put(Record(1000, {"category": "c7"}))
    assert result.rebuilds_triggered == 1

    keys, _ = t.select_keys("category", PredOp.Eq, "c7")
    assert keys == [7, 1000]
    assert t.info("category").encoding == KeyEncoding.Composite
    t.validate()


def test_a_batch_pays_one_rebuild_where_a_loop_pays_one_per_record(tmp_path):
    # The workload matters, and an earlier version of this test got it wrong.
    # It used a uniqueness collision and asserted the batch paid one rebuild --
    # true, and vacuous, because a loop also pays exactly one there: the first
    # rebuild switches the column to a composite key which absorbs every later
    # collision. Only the E6 benchmark noticed.
    #
    # The case where batching matters is a column planned read-only and then
    # written to: its structure is build-only, so the rebuild produces the same
    # build-only structure and the next write dirties it again.
    def build(path):
        store = RecordStore(str(path))
        t = Table(store, shop_schema())
        t.put_batch([Record(i, {"price": str(i * 7)}) for i in range(3000)])
        t.create_index("price")  # read-only: the RMI is a legal choice
        return t

    extra = [Record(100_000 + i, {"price": str(1_000_000 + i * 7)})
             for i in range(15)]

    looped = build(tmp_path / "loop")
    if looped.info("price").kind != IndexKind.RMI:
        pytest.skip("the tree won this column here; no build-only structure "
                    "means no rebuild to batch away")
    for record in extra:
        looped.put(record)

    batched = build(tmp_path / "batch")
    result = batched.put_batch(extra)

    assert result.rows_created == len(extra)
    assert batched.rebuilds() == 1, "a batch rebuilds once, at the end"
    assert looped.rebuilds() > 1, (
        "a loop over the same records must rebuild per record; if it does not, "
        "put_batch's docstring overstates its value"
    )
    batched.validate()
    looped.validate()


def test_a_mixed_workload_stays_in_agreement_with_the_store(table):
    import random

    table.create_index("price", write_fraction=0.3)
    table.create_index("category", write_fraction=0.3)

    rng = random.Random(20260813)
    for _ in range(600):
        action = rng.randrange(3)
        key = rng.randrange(400)
        if action == 0:
            table.put(Record(key, {
                "price": str(rng.randrange(600)),
                "category": CATEGORIES[rng.randrange(3)],
                "title": TITLES[rng.randrange(5)],
            }))
        elif action == 1:
            table.erase(key)
        elif table.get(key) is not None:
            table.update(key, {"price": rng.randrange(600)})

    table.validate()
    for probe in (0, 150, 300, 599, 600):
        keys, _ = table.select_keys("price", PredOp.Lt, probe)
        assert keys == scan(table, "price", lambda s, p=probe: int(s) < p), probe
    for value in CATEGORIES:
        keys, _ = table.select_keys("category", PredOp.Eq, value)
        assert keys == scan(table, "category", lambda s, v=value: s == v), value


# ---------------------------------------------------------------------------
# Reopen
# ---------------------------------------------------------------------------


def test_records_come_back_and_the_catalog_replays_the_decision(tmp_path):
    # The moment the whole stack is visible at once: module 1 brings the rows
    # back, module 4's catalog brings the *decision* back without re-measuring
    # it, and the typed schema is what makes a stored plan interpretable.
    path = str(tmp_path / "shop")
    store = RecordStore(path)
    t = Table(store, shop_schema())
    t.put_batch(shop_rows(400))
    t.create_index("price")
    t.create_index("category")
    chosen = t.info("price").kind
    t.checkpoint()
    del t
    store.close()
    del store

    store2 = RecordStore(path)
    t2 = Table.open(store2)
    assert len(t2) == 400
    assert len(t2.schema) == 4
    assert t2.schema.type_of("category") == LogicalType.String

    t2.create_index("price")
    assert t2.info("price").kind == chosen
    t2.validate()


def test_a_retyped_column_is_refused(tmp_path):
    path = str(tmp_path / "shop")
    store = RecordStore(path)
    t = Table(store, shop_schema())
    t.put_batch(shop_rows(20))
    t.save()
    del t
    store.close()
    del store

    changed = Schema([
        ColumnDef("price", LogicalType.String),  # was Int64
        ColumnDef("category", LogicalType.String),
        ColumnDef("title", LogicalType.String),
        ColumnDef("created_at", LogicalType.Timestamp),
    ])
    store2 = RecordStore(path)
    with pytest.raises(RuntimeError, match="Retyping"):
        Table(store2, changed)


def test_opening_without_a_stored_schema_says_what_is_missing(tmp_path):
    store = RecordStore(str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="no schema"):
        Table.open(store)
