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
    # Three distinct values over 300 rows. Duplicated, so a native key is out --
    # and low-cardinality enough that the bitmap is a candidate and wins on
    # measurement, which is what the Dictionary encoding records.
    assert info.kind == IndexKind.Bitmap
    assert info.encoding == KeyEncoding.Dictionary
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


def test_a_bool_column_gets_a_bitmap_and_nothing_else(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    schema = shop_schema()
    schema.add(ColumnDef("in_stock", LogicalType.Bool))
    t = Table(store, schema)

    rows = shop_rows(200)
    for i, r in enumerate(rows):
        # .set, not .columns[...] = ... -- reading .columns builds a new dict,
        # so assigning into it mutates a temporary and is silently lost.
        r.set("in_stock", "false" if i % 4 == 0 else "true")
    t.put_batch(rows)

    info = t.create_index("in_stock")
    # Two distinct values: an ordered index over it is a sorted list of every
    # row providing no ordering benefit, so the bitmap is the only candidate.
    assert info.kind == IndexKind.Bitmap
    assert info.encoding == KeyEncoding.Dictionary
    assert info.distinct == 2

    assert t.count("in_stock", PredOp.Eq, True) == 150
    assert t.count("in_stock", PredOp.Eq, False) == 50
    keys, _ = t.select_keys("in_stock", PredOp.Eq, False)
    assert len(keys) == 50
    assert all(k % 4 == 0 for k in keys)
    t.validate()


def test_a_vector_column_is_refused_with_the_reason(tmp_path):
    store = RecordStore(str(tmp_path / "s"))
    schema = shop_schema()
    schema.add(ColumnDef("embedding", LogicalType.Vector, 8))
    t = Table(store, schema)

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
    # Genuinely change a column other than the indexed ones. Assigning into
    # .columns would have changed nothing at all, which made an earlier version
    # of this test pass without testing anything.
    assert same.columns["title"] != "puma go"
    same.set("title", "puma go")
    result = table.put(same)

    assert table.get(5).columns["title"] == "puma go"
    # price and category are unchanged, so neither index should have been
    # erased and reinserted for no reason.
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
    # A thousand distinct values over a thousand rows puts the column above
    # both bitmap thresholds, so the bitmap is not a candidate and the native
    # key is guaranteed. At forty values the two were close enough that the
    # winner varied between runs and so did this test.
    store = RecordStore(str(tmp_path / "s"))
    t = Table(store, shop_schema())
    t.put_batch([Record(i, {"category": f"c{i}"}) for i in range(1000)])
    t.create_index("category", write_fraction=0.5)
    assert t.info("category").encoding == KeyEncoding.Native

    result = t.put(Record(100_000, {"category": "c7"}))
    assert result.rebuilds_triggered == 1

    keys, _ = t.select_keys("category", PredOp.Eq, "c7")
    assert keys == [7, 100_000]
    # Which structure it landed on is a measurement outcome; what matters is
    # that it is no longer one mapping a value to a single row.
    assert t.info("category").encoding != KeyEncoding.Native
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


# ---------------------------------------------------------------------------
# Conjunctions, disjunctions and counting
# ---------------------------------------------------------------------------


def bitmap_table(tmp_path, n=300):
    store = RecordStore(str(tmp_path / "conj"))
    schema = shop_schema()
    schema.add(ColumnDef("in_stock", LogicalType.Bool))
    t = Table(store, schema)
    rows = shop_rows(n)
    for i, r in enumerate(rows):
        r.set("in_stock", "false" if i % 3 == 0 else "true")
    t.put_batch(rows)
    return t


def test_a_conjunction_over_two_bitmaps_uses_the_and_path(tmp_path):
    t = bitmap_table(tmp_path)
    t.create_index("category")
    t.create_index("in_stock")

    keys, trace = t.select_all([("category", PredOp.Eq, "shoes"),
                                ("in_stock", PredOp.Eq, True)])
    want = sorted(r.key for r in t.scan()
                  if r.columns["category"] == "shoes"
                  and r.columns["in_stock"] == "true")
    assert keys == want
    assert "bitmap AND" in trace.reason


def test_a_mixed_conjunction_falls_back_to_the_merge_and_agrees(tmp_path):
    t = bitmap_table(tmp_path)
    t.create_index("category")
    t.create_index("price")

    keys, trace = t.select_all([("category", PredOp.Eq, "shoes"),
                                ("price", PredOp.Lt, 250)])
    want = sorted(r.key for r in t.scan()
                  if r.columns["category"] == "shoes"
                  and int(r.columns["price"]) < 250)
    assert keys == want
    assert "sorted merge" in trace.reason


def test_a_conjunction_containing_an_unindexable_operator_is_still_right(tmp_path):
    t = bitmap_table(tmp_path)
    t.create_index("category")
    t.create_index("title")

    keys, _ = t.select_all([("category", PredOp.Eq, "shoes"),
                            ("title", PredOp.Contains, "ike")])
    want = sorted(r.key for r in t.scan()
                  if r.columns["category"] == "shoes"
                  and "ike" in r.columns["title"])
    assert keys == want


def test_no_predicates_means_every_row(tmp_path):
    t = bitmap_table(tmp_path, n=120)
    keys, _ = t.select_all([])
    assert keys == sorted(range(120))


def test_an_impossible_conjunction_is_empty(tmp_path):
    t = bitmap_table(tmp_path)
    t.create_index("category")
    t.create_index("price")
    keys, _ = t.select_all([("category", PredOp.Eq, "socks"),
                            ("price", PredOp.Lt, 250)])
    assert keys == []


def test_a_disjunction_is_the_deduplicated_union(tmp_path):
    t = bitmap_table(tmp_path)
    t.create_index("category")

    keys, _ = t.select_any([("category", PredOp.Eq, "shoes"),
                            ("category", PredOp.Eq, "hats")])
    want = sorted(r.key for r in t.scan()
                  if r.columns["category"] in ("shoes", "hats"))
    assert keys == want
    assert len(keys) == len(set(keys))


def test_count_agrees_with_select_on_every_family(table):
    table.create_index("category")   # low cardinality: a bitmap
    table.create_index("price")      # wide: an ordered index

    for value in CATEGORIES:
        keys, _ = table.select_keys("category", PredOp.Eq, value)
        assert table.count("category", PredOp.Eq, value) == len(keys), value
    for probe in (0, 100, 250, 499, 9999):
        keys, _ = table.select_keys("price", PredOp.Lt, probe)
        assert table.count("price", PredOp.Lt, probe) == len(keys), probe


def test_create_index_as_pins_the_family(tmp_path):
    # choose_index times lookups, where a bitmap loses from about 64 distinct
    # values upward -- so a column whose workload is counts or conjunctions
    # cannot get one by asking nicely.
    store = RecordStore(str(tmp_path / "pin"))
    schema = Schema([ColumnDef("bucket", LogicalType.Int64)])
    t = Table(store, schema)
    t.put_batch([Record(i, {"bucket": str(i % 500)}) for i in range(5000)])

    chosen = t.create_index("bucket")
    forced = t.create_index_as("bucket", IndexKind.Bitmap)
    assert forced.kind == IndexKind.Bitmap
    assert forced.encoding == KeyEncoding.Dictionary
    # Same answers either way; only the cost differs.
    keys, _ = t.select_keys("bucket", PredOp.Lt, 100)
    assert len(keys) == 1000
    assert t.count("bucket", PredOp.Lt, 100) == 1000
    t.validate()
    assert chosen.name == "bucket"
