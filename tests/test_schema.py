"""The typed column layer, through the Python bridge.

Two claims are under test and they pull in opposite directions, which is why
both need saying:

* **A string column is indexed as a string.** It gets a B+ tree keyed on the
  string itself, it answers prefix predicates, and it is never offered a
  learned index -- because a model fitted to string ranks would be fitted to an
  ordering the model itself imposed.
* **The int64 path is untouched.** Every figure in plans/important.md came
  through it, so a refactor that quietly changed its shape would leave the
  numbers describing code that no longer exists.
"""

import pytest

from hylis import (
    ColumnDef,
    ColumnIndex,
    CompareOp,
    IndexKind,
    IndexPlan,
    KeyEncoding,
    LogicalType,
    Record,
    Schema,
    candidates_for,
    format_value,
    measure_shape,
    parse_value,
    prefix_upper_bound,
    type_supports_rmi,
)

BRANDS = ["adidas", "nib", "nike", "nikelab", "nikon", "puma", "zara"]


def rows(n):
    return list(range(n))


def composite_column(type_, keys, row_ids):
    """Build with the composite encoding, whatever choose_index would pick.

    These tests are about the composite structure, and a low-cardinality column
    is now also a bitmap candidate -- so letting the selector decide would make
    them pass or fail on which candidate won a timing race that day.
    """
    plan = IndexPlan()
    plan.kind = IndexKind.BPlusTree
    plan.type = type_
    plan.encoding = KeyEncoding.Composite
    plan.btree_order = 32
    return ColumnIndex.build_typed_with(type_, keys, row_ids, plan)


# ---------------------------------------------------------------------------
# Types and values
# ---------------------------------------------------------------------------


def test_only_numeric_types_can_carry_a_learned_index():
    assert type_supports_rmi(LogicalType.Int64)
    assert type_supports_rmi(LogicalType.Double)
    assert type_supports_rmi(LogicalType.Timestamp)
    assert not type_supports_rmi(LogicalType.String)
    assert not type_supports_rmi(LogicalType.Bool)


@pytest.mark.parametrize(
    "type_, text, expected",
    [
        (LogicalType.Int64, "-42", -42),
        (LogicalType.Double, "1.5", 1.5),
        (LogicalType.String, "  Nike ", "  Nike "),
        (LogicalType.Bool, "TRUE", True),
        (LogicalType.Bool, "0", False),
    ],
)
def test_values_parse_to_their_declared_type(type_, text, expected):
    assert parse_value(type_, text) == expected


@pytest.mark.parametrize(
    "type_, text",
    [
        (LogicalType.Int64, "12abc"),
        (LogicalType.Int64, "1.5"),
        (LogicalType.Int64, ""),
        (LogicalType.Double, "4.2kg"),
        (LogicalType.Double, "nan"),
        (LogicalType.Double, "inf"),
        (LogicalType.Bool, "yes"),
        (LogicalType.Timestamp, "not-a-date"),
        (LogicalType.Timestamp, "2026-13-01"),
    ],
)
def test_bad_values_are_refused_rather_than_coerced(type_, text):
    # Accepting "12abc" as 12 would turn a data-entry typo into a silently
    # wrong index, which is the failure mode with no symptom.
    with pytest.raises(ValueError):
        parse_value(type_, text)


@pytest.mark.parametrize(
    "text, canonical",
    [
        ("2026-08-13T14:30:00Z", "2026-08-13T14:30:00Z"),
        ("2026-08-13 14:30:00", "2026-08-13T14:30:00Z"),
        ("2026-08-13", "2026-08-13T00:00:00Z"),
        ("2026-08-13T14:30:00.250Z", "2026-08-13T14:30:00.250Z"),
        ("0", "1970-01-01T00:00:00Z"),
        # Pre-epoch, where integer division toward zero silently produces the
        # previous day if the conversion is written the obvious way.
        ("1969-12-31T23:59:59Z", "1969-12-31T23:59:59Z"),
        ("1600-02-29T12:00:00Z", "1600-02-29T12:00:00Z"),
    ],
)
def test_timestamps_normalise_to_iso(text, canonical):
    value = parse_value(LogicalType.Timestamp, text)
    assert format_value(LogicalType.Timestamp, value) == canonical


def test_doubles_round_trip_through_their_text_form():
    # Six significant digits, the default, would lose the low bits of a price
    # or a coordinate.
    for v in (0.1, 1 / 3, 1e-300, 1.7976931348623157e308, -2.5e-17):
        text = format_value(LogicalType.Double, v)
        assert parse_value(LogicalType.Double, text) == v


def test_prefix_upper_bound_works_on_bytes():
    # Bytes, not text, because that is the ordering the B+ tree uses. A str is
    # taken as its UTF-8 encoding, and incrementing the last byte of a
    # multi-byte character can produce a sequence that is not valid UTF-8 --
    # correct as a bound, and the reason this returns bytes rather than str.
    assert prefix_upper_bound("nike") == b"nikf"
    assert prefix_upper_bound(b"a\xff") == b"b"


def test_prefix_upper_bound_reports_when_there_is_none():
    # Every string beginning with 0xFF sorts above it and there is no successor
    # to stop at, so the caller must fall back to an unbounded scan.
    assert prefix_upper_bound(b"\xff\xff") is None
    assert prefix_upper_bound("") is None


def test_prefix_on_a_non_ascii_column_still_matches_a_linear_scan():
    # The tree orders by the UTF-8 bytes, and so does the prefix bound, so the
    # two agree even where the successor byte is not itself valid UTF-8.
    values = sorted(["cafe", "café", "caffe", "caz", "da"])
    c = ColumnIndex.build_typed(LogicalType.String, values, rows(len(values)))
    for p in ("ca", "caf", "café", "d"):
        expected = [i for i, w in enumerate(values) if w.startswith(p)]
        assert c.query_prefix(p) == expected, p


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def shop_schema():
    return Schema(
        [
            ColumnDef("price", LogicalType.Int64),
            ColumnDef("weight", LogicalType.Double),
            ColumnDef("category", LogicalType.String),
            ColumnDef("in_stock", LogicalType.Bool),
            ColumnDef("created_at", LogicalType.Timestamp),
            ColumnDef("embedding", LogicalType.Vector, 128),
        ]
    )


def test_schema_reports_its_columns():
    s = shop_schema()
    assert len(s) == 6
    assert "price" in s
    assert "pirce" not in s
    assert s.type_of("category") == LogicalType.String
    assert s.column("embedding").dim == 128
    assert s.scalar_columns() == [
        "price", "weight", "category", "in_stock", "created_at"
    ]
    assert s.vector_columns() == ["embedding"]


def test_an_unknown_column_is_an_error():
    # Catching `pirce` is the whole reason to have a schema rather than a
    # convention. A column you want to store but not index costs one line to
    # declare.
    s = shop_schema()
    assert not s.accepts(Record(1, {"pirce": "40"}))
    with pytest.raises(ValueError, match="pirce"):
        s.validate(Record(1, {"pirce": "40"}))


def test_a_value_of_the_wrong_type_is_an_error():
    s = shop_schema()
    assert not s.accepts(Record(1, {"price": "abc"}))
    assert not s.accepts(Record(1, {"in_stock": "maybe"}))
    assert s.accepts(Record(1, {"price": "4000", "category": "shoes"}))


def test_a_missing_column_is_not_an_error():
    # Asymmetric on purpose: an absent value is a row that matches no predicate
    # on that column, which is close enough to SQL's three-valued logic.
    s = shop_schema()
    assert s.accepts(Record(1, {"price": "4000"}))
    assert s.accepts(Record(2, {}))


def test_an_embedding_in_the_record_payload_is_refused():
    # A 128-float vector is ~700 bytes of base64 per row; allowing it would
    # make the write-ahead log the dominant cost of the system.
    s = shop_schema()
    with pytest.raises(ValueError, match="put_vector"):
        s.validate(Record(1, {"embedding": "0.1,0.2"}))


def test_a_duplicate_column_is_refused():
    s = Schema([ColumnDef("price", LogicalType.Int64)])
    with pytest.raises(ValueError):
        s.add(ColumnDef("price", LogicalType.Double))


def test_a_vector_column_needs_a_dimension():
    s = Schema()
    with pytest.raises(ValueError):
        s.add(ColumnDef("e", LogicalType.Vector))
    with pytest.raises(ValueError):
        s.add(ColumnDef("price", LogicalType.Int64, 8))
    s.add(ColumnDef("e", LogicalType.Vector, 64))
    assert s.column("e").dim == 64


def test_a_schema_survives_a_save_and_load():
    # Without this a stored plan is uninterpretable: "encoding: composite" says
    # nothing unless the key type is known.
    s = shop_schema()
    back = Schema.parse_json(s.serialize())
    assert len(back) == len(s)
    for c in s.columns():
        assert back.type_of(c.name) == c.type
        assert back.column(c.name).dim == c.dim


def test_schema_parses_and_formats_through_the_declared_type():
    s = shop_schema()
    assert s.parse("price", "4000") == 4000
    assert s.format("created_at", s.parse("created_at", "2026-08-13")) == (
        "2026-08-13T00:00:00Z"
    )
    with pytest.raises(ValueError):
        s.parse("price", "abc")


# ---------------------------------------------------------------------------
# Typed columns
# ---------------------------------------------------------------------------


def test_a_string_column_is_indexed_as_a_string():
    c = ColumnIndex.build_typed(LogicalType.String, BRANDS, rows(len(BRANDS)))
    assert c.kind == IndexKind.BPlusTree
    assert c.type == LogicalType.String
    assert c.encoding == KeyEncoding.Native
    assert c.lookup("nike") == [2]
    assert c.lookup("reebok") == []


def test_string_predicates_order_by_bytes():
    # "10" < "9" lexicographically and the other way round numerically. The
    # column's type is what decides, which is why it is recorded.
    c = ColumnIndex.build_typed(LogicalType.String, ["10", "9", "apple"], rows(3))
    assert c.query(CompareOp.Lt, "9") == [0]
    numeric = ColumnIndex.build_typed(LogicalType.Int64, [9, 10], rows(2))
    assert numeric.query(CompareOp.Lt, 10) == [0]


def test_prefix_is_exact_including_its_edges():
    c = ColumnIndex.build_typed(LogicalType.String, BRANDS, rows(len(BRANDS)))
    # "nib" begins with "ni" but not with "nik".
    assert len(c.query_prefix("ni")) == 4
    assert len(c.query_prefix("nik")) == 3
    assert c.query_prefix("nike") == [2, 3]
    assert c.query_prefix("zzz") == []
    # The empty prefix has no successor, so it takes the unbounded fallback.
    assert len(c.query_prefix("")) == len(BRANDS)


def test_prefix_matches_a_linear_scan():
    c = ColumnIndex.build_typed(LogicalType.String, BRANDS, rows(len(BRANDS)))
    for p in ("a", "n", "ni", "nik", "nike", "p", "z", ""):
        expected = [i for i, w in enumerate(BRANDS) if w.startswith(p)]
        assert c.query_prefix(p) == expected, p


def test_prefix_is_refused_on_a_non_string_column():
    c = ColumnIndex.build_typed(LogicalType.Int64, [1, 2, 3], rows(3))
    with pytest.raises(ValueError):
        c.query_prefix("nike")


def test_a_wrong_typed_predicate_raises_rather_than_answering():
    c = ColumnIndex.build_typed(LogicalType.String, ["a", "b"], rows(2))
    with pytest.raises(ValueError):
        c.query(CompareOp.Lt, 5)


def test_duplicated_values_get_the_composite_encoding():
    values = ["bags"] * 4 + ["hats"] * 4 + ["shoes"] * 4
    c = composite_column(LogicalType.String, values, rows(12))

    assert c.encoding == KeyEncoding.Composite
    assert not c.is_native
    assert c.plan().distinct == 3
    assert len(c) == 12
    c.validate()

    assert len(c.lookup("shoes")) == 4
    assert len(c.query(CompareOp.Eq, "hats")) == 4
    assert len(c.query(CompareOp.Lt, "shoes")) == 8
    assert len(c.query(CompareOp.Le, "shoes")) == 12
    assert len(c.query(CompareOp.Gt, "bags")) == 8
    assert len(c.query(CompareOp.Ge, "hats")) == 8


def test_find_returns_none_on_a_composite_column_rather_than_guessing():
    values = [10, 10, 10, 20]
    c = composite_column(LogicalType.Int64, values, rows(4))
    assert c.encoding == KeyEncoding.Composite
    # Three rows hold 10. Returning one of them would be a coin toss, so the
    # caller is steered to lookup() by a None.
    assert c.find(10) is None
    assert c.lookup(10) == [0, 1, 2]


def test_composite_erase_needs_the_row_id():
    c = composite_column(LogicalType.String, ["a", "a", "b"], rows(3))
    # The value alone names two rows, so the value-only overload refuses.
    numeric = composite_column(LogicalType.Int64, [1, 1, 2], rows(3))
    assert not numeric.is_native
    with pytest.raises(ValueError):
        numeric.erase(1)

    # erase_row names the row, and only that row goes. Row 0 is an ordinary
    # row id here, not a stand-in for "unspecified" -- an earlier version used
    # it as a sentinel and threw on the one row it should have removed.
    assert c.erase_row("a", 0)
    assert c.lookup("a") == [1]
    assert c.erase_row("a", 1)
    assert c.lookup("a") == []
    # A row that does not hold this value is not removed, and says so.
    assert not c.erase_row("b", 99)
    c.validate()


def test_every_operator_agrees_with_a_python_oracle():
    import random

    rng = random.Random(7)
    pairs = sorted((rng.randrange(20), i) for i in range(1500))
    keys = [k for k, _ in pairs]
    values = [v for _, v in pairs]

    c = composite_column(LogicalType.Int64, keys, values)
    assert c.encoding == KeyEncoding.Composite

    checks = {
        CompareOp.Eq: lambda k, p: k == p,
        CompareOp.Lt: lambda k, p: k < p,
        CompareOp.Le: lambda k, p: k <= p,
        CompareOp.Gt: lambda k, p: k > p,
        CompareOp.Ge: lambda k, p: k >= p,
    }
    for probe in range(-1, 22):
        for op, keep in checks.items():
            want = sorted(v for k, v in pairs if keep(k, probe))
            assert sorted(c.query(op, probe)) == want, (op, probe)


def test_a_double_column_is_exact():
    keys = sorted({round(i * 0.37, 6) for i in range(5000)})
    c = ColumnIndex.build_typed(LogicalType.Double, keys, rows(len(keys)))
    for i in range(0, len(keys), 97):
        assert c.lookup(keys[i]) == [i]
    assert len(c.query(CompareOp.Lt, keys[100])) == 100


def test_a_timestamp_column_is_detected_as_monotone():
    stamps = [1_700_000_000_000 + i * 60_000 for i in range(2000)]
    c = ColumnIndex.build_typed(LogicalType.Timestamp, stamps, rows(2000))
    assert c.type == LogicalType.Timestamp
    # Monotone is what makes an append-only write path legal, and is why
    # Timestamp is its own type rather than an alias for Int64.
    assert c.plan().monotone
    assert c.find(stamps[100]) == 100


def test_a_bool_column_is_only_ever_offered_a_bitmap():
    # Two distinct values means an ordered index over it is a sorted list of
    # every row in the table providing no ordering benefit whatever.
    shape = measure_shape([0, 0, 1, 1], rows(4))
    plans = candidates_for(LogicalType.Bool, shape)
    assert len(plans) == 1
    assert plans[0].kind == IndexKind.Bitmap
    assert plans[0].encoding == KeyEncoding.Dictionary


# ---------------------------------------------------------------------------
# The candidate filter
# ---------------------------------------------------------------------------


def test_a_string_column_is_never_offered_a_learned_index():
    shape = measure_shape(list(range(1000)), list(range(1000)))
    plans = candidates_for(LogicalType.String, shape)
    assert plans
    # The claim is about the learned index, not about the tree being alone: a
    # low-cardinality string column is also a bitmap candidate.
    assert all(p.kind not in (IndexKind.RMI, IndexKind.DynamicRMI) for p in plans)
    assert any(p.kind == IndexKind.BPlusTree for p in plans)


def test_a_unique_numeric_column_is_offered_one():
    shape = measure_shape(list(range(1000)), list(range(1000)))
    plans = candidates_for(LogicalType.Int64, shape)
    assert any(p.kind == IndexKind.RMI for p in plans)


def test_duplicates_disqualify_the_learned_index():
    # RMIndex.build raises on a repeated key, and a (value, row) pair has no
    # cast to a float. The composite tree is the only structure left.
    keys = [k // 25 for k in range(1000)]
    shape = measure_shape(keys, list(range(1000)))
    assert not shape.unique
    plans = candidates_for(LogicalType.Int64, shape)
    assert plans
    for p in plans:
        assert p.kind not in (IndexKind.RMI, IndexKind.DynamicRMI)
        # Nor a native key, which maps one value to one row.
        assert p.encoding != KeyEncoding.Native
    # 40 distinct values is few enough that the bitmap is offered alongside the
    # composite tree; which of the two wins is measured, not asserted.
    assert any(p.kind == IndexKind.BPlusTree for p in plans)
    assert any(p.kind == IndexKind.Bitmap for p in plans)


def test_a_writable_column_is_still_never_offered_the_static_rmi():
    shape = measure_shape(list(range(1000)), list(range(1000)))
    plans = candidates_for(LogicalType.Int64, shape, write_fraction=0.3)
    assert plans
    assert all(p.kind != IndexKind.RMI for p in plans)


def test_shape_is_measured_in_one_pass():
    shape = measure_shape([1, 1, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5])
    assert shape.rows == 6
    assert shape.distinct == 3
    assert not shape.unique
    assert shape.monotone
    assert shape.duplicate_fraction == pytest.approx(0.5)
    # Not monotone: the rows did not arrive in key order.
    assert not measure_shape([1, 1, 2, 3, 3, 3], [5, 0, 2, 1, 4, 3]).monotone


# ---------------------------------------------------------------------------
# The int64 path must be exactly as it was
# ---------------------------------------------------------------------------


def test_the_int64_path_is_unchanged_by_type_erasure():
    keys = [i * 3 for i in range(5000)]
    c = ColumnIndex.build(keys, rows(5000))
    assert c.type == LogicalType.Int64
    assert c.encoding == KeyEncoding.Native
    assert c.is_native
    assert c.find(300) == 100
    assert c.find(301) is None
    assert c.range(0, 29) == list(range(10))
    assert c.range_query(CompareOp.Lt, 30) == list(range(10))
    assert 0.0 < c.plan().ns_per_lookup < 10_000.0
