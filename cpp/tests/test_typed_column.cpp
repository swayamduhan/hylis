// Tests for the typed column layer of ColumnIndex.
//
// The claim under test is the one at the top of column_index.hpp, restated for
// a world where columns have types: the *type* decides which structures are
// candidates, and only among the survivors does measurement decide. A string
// column handed a learned index would not give a slow answer, it would give a
// meaningless one — so that combination has to be impossible to reach rather
// than merely unlikely to be chosen.
//
// The second theme is that the int64 path this project already published
// numbers for is untouched. Type erasure that quietly changed the cost of a
// point lookup would invalidate every figure in plans/important.md.

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "index/column_index.hpp"
#include "index/index_catalog.hpp"

using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::ColumnShape;
using hylis::index::ColumnValue;
using hylis::index::CompareOp;
using hylis::index::Datum;
using hylis::index::IndexCatalog;
using hylis::index::IndexKind;
using hylis::index::IndexPlan;
using hylis::index::KeyEncoding;
using hylis::index::LogicalType;
using hylis::index::Workload;
using hylis::index::candidates_for;
using hylis::index::measure_shape;
using hylis::index::symbol_of;

namespace {

const std::vector<std::string>& brands() {
    static const std::vector<std::string> words = {
        "adidas", "nib", "nike", "nikelab", "nikon", "puma", "zara"};
    return words;
}

std::vector<ColumnValue> identity(std::size_t n) {
    std::vector<ColumnValue> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<ColumnValue>(i);
    return out;
}

// A duplicated categorical column, already sorted by (value, row) the way an
// extractor would hand it over.
void categories(std::vector<std::string>* values, std::vector<ColumnValue>* rows) {
    const char* names[] = {"bags", "hats", "shoes"};
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 4; ++r) {
            values->push_back(names[c]);
            rows->push_back(c * 4 + r);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Strings are indexed as strings
// ---------------------------------------------------------------------------

TEST(TypedColumns, AStringColumnIsIndexedNativelyAndNeverEncoded) {
    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, brands(),
                                                   identity(brands().size()));

    EXPECT_EQ(c.kind(), IndexKind::BPlusTree);
    EXPECT_EQ(c.type(), LogicalType::String);
    EXPECT_EQ(c.encoding(), KeyEncoding::Native);
    EXPECT_EQ(c.lookup(Datum{std::string("nike")}), std::vector<ColumnValue>{2});
    EXPECT_TRUE(c.lookup(Datum{std::string("reebok")}).empty());
}

TEST(TypedColumns, StringPredicatesOrderByBytesNotByNumericValue) {
    const std::vector<std::string> values = {"10", "9", "apple", "banana"};
    const ColumnIndex c =
        ColumnIndex::build_typed(LogicalType::String, values, identity(4));

    // "10" < "9" lexicographically, the opposite of the numeric answer.
    // Getting this wrong silently is exactly why the type is recorded.
    const auto below = c.query(CompareOp::Lt, Datum{std::string("9")});
    ASSERT_EQ(below.size(), 1u);
    EXPECT_EQ(below.front(), 0);
}

TEST(TypedColumns, PrefixIsExactAcrossItsEdgeCases) {
    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, brands(),
                                                   identity(brands().size()));

    // "nib" begins with "ni" but not with "nik".
    EXPECT_EQ(c.query_prefix("ni").size(), 4u);
    EXPECT_EQ(c.query_prefix("nik").size(), 3u);
    EXPECT_EQ(c.query_prefix("nike"), (std::vector<ColumnValue>{2, 3}));
    EXPECT_TRUE(c.query_prefix("zzz").empty());
    // The empty prefix has no successor to stop at, so it takes the unbounded
    // fallback and must still return everything.
    EXPECT_EQ(c.query_prefix("").size(), brands().size());
}

TEST(TypedColumns, PrefixMatchesALinearScanOnRandomData) {
    std::mt19937_64 rng(31);
    std::vector<std::string> values;
    for (int i = 0; i < 3000; ++i) {
        std::string s;
        const int len = 1 + static_cast<int>(rng() % 6);
        for (int j = 0; j < len; ++j) s += static_cast<char>('a' + rng() % 4);
        values.push_back(s);
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, values,
                                                   identity(values.size()));

    const std::vector<std::string> probes = {"a",  "ab", "abc",  "d",
                                             "dd", "zz", "abcd"};
    for (const std::string& p : probes) {
        std::vector<ColumnValue> want;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i].rfind(p, 0) == 0) want.push_back(static_cast<ColumnValue>(i));
        }
        EXPECT_EQ(c.query_prefix(p), want) << "prefix '" << p << "'";
    }
}

TEST(TypedColumns, PrefixIsRefusedOnANonStringColumn) {
    std::vector<ColumnKey> keys;
    for (int i = 0; i < 500; ++i) keys.push_back(i * 3);
    const ColumnIndex c = ColumnIndex::build(keys, identity(keys.size()));
    EXPECT_THROW(c.query_prefix("nike"), std::invalid_argument);
}

TEST(TypedColumns, AWrongTypedPredicateThrowsRatherThanAnswering) {
    const std::vector<std::string> values = {"a", "b"};
    const ColumnIndex c =
        ColumnIndex::build_typed(LogicalType::String, values, identity(2));
    EXPECT_THROW(c.query(CompareOp::Lt, Datum{std::int64_t{5}}),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Duplicated columns and the composite encoding
// ---------------------------------------------------------------------------

TEST(CompositeEncoding, DuplicatedValuesAreIndexableAndEveryOperatorIsExact) {
    std::vector<std::string> values;
    std::vector<ColumnValue> rows;
    categories(&values, &rows);

    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, values, rows);

    EXPECT_EQ(c.encoding(), KeyEncoding::Composite);
    EXPECT_FALSE(c.is_native());
    EXPECT_EQ(c.plan().distinct, 3u);
    EXPECT_EQ(c.size(), 12u);
    EXPECT_NO_THROW(c.validate());

    EXPECT_EQ(c.lookup(Datum{std::string("shoes")}).size(), 4u);
    EXPECT_EQ(c.query(CompareOp::Eq, Datum{std::string("hats")}).size(), 4u);
    EXPECT_EQ(c.query(CompareOp::Lt, Datum{std::string("shoes")}).size(), 8u);
    EXPECT_EQ(c.query(CompareOp::Le, Datum{std::string("shoes")}).size(), 12u);
    EXPECT_EQ(c.query(CompareOp::Gt, Datum{std::string("bags")}).size(), 8u);
    EXPECT_EQ(c.query(CompareOp::Ge, Datum{std::string("hats")}).size(), 8u);
    EXPECT_EQ(c.query_prefix("ba").size(), 4u);
}

TEST(CompositeEncoding, FindHasNothingToPointAtAndSaysSo) {
    std::vector<std::string> values;
    std::vector<ColumnValue> rows;
    categories(&values, &rows);
    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, values, rows);

    // Four rows hold "shoes". Returning one of them would be a guess, so the
    // caller is steered to lookup() by a null rather than by a coin toss.
    EXPECT_EQ(c.find_native(Datum{std::string("shoes")}), nullptr);
    EXPECT_EQ(c.lookup(Datum{std::string("shoes")}).size(), 4u);
}

TEST(CompositeEncoding, BoundsStayExactAtTheExtremeRowIds) {
    // The composite probes use INT64_MIN and INT64_MAX as bounds. A real row
    // sitting exactly on one has to land on the correct side, or every
    // predicate is off by one row for precisely the rows hardest to notice.
    const std::vector<std::string> values = {"v", "v"};
    const std::vector<ColumnValue> rows = {std::numeric_limits<ColumnValue>::min(),
                                           std::numeric_limits<ColumnValue>::max()};
    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, values, rows);

    EXPECT_EQ(c.lookup(Datum{std::string("v")}).size(), 2u);
    EXPECT_EQ(c.query(CompareOp::Le, Datum{std::string("v")}).size(), 2u);
    EXPECT_EQ(c.query(CompareOp::Ge, Datum{std::string("v")}).size(), 2u);
    EXPECT_TRUE(c.query(CompareOp::Gt, Datum{std::string("v")}).empty());
    EXPECT_TRUE(c.query(CompareOp::Lt, Datum{std::string("v")}).empty());
}

TEST(CompositeEncoding, ErasingNeedsTheRowIdBecauseTheKeyContainsIt) {
    std::vector<std::string> values = {"a", "a", "b"};
    ColumnIndex c = ColumnIndex::build_typed(LogicalType::String, values, identity(3));

    EXPECT_THROW(c.erase_row(Datum{std::string("a")}, 0), std::invalid_argument);
    EXPECT_TRUE(c.erase_row(Datum{std::string("a")}, 1));
    EXPECT_EQ(c.lookup(Datum{std::string("a")}), std::vector<ColumnValue>{0});
    EXPECT_NO_THROW(c.validate());
}

TEST(CompositeEncoding, InsertsAndErasesAgreeWithAMultimapOracle) {
    std::multimap<ColumnKey, ColumnValue> oracle;
    std::vector<std::pair<ColumnKey, ColumnValue>> pairs;
    std::mt19937_64 rng(808);
    for (int i = 0; i < 2000; ++i) {
        pairs.emplace_back(static_cast<ColumnKey>(rng() % 30), i);
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> rows;
    for (const auto& [k, v] : pairs) {
        keys.push_back(k);
        rows.push_back(v);
        oracle.emplace(k, v);
    }

    ColumnIndex c = ColumnIndex::build_typed(LogicalType::Int64, keys, rows);
    ASSERT_EQ(c.encoding(), KeyEncoding::Composite);

    // A mixed write stream over the same structure and the same oracle.
    for (int i = 0; i < 800; ++i) {
        if (i % 3 == 0) {
            const ColumnKey k = static_cast<ColumnKey>(rng() % 30);
            const ColumnValue row = 100000 + i;
            c.insert_row(Datum{k}, row);
            oracle.emplace(k, row);
        } else {
            const std::size_t at = rng() % pairs.size();
            const auto& [k, v] = pairs[at];
            const bool removed = c.erase_row(Datum{k}, v);
            const auto range = oracle.equal_range(k);
            bool expected = false;
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == v) { oracle.erase(it); expected = true; break; }
            }
            EXPECT_EQ(removed, expected);
        }
    }
    EXPECT_NO_THROW(c.validate());
    EXPECT_EQ(c.size(), oracle.size());

    for (ColumnKey probe = -2; probe <= 32; ++probe) {
        std::multiset<ColumnValue> want;
        for (const auto& [k, v] : oracle) {
            if (k == probe) want.insert(v);
        }
        const std::vector<ColumnValue> got = c.query(CompareOp::Eq, Datum{probe});
        EXPECT_EQ(std::multiset<ColumnValue>(got.begin(), got.end()), want)
            << "probe " << probe;
    }
}

TEST(CompositeEncoding, EveryOperatorAgreesWithAMultimapOracle) {
    std::mt19937_64 rng(4242);
    std::vector<std::pair<ColumnKey, ColumnValue>> pairs;
    for (int i = 0; i < 4000; ++i) {
        pairs.emplace_back(static_cast<ColumnKey>(rng() % 40), i);
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> rows;
    std::multimap<ColumnKey, ColumnValue> oracle;
    for (const auto& [k, v] : pairs) {
        keys.push_back(k);
        rows.push_back(v);
        oracle.emplace(k, v);
    }

    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::Int64, keys, rows);
    ASSERT_EQ(c.encoding(), KeyEncoding::Composite);

    for (ColumnKey probe = -2; probe <= 42; ++probe) {
        for (CompareOp op : {CompareOp::Eq, CompareOp::Lt, CompareOp::Le,
                             CompareOp::Gt, CompareOp::Ge}) {
            std::multiset<ColumnValue> want;
            for (const auto& [k, v] : oracle) {
                const bool keep = op == CompareOp::Eq   ? k == probe
                                  : op == CompareOp::Lt ? k < probe
                                  : op == CompareOp::Le ? k <= probe
                                  : op == CompareOp::Gt ? k > probe
                                                        : k >= probe;
                if (keep) want.insert(v);
            }
            const std::vector<ColumnValue> got = c.query(op, Datum{probe});
            EXPECT_EQ(std::multiset<ColumnValue>(got.begin(), got.end()), want)
                << "op " << symbol_of(op) << " probe " << probe;
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric types
// ---------------------------------------------------------------------------

TEST(TypedColumns, ADoubleColumnIsExactAndCanCarryALearnedIndex) {
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<double> pick(0.0, 1000.0);
    std::vector<double> keys;
    for (int i = 0; i < 20000; ++i) keys.push_back(pick(rng));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    const std::vector<ColumnValue> rows = identity(keys.size());
    const ColumnIndex c = ColumnIndex::build_typed(LogicalType::Double, keys, rows);

    for (std::size_t i = 0; i < keys.size(); i += 37) {
        const ColumnValue* got = c.find_f64(keys[i]);
        ASSERT_NE(got, nullptr) << "key " << keys[i] << " at " << i;
        EXPECT_EQ(*got, static_cast<ColumnValue>(i));
    }
    EXPECT_EQ(c.query(CompareOp::Lt, Datum{keys[100]}).size(), 100u);
}

TEST(TypedColumns, ATimestampColumnIsOrderedAndDetectedAsMonotone) {
    std::vector<ColumnKey> stamps;
    for (int i = 0; i < 5000; ++i) {
        stamps.push_back(1700000000000LL + i * 60000LL);
    }
    const ColumnIndex c =
        ColumnIndex::build_typed(LogicalType::Timestamp, stamps, identity(5000));

    EXPECT_EQ(c.type(), LogicalType::Timestamp);
    // Monotone is what makes an append-only write path legal, and is the
    // reason Timestamp is its own logical type rather than an alias.
    EXPECT_TRUE(c.plan().monotone);
    EXPECT_EQ(*c.find(stamps[100]), 100);
    EXPECT_EQ(c.range_query(CompareOp::Lt, stamps[50]).size(), 50u);
}

// ---------------------------------------------------------------------------
// The candidate filter
// ---------------------------------------------------------------------------

TEST(CandidateFilter, AStringColumnIsNeverOfferedALearnedIndex) {
    ColumnShape shape;
    shape.rows = 100000;
    shape.distinct = 100000;
    shape.unique = true;

    const auto candidates = candidates_for(LogicalType::String, shape, Workload{});
    ASSERT_FALSE(candidates.empty());
    for (const IndexPlan& p : candidates) EXPECT_EQ(p.kind, IndexKind::BPlusTree);
}

TEST(CandidateFilter, ANumericColumnIsOfferedOneWhenItsValuesAreUnique) {
    ColumnShape shape;
    shape.rows = 100000;
    shape.distinct = 100000;
    shape.unique = true;

    bool saw_rmi = false;
    for (const IndexPlan& p : candidates_for(LogicalType::Int64, shape, Workload{})) {
        saw_rmi |= p.kind == IndexKind::RMI;
    }
    EXPECT_TRUE(saw_rmi);
}

TEST(CandidateFilter, DuplicatesDisqualifyTheLearnedIndexEvenWhenNumeric) {
    // RMIndex::build throws on a repeated key, and a (value, row) pair has no
    // cast to double. So the composite tree is the only structure left, and
    // offering anything else would mean building something that cannot answer.
    ColumnShape shape;
    shape.rows = 100000;
    shape.distinct = 40;
    shape.unique = false;

    const auto candidates = candidates_for(LogicalType::Int64, shape, Workload{});
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates.front().kind, IndexKind::BPlusTree);
    EXPECT_EQ(candidates.front().encoding, KeyEncoding::Composite);
}

TEST(CandidateFilter, AWritableColumnIsStillNeverOfferedTheStaticRmi) {
    ColumnShape shape;
    shape.rows = 100000;
    shape.distinct = 100000;
    shape.unique = true;

    Workload writes;
    writes.write_fraction = 0.3;
    for (const IndexPlan& p : candidates_for(LogicalType::Int64, shape, writes)) {
        EXPECT_NE(p.kind, IndexKind::RMI);
    }
}

TEST(CandidateFilter, ShapeIsMeasuredInOnePassAndRecordedInThePlan) {
    const std::vector<ColumnKey> keys = {1, 1, 2, 3, 3, 3};
    const std::vector<ColumnValue> rows = {0, 1, 2, 3, 4, 5};
    const ColumnShape shape = measure_shape(keys, rows);

    EXPECT_EQ(shape.rows, 6u);
    EXPECT_EQ(shape.distinct, 3u);
    EXPECT_FALSE(shape.unique);
    EXPECT_TRUE(shape.monotone);
    EXPECT_NEAR(shape.duplicate_fraction(), 0.5, 1e-9);

    // Not monotone: the rows did not arrive in key order.
    const std::vector<ColumnValue> shuffled = {5, 0, 2, 1, 4, 3};
    EXPECT_FALSE(measure_shape(keys, shuffled).monotone);
}

// ---------------------------------------------------------------------------
// Accounting and persistence
// ---------------------------------------------------------------------------

TEST(TypedColumns, StringKeysAreChargedForTheirHeapCharacters) {
    // btree.hpp charges capacity() * sizeof(Key) per node, which for a string
    // is the object header and not the characters. Undercounting would make a
    // string column look several times smaller than it is and would feed a
    // wrong number to choose_index's size budget.
    std::vector<std::string> longs;
    std::vector<std::string> shorts;
    for (int i = 0; i < 500; ++i) {
        longs.push_back(std::string(200, 'a') + std::to_string(100000 + i));
        shorts.push_back(std::to_string(100000 + i));
    }
    std::sort(longs.begin(), longs.end());
    std::sort(shorts.begin(), shorts.end());

    const ColumnIndex big =
        ColumnIndex::build_typed(LogicalType::String, longs, identity(500));
    const ColumnIndex small =
        ColumnIndex::build_typed(LogicalType::String, shorts, identity(500));

    EXPECT_GT(big.plan().index_bytes, small.plan().index_bytes + 500 * 200);
}

TEST(TypedColumns, TheTypeAndEncodingSurviveACatalogRoundTrip) {
    IndexPlan plan;
    plan.kind = IndexKind::BPlusTree;
    plan.type = LogicalType::String;
    plan.encoding = KeyEncoding::Composite;
    plan.distinct = 17;
    plan.monotone = true;

    IndexCatalog catalog;
    catalog.set("category", plan);
    const IndexPlan back = *IndexCatalog::parse(catalog.serialize()).get("category");

    // Without these a stored plan records the answer but not the question:
    // "composite" is uninterpretable unless the key type is known.
    EXPECT_EQ(back.type, LogicalType::String);
    EXPECT_EQ(back.encoding, KeyEncoding::Composite);
    EXPECT_EQ(back.distinct, 17u);
    EXPECT_TRUE(back.monotone);
}

TEST(TypedColumns, TheInt64PathIsUnchangedByTypeErasure) {
    // Every measured figure this project has published came through this path.
    // If a refactor moved it, the numbers in plans/important.md would silently
    // stop describing the code.
    std::vector<ColumnKey> keys;
    for (int i = 0; i < 20000; ++i) keys.push_back(i * 3);
    const std::vector<ColumnValue> rows = identity(keys.size());

    const ColumnIndex c = ColumnIndex::build(keys, rows);
    EXPECT_EQ(c.type(), LogicalType::Int64);
    EXPECT_EQ(c.encoding(), KeyEncoding::Native);
    EXPECT_TRUE(c.is_native());
    EXPECT_EQ(*c.find(300), 100);
    EXPECT_EQ(c.find(301), nullptr);
    EXPECT_EQ(c.range(0, 29).size(), 10u);
    EXPECT_EQ(c.range_query(CompareOp::Lt, 30).size(), 10u);
    // The plan still reports a measurement, and it is not absurd.
    EXPECT_GT(c.plan().ns_per_lookup, 0.0);
    EXPECT_LT(c.plan().ns_per_lookup, 10000.0);
}
