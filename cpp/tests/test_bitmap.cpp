// Tests for the dictionary-encoded bitmap index.
//
// Two properties carry everything else here.
//
// **The bitmaps partition the live rows.** A row set in two of them would be
// returned twice by a range query and counted twice by count(), which is the
// failure mode a bitmap index has and an ordered index does not. validate()
// checks it and most of what follows drives writes until it would break.
//
// **The padding bits are not rows.** The last word of an n-bit set has 64 - n%64
// bits that belong to nobody, and a complement sets all of them. Every one
// would then be counted and decoded as a phantom row beyond the end of the
// table, so the complement path is where the arithmetic has to be exactly
// right rather than approximately.

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "index/bitmap.hpp"

using hylis::index::Bitset;
using hylis::index::BitmapIndex;

namespace {

// The worked example from the design notes: twelve rows of an e-commerce
// table, so a reader can check the bitmaps by hand against the prose.
const int kCategoryOf[12] = {2, 0, 2, 1, 0, 2, 1, 2, 0, 1, 2, 0};
const char* kNames[3] = {"bags", "hats", "shoes"};
const int kInStockOf[12] = {1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1};

template <typename T>
void sorted_pairs(const std::vector<T>& values, std::vector<T>* out_values,
                  std::vector<std::int64_t>* out_rows) {
    std::vector<std::pair<T, std::int64_t>> pairs;
    for (std::size_t i = 0; i < values.size(); ++i) {
        pairs.emplace_back(values[i], static_cast<std::int64_t>(i));
    }
    std::sort(pairs.begin(), pairs.end());
    for (const auto& [value, row] : pairs) {
        out_values->push_back(value);
        out_rows->push_back(row);
    }
}

BitmapIndex<std::string> category_index() {
    std::vector<std::string> raw;
    for (int c : kCategoryOf) raw.push_back(kNames[c]);
    std::vector<std::string> values;
    std::vector<std::int64_t> rows;
    sorted_pairs(raw, &values, &rows);

    BitmapIndex<std::string> index;
    index.build(values, rows);
    return index;
}

BitmapIndex<std::int64_t> in_stock_index() {
    std::vector<std::int64_t> raw(kInStockOf, kInStockOf + 12);
    std::vector<std::int64_t> values;
    std::vector<std::int64_t> rows;
    sorted_pairs(raw, &values, &rows);

    BitmapIndex<std::int64_t> index;
    index.build(values, rows);
    return index;
}

}  // namespace

// ---------------------------------------------------------------------------
// Bitset
// ---------------------------------------------------------------------------

TEST(BitsetBasics, SetTestClearAndCount) {
    Bitset b(100);
    EXPECT_EQ(b.size(), 100u);
    EXPECT_EQ(b.count(), 0u);

    b.set(0);
    b.set(63);
    b.set(64);
    b.set(99);
    EXPECT_EQ(b.count(), 4u);
    EXPECT_TRUE(b.test(63));
    EXPECT_TRUE(b.test(64));
    EXPECT_FALSE(b.test(65));

    b.clear(64);
    EXPECT_EQ(b.count(), 3u);
    EXPECT_FALSE(b.test(64));
}

TEST(BitsetBasics, ForEachYieldsAscendingPositions) {
    Bitset b(200);
    const std::vector<std::size_t> want = {0, 1, 63, 64, 65, 128, 199};
    for (std::size_t i : want) b.set(i);

    std::vector<std::size_t> got;
    b.for_each([&](std::size_t i) { got.push_back(i); });
    EXPECT_EQ(got, want);
}

TEST(BitsetBasics, ComplementDoesNotInventRowsPastTheEnd) {
    // 100 bits occupy two 64-bit words, so 28 bits of the second word belong
    // to no row. A complement sets them, and without masking each would be
    // counted and decoded as a row beyond the end of the table.
    Bitset b(100);
    b.set(5);
    b.flip();
    EXPECT_EQ(b.count(), 99u);
    EXPECT_FALSE(b.test(5));

    std::size_t highest = 0;
    b.for_each([&](std::size_t i) { highest = std::max(highest, i); });
    EXPECT_LT(highest, 100u);
}

TEST(BitsetBasics, ComplementIsExactAtAWordBoundary) {
    // 128 bits fill two words exactly, so there is no padding to mask. The
    // boundary case where the masking code must do nothing.
    Bitset b(128);
    b.set(0);
    b.flip();
    EXPECT_EQ(b.count(), 127u);
}

TEST(BitsetBasics, AndOrAgreeWithSetOperations) {
    Bitset a(70);
    Bitset b(70);
    for (std::size_t i : {1u, 5u, 64u, 69u}) a.set(i);
    for (std::size_t i : {5u, 6u, 69u}) b.set(i);

    Bitset both = a;
    both &= b;
    std::vector<std::size_t> got;
    both.for_each([&](std::size_t i) { got.push_back(i); });
    EXPECT_EQ(got, (std::vector<std::size_t>{5, 69}));

    Bitset either = a;
    either |= b;
    got.clear();
    either.for_each([&](std::size_t i) { got.push_back(i); });
    EXPECT_EQ(got, (std::vector<std::size_t>{1, 5, 6, 64, 69}));
}

TEST(BitsetBasics, CombiningDifferentWidthsIsRefused) {
    // Two columns whose bitmaps cover different row sets would produce an
    // answer that looks plausible and matches neither predicate.
    Bitset a(64);
    Bitset b(65);
    EXPECT_THROW(a &= b, std::invalid_argument);
    EXPECT_THROW(a |= b, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// The worked example
// ---------------------------------------------------------------------------

TEST(BitmapExample, EqualityMatchesTheHandCountedTable) {
    const auto index = category_index();
    EXPECT_EQ(index.distinct(), 3u);
    EXPECT_EQ(index.rows(), 12u);

    EXPECT_EQ(index.equal("bags").count(), 4u);
    EXPECT_EQ(index.equal("hats").count(), 3u);
    EXPECT_EQ(index.equal("shoes").count(), 5u);
    EXPECT_EQ(index.equal("socks").count(), 0u);

    EXPECT_EQ(index.decode(index.equal("shoes")),
              (std::vector<std::int64_t>{0, 2, 5, 7, 10}));
    EXPECT_EQ(index.decode(index.equal("hats")),
              (std::vector<std::int64_t>{3, 6, 9}));
}

TEST(BitmapExample, ARangeIsAContiguousRunOfCodes) {
    const auto index = category_index();
    // category < 'shoes' is codes [0, 2): bags OR hats.
    const Bitset below = index.run(0, index.lower_code("shoes"));
    EXPECT_EQ(below.count(), 7u);
    EXPECT_EQ(index.decode(below),
              (std::vector<std::int64_t>{1, 3, 4, 6, 8, 9, 11}));
    // Complement check: 12 rows minus 5 shoes.
    EXPECT_EQ(below.count(), index.rows() - index.equal("shoes").count());
}

TEST(BitmapExample, ConjunctionIsAWordParallelAnd) {
    const auto categories = category_index();
    const auto in_stock = in_stock_index();

    Bitset both = categories.equal("shoes");
    both &= in_stock.equal(1);

    EXPECT_EQ(both.count(), 3u);
    EXPECT_EQ(categories.decode(both), (std::vector<std::int64_t>{0, 2, 7}));
}

TEST(BitmapExample, DecodingYieldsAscendingRecordKeys) {
    // The row table is kept sorted precisely so this needs no sort afterwards,
    // which is what lets Table return ordered results without paying for one.
    const auto index = category_index();
    for (const char* name : kNames) {
        const auto rows = index.decode(index.equal(name));
        EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end())) << name;
    }
}

// ---------------------------------------------------------------------------
// The row table
// ---------------------------------------------------------------------------

TEST(BitmapRows, SparseRecordKeysDecodeBackToThemselves) {
    // Record keys are arbitrary int64, not dense positions. Getting the table
    // wrong would return positions rather than keys, which look like rows.
    const std::vector<std::string> values = {"a", "a", "b", "b"};
    const std::vector<std::int64_t> rows = {5, 900, 17, 100000};
    std::vector<std::pair<std::string, std::int64_t>> pairs;
    for (std::size_t i = 0; i < values.size(); ++i) pairs.emplace_back(values[i], rows[i]);
    std::sort(pairs.begin(), pairs.end());

    std::vector<std::string> v;
    std::vector<std::int64_t> r;
    for (const auto& [value, row] : pairs) { v.push_back(value); r.push_back(row); }

    BitmapIndex<std::string> index;
    index.build(v, r);
    EXPECT_EQ(index.decode(index.equal("a")), (std::vector<std::int64_t>{5, 900}));
    EXPECT_EQ(index.decode(index.equal("b")), (std::vector<std::int64_t>{17, 100000}));
    EXPECT_NO_THROW(index.validate());
}

TEST(BitmapRows, ARowSpaceLargerThanTheValuesLeavesTheExtraRowsMatchingNothing) {
    // Two bitmaps can only be combined if position i means the same row in
    // both, so a column present on half the table still needs a position for
    // every row. Those rows are in no bitmap, which is exactly "absent matches
    // no predicate".
    const std::vector<std::string> values = {"a", "b"};
    const std::vector<std::int64_t> rows = {1, 3};
    const std::vector<std::int64_t> space = {0, 1, 2, 3, 4};

    BitmapIndex<std::string> index;
    index.build(values, rows, &space);

    EXPECT_EQ(index.rows(), 5u);
    EXPECT_EQ(index.size(), 2u);
    EXPECT_EQ(index.decode(index.equal("a")), std::vector<std::int64_t>{1});
    // Everything, over the whole dictionary, is still only the two live rows.
    EXPECT_EQ(index.run(0, index.distinct()).count(), 2u);
    EXPECT_NO_THROW(index.validate());
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

TEST(BitmapWrites, AssignMovesARowBetweenValues) {
    auto index = category_index();
    ASSERT_TRUE(index.assign("hats", 0));  // row 0 was shoes

    EXPECT_EQ(index.equal("shoes").count(), 4u);
    EXPECT_EQ(index.equal("hats").count(), 4u);
    EXPECT_EQ(index.size(), 12u);
    EXPECT_NO_THROW(index.validate());
}

TEST(BitmapWrites, AssignCanIntroduceANewValue) {
    auto index = category_index();
    ASSERT_TRUE(index.assign("coats", 3));

    EXPECT_EQ(index.distinct(), 4u);
    EXPECT_EQ(index.equal("coats").count(), 1u);
    EXPECT_EQ(index.equal("hats").count(), 2u);
    // The dictionary stays sorted, so ranges stay contiguous runs: "coats"
    // sorts first, and every other code shifted up by one.
    EXPECT_EQ(index.dictionary().front(), "bags");
    EXPECT_EQ(index.run(0, index.lower_code("hats")).count(), 5u);
    EXPECT_NO_THROW(index.validate());
}

TEST(BitmapWrites, DeleteIsExactAndNeedsNoTombstone) {
    auto index = category_index();
    ASSERT_TRUE(index.erase(0));
    EXPECT_EQ(index.size(), 11u);
    EXPECT_EQ(index.equal("shoes").count(), 4u);
    EXPECT_FALSE(index.contains_row(0));
    EXPECT_FALSE(index.erase(0));  // already gone
    EXPECT_NO_THROW(index.validate());
}

TEST(BitmapWrites, AComplementAfterADeleteDoesNotResurrectTheRow) {
    // The complement path builds "everything else and invert", and a deleted
    // row belongs to no value — so without the live mask it would come back
    // as a member of every range wide enough to take the complement.
    auto index = category_index();
    ASSERT_TRUE(index.erase(4));

    const Bitset everything = index.run(0, index.distinct());
    EXPECT_EQ(everything.count(), index.size());
    const auto rows = index.decode(everything);
    EXPECT_EQ(std::find(rows.begin(), rows.end(), 4), rows.end());
    EXPECT_NO_THROW(index.validate());
}

TEST(BitmapWrites, AppendTakesAKeyAfterTheLastAndRefusesOneBefore) {
    auto index = category_index();
    EXPECT_TRUE(index.append("socks", 12));
    EXPECT_EQ(index.rows(), 13u);
    EXPECT_EQ(index.equal("socks").count(), 1u);

    // A key landing mid-table would shift every position after it, which
    // invalidates every bitmap at once. Refusing sends the caller to a rebuild.
    EXPECT_FALSE(index.append("bags", 5));
    EXPECT_FALSE(index.append("bags", 12));  // not strictly after
    EXPECT_NO_THROW(index.validate());
}

TEST(BitmapWrites, AppendingPastTheIdentityShortcutStillWorks) {
    // Rows 0..n-1 skip the row table entirely. Appending a key that breaks
    // that pattern has to materialise the table first, or every later lookup
    // would resolve to the wrong position.
    std::vector<std::int64_t> values = {0, 0, 1, 1};
    std::vector<std::int64_t> rows = {0, 1, 2, 3};
    BitmapIndex<std::int64_t> index;
    index.build(values, rows);

    ASSERT_TRUE(index.append(1, 500));
    EXPECT_EQ(index.decode(index.equal(1)), (std::vector<std::int64_t>{2, 3, 500}));
    ASSERT_TRUE(index.append(0, 900));
    EXPECT_EQ(index.decode(index.equal(0)), (std::vector<std::int64_t>{0, 1, 900}));
    EXPECT_NO_THROW(index.validate());
}

// ---------------------------------------------------------------------------
// Differential
// ---------------------------------------------------------------------------

TEST(BitmapDifferential, EveryQueryAgreesWithAMapOracleThroughAWriteStream) {
    std::mt19937_64 rng(90210);
    std::map<std::int64_t, std::int64_t> oracle;  // row -> value

    std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
    for (std::int64_t row = 0; row < 3000; ++row) {
        const std::int64_t value = static_cast<std::int64_t>(rng() % 12);
        pairs.emplace_back(value, row);
        oracle[row] = value;
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<std::int64_t> values;
    std::vector<std::int64_t> rows;
    for (const auto& [value, row] : pairs) {
        values.push_back(value);
        rows.push_back(row);
    }

    BitmapIndex<std::int64_t> index;
    index.build(values, rows);

    // A mixed stream of moves and deletes over the same oracle.
    for (int step = 0; step < 2000; ++step) {
        const std::int64_t row = static_cast<std::int64_t>(rng() % 3000);
        if (step % 3 == 0) {
            if (index.erase(row)) oracle.erase(row);
        } else {
            const std::int64_t value = static_cast<std::int64_t>(rng() % 12);
            if (index.assign(value, row)) oracle[row] = value;
        }
    }
    ASSERT_NO_THROW(index.validate());
    EXPECT_EQ(index.size(), oracle.size());

    for (std::int64_t probe = -1; probe <= 13; ++probe) {
        std::vector<std::int64_t> want_eq;
        std::vector<std::int64_t> want_lt;
        std::vector<std::int64_t> want_ge;
        for (const auto& [row, value] : oracle) {
            if (value == probe) want_eq.push_back(row);
            if (value < probe) want_lt.push_back(row);
            if (value >= probe) want_ge.push_back(row);
        }

        EXPECT_EQ(index.decode(index.equal(probe)), want_eq) << "== " << probe;
        EXPECT_EQ(index.decode(index.run(0, index.lower_code(probe))), want_lt)
            << "< " << probe;
        EXPECT_EQ(index.decode(index.run(index.lower_code(probe), index.distinct())),
                  want_ge)
            << ">= " << probe;

        // count() must agree with the list it declines to build.
        EXPECT_EQ(index.equal(probe).count(), want_eq.size());
        EXPECT_EQ(index.run(0, index.lower_code(probe)).count(), want_lt.size());
    }
}

TEST(BitmapDifferential, MemoryGrowsWithTheDistinctCount) {
    // The bound the whole family lives under: linear in n, and linear in d.
    // If this ever stops holding, the cardinality threshold is measuring
    // something other than what it thinks.
    const std::size_t n = 4096;
    std::size_t previous = 0;
    for (std::size_t distinct : {2u, 8u, 64u, 512u}) {
        std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
        for (std::size_t row = 0; row < n; ++row) {
            pairs.emplace_back(static_cast<std::int64_t>(row % distinct),
                               static_cast<std::int64_t>(row));
        }
        std::sort(pairs.begin(), pairs.end());

        std::vector<std::int64_t> values;
        std::vector<std::int64_t> rows;
        for (const auto& [value, row] : pairs) {
            values.push_back(value);
            rows.push_back(row);
        }
        BitmapIndex<std::int64_t> index;
        index.build(values, rows);

        const std::size_t bytes = index.memory_bytes();
        EXPECT_GT(bytes, previous) << "distinct = " << distinct;
        // Roughly d * n / 8, and never wildly under it.
        EXPECT_GE(bytes, distinct * n / 8);
        previous = bytes;
    }
}
