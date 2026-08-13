// Tests for Table: the join between the record store and the indexes.
//
// The claim under test is that an index answers exactly what a full scan of
// the store would answer. Most of what follows is that claim in various forms,
// because every other property of this layer is worthless without it: a
// predicate served by a B+ tree, by a learned index, or by no index at all
// must return the same rows, or choosing between them stops being a
// performance decision and becomes a semantic one.
//
// The second theme is the write path. Maintenance is exact and incremental for
// every mutable structure — composite keys are (value, record key), so
// changing a row's value is an erase and an insert with nothing renumbered.
// The lazy-rebuild machinery survives for exactly two cases, and the tests
// name both rather than leaving them to be inferred from a counter.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "query/table.hpp"

using hylis::index::Datum;
using hylis::index::IndexCatalog;
using hylis::index::IndexKind;
using hylis::index::KeyEncoding;
using hylis::index::LogicalType;
using hylis::index::Workload;
using hylis::query::ColumnDef;
using hylis::query::PredOp;
using hylis::query::Predicate;
using hylis::query::QueryTrace;
using hylis::query::Schema;
using hylis::query::Table;
using hylis::storage::Record;
using hylis::storage::RecordStore;

namespace {

namespace fs = std::filesystem;

// A per-test directory under the build tree, removed on construction so a
// previous run's WAL cannot leak into this one.
class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_(fs::path(HYLIS_TEST_TMP) / ("table_" + name)) {
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::string str() const { return path_.string(); }

private:
    fs::path path_;
};

Schema shop_schema() {
    Schema s;
    s.add(ColumnDef("price", LogicalType::Int64));
    s.add(ColumnDef("category", LogicalType::String));
    s.add(ColumnDef("title", LogicalType::String));
    s.add(ColumnDef("created_at", LogicalType::Timestamp));
    return s;
}

const char* kCategories[] = {"bags", "hats", "shoes"};
const char* kTitles[] = {"nike air", "nike zoom", "adidas run", "puma go",
                         "nikon lens"};

// Every tenth row has no timestamp, so the absent-value path is exercised by
// the ordinary fixture rather than only by its own test.
std::vector<Record> shop_rows(int n) {
    std::vector<Record> rows;
    for (int i = 0; i < n; ++i) {
        Record r(i);
        r.columns["price"] = std::to_string((i * 37) % 500);
        r.columns["category"] = kCategories[i % 3];
        r.columns["title"] = kTitles[i % 5];
        if (i % 10 != 0) {
            r.columns["created_at"] =
                "2026-01-" + std::string(i % 28 < 9 ? "0" : "") +
                std::to_string(i % 28 + 1) + "T12:00:00Z";
        }
        rows.push_back(r);
    }
    return rows;
}

std::set<std::int64_t> keys_of(const std::vector<std::int64_t>& v) {
    return std::set<std::int64_t>(v.begin(), v.end());
}

// The oracle every query test is graded against: a full scan of the store,
// evaluating the predicate in Python-free C++ with no index involved.
template <typename Fn>
std::set<std::int64_t> scan_oracle(const RecordStore& store,
                                   const std::string& column, bool want_absent,
                                   Fn keep) {
    std::set<std::int64_t> out;
    for (const Record& r : store.records()) {
        const auto it = r.columns.find(column);
        if (it == r.columns.end()) {
            if (want_absent) out.insert(r.key);
            continue;
        }
        if (!want_absent && keep(it->second)) out.insert(r.key);
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

TEST(TableBasics, IndexesEveryScalarTypeAndReportsWhatItChose) {
    TempDir dir("build_all");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(300));

    for (const char* name : {"price", "category", "title", "created_at"}) {
        EXPECT_NO_THROW(table.create_index(name)) << name;
        EXPECT_TRUE(table.has_index(name)) << name;
    }

    const auto info = table.info("category");
    EXPECT_TRUE(info.indexed);
    EXPECT_EQ(info.type, LogicalType::String);
    EXPECT_EQ(info.distinct, 3u);
    EXPECT_FALSE(info.unique);
    // Three distinct values over 300 rows: duplicated, so composite.
    EXPECT_EQ(info.encoding, KeyEncoding::Composite);
    EXPECT_EQ(info.rows, 300u);
    EXPECT_EQ(info.skipped, 0u);

    // Every tenth row has no timestamp, so it is in no index and answers no
    // predicate on that column.
    const auto stamped = table.info("created_at");
    EXPECT_EQ(stamped.rows, 270u);
    EXPECT_EQ(stamped.skipped, 30u);

    EXPECT_NO_THROW(table.validate());
}

TEST(TableBasics, ADescribeRowExistsForEveryDeclaredColumn) {
    TempDir dir("describe");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(50));
    table.create_index("price");

    const auto described = table.describe();
    ASSERT_EQ(described.size(), 4u);
    std::size_t indexed = 0;
    for (const auto& info : described) indexed += info.indexed;
    EXPECT_EQ(indexed, 1u) << "only the column with an index should say so";
}

TEST(TableBasics, BoolAndVectorColumnsAreRefusedWithTheReason) {
    TempDir dir("refuse");
    RecordStore store(dir.str());
    Schema s = shop_schema();
    s.add(ColumnDef("in_stock", LogicalType::Bool));
    s.add(ColumnDef("embedding", LogicalType::Vector, 8));
    Table table(store, std::move(s));

    EXPECT_THROW(table.create_index("in_stock"), std::invalid_argument);
    EXPECT_THROW(table.create_index("embedding"), std::invalid_argument);
}

TEST(TableBasics, AnIllTypedRecordIsRefusedBeforeAnythingIsWritten) {
    TempDir dir("reject");
    RecordStore store(dir.str());
    Table table(store, shop_schema());

    EXPECT_THROW(table.put(Record(1, {{"price", "abc"}})), std::invalid_argument);
    EXPECT_THROW(table.put(Record(2, {{"pirce", "40"}})), std::invalid_argument);
    // Neither reached the store: a record that would half-load is refused.
    EXPECT_EQ(table.size(), 0u);
    EXPECT_FALSE(table.get(1).has_value());
}

// ---------------------------------------------------------------------------
// Queries: an index must answer what a scan would
// ---------------------------------------------------------------------------

TEST(TableQueries, EveryOperatorAgreesWithAFullScan) {
    TempDir dir("agree");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(400));
    table.create_index("price");
    table.create_index("category");
    table.create_index("title");

    QueryTrace trace;

    for (std::int64_t probe : {-1, 0, 37, 250, 499, 500, 9999}) {
        const auto want_lt = scan_oracle(store, "price", false,
                                         [&](const std::string& s) {
                                             return std::stoll(s) < probe;
                                         });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("price", PredOp::Lt, Datum{probe}), &trace)),
                  want_lt)
            << "price < " << probe;
        EXPECT_TRUE(trace.used_index);

        const auto want_ge = scan_oracle(store, "price", false,
                                         [&](const std::string& s) {
                                             return std::stoll(s) >= probe;
                                         });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("price", PredOp::Ge, Datum{probe}))),
                  want_ge)
            << "price >= " << probe;

        const auto want_eq = scan_oracle(store, "price", false,
                                         [&](const std::string& s) {
                                             return std::stoll(s) == probe;
                                         });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("price", PredOp::Eq, Datum{probe}))),
                  want_eq)
            << "price == " << probe;
    }

    for (const char* category : {"bags", "hats", "shoes", "socks"}) {
        const std::string value = category;
        const auto want = scan_oracle(store, "category", false,
                                      [&](const std::string& s) {
                                          return s == value;
                                      });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("category", PredOp::Eq, Datum{value}))),
                  want)
            << "category == " << category;
    }
}

TEST(TableQueries, BetweenIsInclusiveAtBothEnds) {
    TempDir dir("between");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(400));
    table.create_index("price");

    const auto want = scan_oracle(store, "price", false,
                                  [](const std::string& s) {
                                      const long long v = std::stoll(s);
                                      return v >= 100 && v <= 200;
                                  });
    EXPECT_EQ(keys_of(table.select_keys(Predicate(
                  "price", PredOp::Between, Datum{std::int64_t{100}},
                  Datum{std::int64_t{200}}))),
              want);
}

TEST(TableQueries, PrefixIsServedByTheIndexAndMatchesAScan) {
    TempDir dir("prefix");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(300));
    table.create_index("title");

    QueryTrace trace;
    for (const std::string prefix : {"nike", "nik", "ni", "adidas", "z", ""}) {
        const auto want = scan_oracle(store, "title", false,
                                      [&](const std::string& s) {
                                          return s.rfind(prefix, 0) == 0;
                                      });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("title", PredOp::Prefix, Datum{prefix}), &trace)),
                  want)
            << "prefix '" << prefix << "'";
        EXPECT_TRUE(trace.used_index) << "prefix '" << prefix << "'";
    }

    // "nike air" and "nike zoom" begin with "nike"; "nikon lens" does not.
    EXPECT_LT(table.count(Predicate("title", PredOp::Prefix, Datum{std::string("nike")})),
              table.count(Predicate("title", PredOp::Prefix, Datum{std::string("nik")})));
}

TEST(TableQueries, ContainsAndIsNullScanAndSayThatTheyDid) {
    TempDir dir("scan_ops");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(300));
    table.create_index("title");
    table.create_index("created_at");

    // An infix match has no ordering to exploit, so no index here can serve
    // it. Answering it anyway, and saying it was a scan, beats refusing.
    QueryTrace trace;
    const auto want_contains =
        scan_oracle(store, "title", false, [](const std::string& s) {
            return s.find("ike") != std::string::npos;
        });
    EXPECT_EQ(keys_of(table.select_keys(
                  Predicate("title", PredOp::Contains, Datum{std::string("ike")}),
                  &trace)),
              want_contains);
    EXPECT_FALSE(trace.used_index);
    EXPECT_EQ(trace.scanned, store.size());
    EXPECT_NE(trace.reason.find("scanning"), std::string::npos);

    // An absent value is absent from the index by construction, so IsNull is
    // the one predicate whose answer no index can hold.
    const auto want_null = scan_oracle(store, "created_at", true,
                                       [](const std::string&) { return false; });
    EXPECT_EQ(keys_of(table.select_keys(
                  Predicate("created_at", PredOp::IsNull, Datum{}), &trace)),
              want_null);
    EXPECT_EQ(want_null.size(), 30u);
    EXPECT_FALSE(trace.used_index);
}

TEST(TableQueries, AnUnindexedColumnStillAnswersCorrectly) {
    TempDir dir("no_index");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(200));
    // No create_index at all.

    QueryTrace trace;
    const auto want = scan_oracle(store, "price", false,
                                  [](const std::string& s) {
                                      return std::stoll(s) < 250;
                                  });
    EXPECT_EQ(keys_of(table.select_keys(
                  Predicate("price", PredOp::Lt, Datum{std::int64_t{250}}), &trace)),
              want);
    EXPECT_FALSE(trace.used_index);
    EXPECT_EQ(trace.scanned, 200u);

    // Building the index must not change the answer, only the cost. This is
    // the property that makes index choice a performance decision.
    table.create_index("price");
    EXPECT_EQ(keys_of(table.select_keys(
                  Predicate("price", PredOp::Lt, Datum{std::int64_t{250}}), &trace)),
              want);
    EXPECT_TRUE(trace.used_index);
}

TEST(TableQueries, ResultsComeBackSortedByRecordKey) {
    // An index returns rows in the column's order and a scan returns them in
    // the store's hash order. Without sorting, the ordering would depend on
    // which structure happened to answer.
    TempDir dir("sorted");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(120));
    table.create_index("category");

    const auto indexed =
        table.select_keys(Predicate("category", PredOp::Eq, Datum{std::string("shoes")}));
    EXPECT_TRUE(std::is_sorted(indexed.begin(), indexed.end()));

    const auto scanned = table.select_keys(
        Predicate("title", PredOp::Contains, Datum{std::string("nike")}));
    EXPECT_TRUE(std::is_sorted(scanned.begin(), scanned.end()));
}

TEST(TableQueries, SelectReturnsTheRecordsTheStoreHolds) {
    TempDir dir("records");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(100));
    table.create_index("category");

    const auto rows =
        table.select(Predicate("category", PredOp::Eq, Datum{std::string("hats")}));
    ASSERT_FALSE(rows.empty());
    for (const Record& r : rows) {
        EXPECT_EQ(r.get("category"), "hats");
        const auto stored = table.get(r.key);
        ASSERT_TRUE(stored.has_value());
        EXPECT_EQ(stored->columns, r.columns);
    }
}

TEST(TableQueries, AStringOperatorOnANumericColumnIsRefused) {
    TempDir dir("wrong_op");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(20));
    table.create_index("price");

    EXPECT_THROW(table.select_keys(
                     Predicate("price", PredOp::Prefix, Datum{std::string("1")})),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// The write path
// ---------------------------------------------------------------------------

TEST(TableWrites, AnUpdateMovesTheRowInEveryIndexItTouches) {
    TempDir dir("update");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(200));
    // Declared writable, so the selector may not hand these columns the static
    // RMI. Asking for a read-only structure and then writing to it is a legal
    // thing to do -- it costs a rebuild -- but it is a different test, and
    // leaving it to chance here would make this one flaky on the timing.
    table.create_index("price", Workload{0.3});
    table.create_index("category", Workload{0.3});

    Record moved(7);
    moved.columns["price"] = "9999";
    moved.columns["category"] = "shoes";
    moved.columns["title"] = "nike air";

    const auto result = table.put(moved);
    EXPECT_FALSE(result.created);
    EXPECT_GT(result.indexes_touched, 0u);
    EXPECT_EQ(result.rebuilds_triggered, 0u)
        << "a composite key takes a mid-range write in O(log n); nothing "
           "should have needed rebuilding";

    EXPECT_EQ(table.select_keys(Predicate("price", PredOp::Eq,
                                          Datum{std::int64_t{9999}})),
              std::vector<std::int64_t>{7});
    // The old value must no longer name row 7.
    const auto old_price =
        table.select_keys(Predicate("price", PredOp::Eq, Datum{std::int64_t{7 * 37}}));
    EXPECT_EQ(std::find(old_price.begin(), old_price.end(), 7), old_price.end());
    EXPECT_NO_THROW(table.validate());
}

TEST(TableWrites, AnUnchangedColumnIsNotTouched) {
    TempDir dir("unchanged");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(100));
    table.create_index("price", Workload{0.3});
    table.create_index("category", Workload{0.3});

    // Rewrite row 5 with only `title` different.
    Record same = *table.get(5);
    same.columns["title"] = "puma go";
    const auto result = table.put(same);

    // price and category are unchanged, so neither index should have been
    // erased-and-reinserted for no reason.
    EXPECT_EQ(result.indexes_touched, 0u);
    EXPECT_NO_THROW(table.validate());
}

TEST(TableWrites, ErasingRemovesTheRowFromEveryIndex) {
    TempDir dir("erase");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(150));
    table.create_index("price", Workload{0.3});
    table.create_index("category", Workload{0.3});

    const auto result = table.erase(3);
    EXPECT_GT(result.indexes_touched, 0u);
    EXPECT_EQ(result.rebuilds_triggered, 0u)
        << "a delete is exact in every structure here and never rebuilds";
    EXPECT_FALSE(table.get(3).has_value());

    for (const auto& keys : {table.select_keys(Predicate(
                                 "category", PredOp::Eq, Datum{std::string("bags")})),
                             table.select_keys(Predicate(
                                 "price", PredOp::Ge, Datum{std::int64_t{0}}))}) {
        EXPECT_EQ(std::find(keys.begin(), keys.end(), 3), keys.end());
    }
    EXPECT_NO_THROW(table.validate());
}

TEST(TableWrites, ErasingAnAbsentRowIsANoOp) {
    TempDir dir("erase_missing");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(20));
    table.create_index("price");

    const auto result = table.erase(9999);
    EXPECT_EQ(result.indexes_touched, 0u);
    EXPECT_EQ(table.size(), 20u);
}

TEST(TableWrites, AStaticRmiColumnRebuildsRatherThanThrowing) {
    // The first of the two surviving rebuild cases. A static RMI is
    // build-only: it was a legal choice under a read-only workload and stopped
    // being one the moment a write arrived. ColumnIndex::insert throws by
    // design rather than returning false, and Table must never let that reach
    // the caller.
    TempDir dir("static_rmi");
    RecordStore store(dir.str());
    Table table(store, shop_schema());

    std::vector<Record> rows;
    for (int i = 0; i < 3000; ++i) {
        Record r(i);
        r.columns["price"] = std::to_string(i * 7);  // unique and ascending
        rows.push_back(r);
    }
    table.put_batch(rows);
    table.create_index("price", Workload{});  // read-only: the RMI is legal

    if (table.info("price").kind != IndexKind::RMI) {
        GTEST_SKIP() << "the tree won this column on this machine; the case "
                        "under test needs the immutable structure";
    }

    Record extra(99999);
    extra.columns["price"] = "123456";
    const auto result = table.put(extra);
    EXPECT_EQ(result.rebuilds_triggered, 1u);

    // The rebuild happens on the next read, and the answer is correct.
    EXPECT_EQ(table.select_keys(Predicate("price", PredOp::Eq,
                                          Datum{std::int64_t{123456}})),
              std::vector<std::int64_t>{99999});
    EXPECT_EQ(table.rebuilds(), 1u);
    EXPECT_NO_THROW(table.validate());
}

TEST(TableWrites, ANativelyKeyedColumnRebuildsWhenAWriteMakesItNonUnique) {
    // The second surviving rebuild case, and the subtle one. A native key was
    // chosen because the values were unique; an insert that collides makes
    // them not, and the tree overwrote the colliding row's entry rather than
    // storing both. Without the rebuild the index would silently be missing a
    // row.
    TempDir dir("native_collide");
    RecordStore store(dir.str());
    Table table(store, shop_schema());

    std::vector<Record> rows;
    for (int i = 0; i < 40; ++i) {
        Record r(i);
        r.columns["category"] = "c" + std::to_string(i);  // all distinct
        rows.push_back(r);
    }
    table.put_batch(rows);
    table.create_index("category", Workload{0.5});
    ASSERT_EQ(table.info("category").encoding, KeyEncoding::Native);

    Record collide(1000);
    collide.columns["category"] = "c7";  // now two rows share a value
    const auto result = table.put(collide);
    EXPECT_EQ(result.rebuilds_triggered, 1u);

    // After the rebuild the column is composite and both rows are findable.
    const auto rows_with_c7 =
        table.select_keys(Predicate("category", PredOp::Eq, Datum{std::string("c7")}));
    EXPECT_EQ(rows_with_c7, (std::vector<std::int64_t>{7, 1000}));
    EXPECT_EQ(table.info("category").encoding, KeyEncoding::Composite);
    EXPECT_NO_THROW(table.validate());
}

TEST(TableWrites, ABatchPaysOneRebuildWhereALoopPaysOnePerRecord) {
    // Not an optimisation, the only usable bulk-load path.
    //
    // The workload matters, and an earlier version of this test got it wrong.
    // It used a uniqueness collision, asserted the batch paid one rebuild, and
    // passed — but a *loop* also pays exactly one there, because the first
    // rebuild switches the column to a composite key which absorbs every later
    // collision. The assertion was true and vacuous, and only the E6 benchmark
    // noticed.
    //
    // The case where batching actually matters is a column planned read-only
    // and then written to: its structure is build-only, so the rebuild
    // produces the same build-only structure and the next write dirties it
    // again. So the two paths are compared directly rather than one of them
    // being measured against a number.
    const auto build = [](const std::string& name, Table* table) {
        std::vector<Record> rows;
        for (int i = 0; i < 3000; ++i) {
            Record r(i);
            r.columns["price"] = std::to_string(i * 7);  // unique, ascending
            rows.push_back(r);
        }
        table->put_batch(rows);
        table->create_index("price", Workload{});  // read-only: the RMI is legal
        (void)name;
    };

    std::vector<Record> extra;
    for (int i = 0; i < 15; ++i) {
        Record r(100000 + i);
        r.columns["price"] = std::to_string(1000000 + i * 7);
        extra.push_back(r);
    }

    TempDir loop_dir("batch_loop");
    RecordStore loop_store(loop_dir.str());
    Table looped(loop_store, shop_schema());
    build("price", &looped);
    if (looped.info("price").kind != IndexKind::RMI) {
        GTEST_SKIP() << "the tree won this column on this machine, so no arm "
                        "has a build-only structure and there is no rebuild "
                        "to batch away";
    }
    for (const Record& r : extra) looped.put(r);

    TempDir batch_dir("batch_batch");
    RecordStore batch_store(batch_dir.str());
    Table batched(batch_store, shop_schema());
    build("price", &batched);
    const auto result = batched.put_batch(extra);

    EXPECT_EQ(result.rows_created, extra.size());
    EXPECT_EQ(batched.rebuilds(), 1u) << "a batch rebuilds once, at the end";
    EXPECT_GT(looped.rebuilds(), 1u)
        << "a loop over the same records must rebuild per record; if it does "
           "not, put_batch's docstring in table.hpp overstates its value";
    EXPECT_NO_THROW(batched.validate());
    EXPECT_NO_THROW(looped.validate());
}

TEST(TableWrites, UpdateChangesOnlyTheNamedColumns) {
    TempDir dir("partial");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(50));
    table.create_index("price", Workload{0.3});

    const std::string title_before = table.get(4)->get("title");
    table.update(4, {{"price", Datum{std::int64_t{777}}}});

    const auto after = table.get(4);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(after->get("price"), "777");
    EXPECT_EQ(after->get("title"), title_before);
    EXPECT_EQ(table.select_keys(Predicate("price", PredOp::Eq, Datum{std::int64_t{777}})),
              std::vector<std::int64_t>{4});
}

TEST(TableWrites, UpdatingAnAbsentRowIsAnError) {
    TempDir dir("update_missing");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    EXPECT_THROW(table.update(1, {{"price", Datum{std::int64_t{1}}}}),
                 std::invalid_argument);
}

TEST(TableWrites, ARandomisedWorkloadStaysInAgreementWithTheStore) {
    // The assertion that catches every maintenance bug at once, in the pattern
    // the B+ tree and dynamic RMI suites already use: a long mixed stream of
    // puts, updates and erases, with the index checked against a full scan of
    // the store at the end.
    TempDir dir("fuzz");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(300));
    table.create_index("price", Workload{0.3});
    table.create_index("category", Workload{0.3});

    std::mt19937_64 rng(20260813);
    for (int step = 0; step < 1500; ++step) {
        const int action = static_cast<int>(rng() % 3);
        const std::int64_t key = static_cast<std::int64_t>(rng() % 400);
        if (action == 0) {
            Record r(key);
            r.columns["price"] = std::to_string(rng() % 600);
            r.columns["category"] = kCategories[rng() % 3];
            r.columns["title"] = kTitles[rng() % 5];
            table.put(r);
        } else if (action == 1) {
            table.erase(key);
        } else if (table.get(key).has_value()) {
            table.update(key, {{"price", Datum{static_cast<std::int64_t>(rng() % 600)}}});
        }
    }

    EXPECT_NO_THROW(table.validate());
    for (std::int64_t probe : {0, 150, 300, 599, 600}) {
        const auto want = scan_oracle(store, "price", false,
                                      [&](const std::string& s) {
                                          return std::stoll(s) < probe;
                                      });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("price", PredOp::Lt, Datum{probe}))),
                  want)
            << "price < " << probe;
    }
    for (const char* category : kCategories) {
        const std::string value = category;
        const auto want = scan_oracle(store, "category", false,
                                      [&](const std::string& s) {
                                          return s == value;
                                      });
        EXPECT_EQ(keys_of(table.select_keys(
                      Predicate("category", PredOp::Eq, Datum{value}))),
                  want)
            << "category == " << category;
    }
}

// ---------------------------------------------------------------------------
// Reopen
// ---------------------------------------------------------------------------

TEST(TableReopen, RecordsComeBackFromTheWalAndTheCatalogReplaysTheDecision) {
    // The moment the whole stack is visible at once: module 1 brings the rows
    // back, module 4's catalog brings the *decision* back without re-measuring
    // it, and the typed schema is what makes a stored plan interpretable.
    TempDir dir("reopen");
    IndexKind chosen = IndexKind::BPlusTree;
    KeyEncoding encoding = KeyEncoding::Native;

    {
        RecordStore store(dir.str());
        Table table(store, shop_schema());
        table.put_batch(shop_rows(500));
        table.create_index("price");
        table.create_index("category");
        chosen = table.info("price").kind;
        encoding = table.info("category").encoding;
        table.checkpoint();
        store.close();
    }

    RecordStore store(dir.str());
    Table table = Table::open(store);
    EXPECT_EQ(table.size(), 500u);
    EXPECT_EQ(table.schema().size(), 4u);
    EXPECT_EQ(table.schema().type_of("category"), LogicalType::String);

    // The plan survived, with its measured evidence.
    const auto stored = table.catalog().get("price");
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->kind, chosen);
    EXPECT_GT(stored->ns_per_lookup, 0.0);

    IndexCatalog::Decision decision;
    // Rebuilding on the same data must *replay*, not re-measure: producing a
    // plan means building and timing every candidate, and that is the
    // expensive part this file exists to avoid repeating.
    std::vector<std::int64_t> keys;
    std::vector<std::int64_t> rows;
    for (const Record& r : store.records()) {
        const auto it = r.columns.find("price");
        if (it == r.columns.end()) continue;
        keys.push_back(std::stoll(it->second));
        rows.push_back(r.key);
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
    for (std::size_t i = 0; i < keys.size(); ++i) pairs.emplace_back(keys[i], rows[i]);
    std::sort(pairs.begin(), pairs.end());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        keys[i] = pairs[i].first;
        rows[i] = pairs[i].second;
    }
    IndexCatalog catalog = IndexCatalog::load(
        (std::filesystem::path(dir.str()) / Table::CATALOG_NAME).string());
    catalog.build_column_typed("price", LogicalType::Int64, keys, rows,
                               Workload{}, &decision);
    EXPECT_EQ(decision.action, IndexCatalog::Action::Replayed);

    table.create_index("category");
    EXPECT_EQ(table.info("category").encoding, encoding);
    EXPECT_NO_THROW(table.validate());
}

TEST(TableReopen, ARetypedColumnIsRefusedRatherThanSilentlyReindexed) {
    TempDir dir("retype");
    {
        RecordStore store(dir.str());
        Table table(store, shop_schema());
        table.put_batch(shop_rows(20));
        table.save();
        store.close();
    }

    Schema changed;
    changed.add(ColumnDef("price", LogicalType::String));  // was Int64
    changed.add(ColumnDef("category", LogicalType::String));
    changed.add(ColumnDef("title", LogicalType::String));
    changed.add(ColumnDef("created_at", LogicalType::Timestamp));

    RecordStore store(dir.str());
    EXPECT_THROW(Table(store, changed), std::runtime_error);
}

TEST(TableReopen, AddingAColumnIsAllowed) {
    TempDir dir("add_column");
    {
        RecordStore store(dir.str());
        Table table(store, shop_schema());
        table.put_batch(shop_rows(20));
        table.save();
        store.close();
    }

    Schema wider = shop_schema();
    wider.add(ColumnDef("brand", LogicalType::String));

    RecordStore store(dir.str());
    Table table(store, wider);
    EXPECT_EQ(table.schema().size(), 5u);
    // Existing rows have no value for it, so they are in no index and match
    // no predicate on it.
    table.create_index("brand");
    EXPECT_EQ(table.info("brand").rows, 0u);
    EXPECT_EQ(table.info("brand").skipped, 20u);
}

TEST(TableReopen, OpeningWithoutAStoredSchemaSaysWhatIsMissing) {
    TempDir dir("no_schema");
    RecordStore store(dir.str());
    EXPECT_THROW(Table::open(store), std::runtime_error);
}
