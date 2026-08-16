// Tests for vector columns: embeddings that belong to a table's rows.
//
// Three claims, and they are the three things phase E actually added.
//
//   1. **The two id spaces stay joined.** A vector index numbers rows 0..n-1
//      into a float buffer; the table numbers them by record key. Every test
//      that asks a question in keys and checks an answer in keys is testing
//      that translation, because getting it wrong returns plausible neighbours
//      belonging to the wrong rows.
//   2. **A deleted row leaves every answer immediately.** HNSW cannot give a
//      node back, so deletion is a mask rather than a removal, and the thing
//      that must be true is that the orphan is unreachable — not that it is
//      gone.
//   3. **A reopen reproduces the index, not something like it.** Only the
//      vectors are stored; the graph is rebuilt by replaying the same
//      insertions. With the seed fixed that is exact, so the assertion is
//      neighbour-for-neighbour equality rather than a recall threshold.
//
// The oracle throughout is the exact index, which the column always keeps.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "query/table.hpp"

using hylis::index::Metric;
using hylis::index::LogicalType;
using hylis::query::ColumnDef;
using hylis::query::HybridTrace;
using hylis::query::PlanKind;
using hylis::query::PredOp;
using hylis::query::Predicate;
using hylis::query::Schema;
using hylis::query::Table;
using hylis::query::VectorColumn;
using hylis::query::VectorMatch;
using hylis::query::VectorPlan;
using hylis::query::VectorStructure;
using hylis::storage::Record;
using hylis::storage::RecordStore;

namespace {

namespace fs = std::filesystem;

class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_(fs::path(HYLIS_TEST_TMP) / ("vector_" + name)) {
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    fs::path path() const { return path_; }
    std::string str() const { return path_.string(); }

private:
    fs::path path_;
};

constexpr std::size_t kDim = 8;

Schema shop_schema() {
    Schema s;
    s.add(ColumnDef("price", LogicalType::Int64));
    s.add(ColumnDef("band", LogicalType::Int64));
    s.add(ColumnDef("category", LogicalType::String));
    s.add(ColumnDef("image", LogicalType::Vector, kDim));
    return s;
}

// Clustered rather than uniform: uniform vectors in 8 dimensions have nearly
// equal pairwise distances, so a wrong answer looks like a right one and every
// ranking assertion becomes a coin flip.
std::vector<float> clustered(std::size_t n, std::uint64_t seed = 7) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> jitter(0.0f, 0.05f);
    std::uniform_int_distribution<int> pick(0, 7);

    std::vector<std::vector<float>> centres;
    std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
    for (int c = 0; c < 8; ++c) {
        std::vector<float> centre(kDim);
        for (float& v : centre) v = spread(rng);
        centres.push_back(centre);
    }

    std::vector<float> out;
    out.reserve(n * kDim);
    for (std::size_t i = 0; i < n; ++i) {
        const std::vector<float>& centre = centres[static_cast<std::size_t>(pick(rng))];
        for (std::size_t d = 0; d < kDim; ++d) out.push_back(centre[d] + jitter(rng));
    }
    return out;
}

std::vector<Record> shop_rows(int n) {
    std::vector<Record> rows;
    for (int i = 0; i < n; ++i) {
        Record r(i);
        const int price = (i * 37) % 500;
        r.columns["price"] = std::to_string(price);
        r.columns["band"] = std::to_string(price / 100);
        r.columns["category"] = (i % 3 == 0) ? "bags" : (i % 3 == 1 ? "hats" : "shoes");
        rows.push_back(r);
    }
    return rows;
}

std::vector<float> row_of(const std::vector<float>& data, std::size_t i) {
    return std::vector<float>(data.begin() + static_cast<std::ptrdiff_t>(i * kDim),
                              data.begin() + static_cast<std::ptrdiff_t>((i + 1) * kDim));
}

std::vector<std::int64_t> keys_of(const std::vector<VectorMatch>& matches) {
    std::vector<std::int64_t> out;
    out.reserve(matches.size());
    for (const VectorMatch& m : matches) out.push_back(m.key);
    return out;
}

// A loaded table: n records, n embeddings, row id == record key.
struct Fixture {
    explicit Fixture(TempDir& dir, int n, VectorPlan plan = VectorPlan{})
        : store(dir.str()), table(store, shop_schema()), data(clustered(static_cast<std::size_t>(n))) {
        table.put_batch(shop_rows(n));
        table.create_vector_index("image", plan);
        std::vector<std::int64_t> keys;
        for (int i = 0; i < n; ++i) keys.push_back(i);
        table.put_vectors("image", keys, data);
    }

    RecordStore store;
    Table table;
    std::vector<float> data;
};

}  // namespace

// ---------------------------------------------------------------------------
// The key <-> row join
// ---------------------------------------------------------------------------

TEST(VectorColumn, AnswersInRecordKeysNotRowIds) {
    TempDir dir("keys_not_rows");
    // Keys deliberately not 0..n-1: with contiguous keys the translation is
    // the identity and a missing translation would pass every test.
    RecordStore store(dir.str());
    Table table(store, shop_schema());

    std::vector<Record> rows;
    std::vector<std::int64_t> keys;
    for (int i = 0; i < 40; ++i) {
        const std::int64_t key = 1000 + i * 7;
        Record r(key);
        r.columns["price"] = std::to_string(i);
        r.columns["band"] = "0";
        r.columns["category"] = "bags";
        rows.push_back(r);
        keys.push_back(key);
    }
    table.put_batch(rows);
    table.create_vector_index("image", VectorPlan{VectorStructure::Exact});

    const std::vector<float> data = clustered(40);
    table.put_vectors("image", keys, data);

    const auto matches = table.knn("image", row_of(data, 5), 3);
    ASSERT_FALSE(matches.empty());
    // Its own vector, so it is its own nearest neighbour.
    EXPECT_EQ(matches.front().key, 1000 + 5 * 7);
    EXPECT_EQ(matches.front().row, 5);
    for (const VectorMatch& m : matches) {
        EXPECT_TRUE(table.get(m.key).has_value())
            << "knn returned key " << m.key << ", which is not a record";
    }
    EXPECT_FALSE(table.vector_info("image").rows_are_keys);
}

TEST(VectorColumn, RowsAreKeysOnlyWhenTheyReallyAre) {
    TempDir contiguous("rows_are_keys");
    Fixture f(contiguous, 30, VectorPlan{VectorStructure::Exact});
    EXPECT_TRUE(f.table.vector_info("image").rows_are_keys);

    // One replacement appends a row past the end, and the two spaces part.
    f.table.put_vector(3, "image", row_of(f.data, 4));
    EXPECT_FALSE(f.table.vector_info("image").rows_are_keys);
}

TEST(VectorColumn, RefusesAnEmbeddingForARecordThatDoesNotExist) {
    TempDir dir("orphan_record");
    Fixture f(dir, 20, VectorPlan{VectorStructure::Exact});
    EXPECT_THROW(f.table.put_vector(9999, "image", row_of(f.data, 0)),
                 std::invalid_argument);
}

TEST(VectorColumn, RefusesAVectorInTheRecordPayload) {
    TempDir dir("vector_in_payload");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    Record r(1);
    r.columns["image"] = "0.1 0.2 0.3";
    // The schema catches it, and the message has to name put_vector -- a
    // refusal that does not say what to do instead is a worse refusal.
    try {
        table.put(r);
        FAIL() << "a vector in the payload was accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("put_vector"), std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

TEST(VectorColumn, TheGraphAgreesWithTheExactScanOnEveryQuery) {
    TempDir dir("graph_vs_exact");
    Fixture f(dir, 500);
    ASSERT_TRUE(f.table.vector_info("image").has_graph);

    std::size_t agreed = 0;
    for (std::size_t q = 0; q < 20; ++q) {
        const std::vector<float> query = row_of(f.data, q * 13);
        const auto approx = f.table.knn("image", query, 5, 64, /*exact=*/false);
        const auto truth = f.table.knn("image", query, 5, 0, /*exact=*/true);
        ASSERT_EQ(approx.size(), truth.size());
        // The nearest is the query's own row either way; the rest is recall,
        // which is a measurement rather than an assertion.
        EXPECT_EQ(approx.front().key, truth.front().key);
        if (keys_of(approx) == keys_of(truth)) ++agreed;
    }
    // Not 20/20: HNSW is approximate by construction, and a test that demanded
    // exactness would be asserting the wrong property. This asserts it is not
    // *broken*, which is what a graph disagreeing on half its queries would be.
    EXPECT_GE(agreed, 16u);
}

TEST(VectorColumn, MoreLikeThisExcludesItsOwnSeed) {
    TempDir dir("knn_by_key");
    Fixture f(dir, 200, VectorPlan{VectorStructure::Exact});

    const auto matches = f.table.knn_by_key("image", 42, 5);
    ASSERT_EQ(matches.size(), 5u);
    for (const VectorMatch& m : matches) EXPECT_NE(m.key, 42);

    // And it is otherwise exactly the knn answer with the seed removed: a seed
    // dropped by post-filtering would have returned four rows, not five.
    const auto plain = f.table.knn("image", row_of(f.data, 42), 6);
    std::vector<std::int64_t> expected = keys_of(plain);
    expected.erase(std::remove(expected.begin(), expected.end(), 42),
                   expected.end());
    EXPECT_EQ(keys_of(matches), expected);
}

TEST(VectorColumn, MoreLikeThisSaysSoWhenTheRowHasNoEmbedding) {
    TempDir dir("knn_by_key_missing");
    Fixture f(dir, 20, VectorPlan{VectorStructure::Exact});
    Record extra(9001);
    extra.columns["price"] = "1";
    f.table.put(extra);
    EXPECT_THROW(f.table.knn_by_key("image", 9001, 3), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Deletion, orphans and compaction
// ---------------------------------------------------------------------------

TEST(VectorColumn, ADeletedRowLeavesEveryAnswerAtOnce) {
    TempDir dir("delete_leaves");
    Fixture f(dir, 300);

    const std::vector<float> query = row_of(f.data, 100);
    const auto before = f.table.knn("image", query, 5, 64);
    ASSERT_EQ(before.front().key, 100);

    f.table.erase(100);
    EXPECT_EQ(f.table.vector_info("image").orphans, 1u);
    EXPECT_EQ(f.table.vector_info("image").rows, 299u);

    for (bool exact : {true, false}) {
        const auto after = f.table.knn("image", query, 5, 64, exact);
        EXPECT_EQ(after.size(), 5u) << "exact=" << exact;
        for (const VectorMatch& m : after) {
            EXPECT_NE(m.key, 100) << "the deleted row came back, exact=" << exact;
        }
    }
    EXPECT_NO_THROW(f.table.validate());
}

TEST(VectorColumn, ReplacingAnEmbeddingOrphansTheOldOneAndAnswersWithTheNew) {
    TempDir dir("replace");
    Fixture f(dir, 200, VectorPlan{VectorStructure::Exact});

    // Give row 7 row 150's vector. Searching near 150 must now find 7.
    f.table.put_vector(7, "image", row_of(f.data, 150));
    EXPECT_EQ(f.table.vector_info("image").orphans, 1u);
    EXPECT_EQ(f.table.vector_info("image").rows, 200u);

    const auto near150 = f.table.knn("image", row_of(f.data, 150), 3);
    const std::vector<std::int64_t> found = keys_of(near150);
    EXPECT_NE(std::find(found.begin(), found.end(), 7), found.end());

    // And 7's old vector must be unreachable: a search at exactly that point
    // may not return 7 twice, nor return the orphan row.
    const auto near7 = f.table.knn("image", row_of(f.data, 7), 5);
    const std::vector<std::int64_t> near7_keys = keys_of(near7);
    const std::set<std::int64_t> unique(near7_keys.begin(), near7_keys.end());
    EXPECT_EQ(unique.size(), near7.size());
    EXPECT_NO_THROW(f.table.validate());
}

TEST(VectorColumn, CompactionReclaimsOrphansAndChangesNoAnswer) {
    TempDir dir("compact");
    Fixture f(dir, 400);

    const std::vector<float> query = row_of(f.data, 33);
    for (std::int64_t key : {5, 17, 200, 201, 202}) f.table.erase(key);
    ASSERT_EQ(f.table.vector_info("image").orphans, 5u);

    const auto before = f.table.knn("image", query, 8, 64, /*exact=*/true);
    const std::size_t reclaimed = f.table.compact_vectors("image");
    EXPECT_EQ(reclaimed, 5u);
    EXPECT_EQ(f.table.vector_info("image").orphans, 0u);
    EXPECT_EQ(f.table.vector_info("image").rows, 395u);

    const auto after = f.table.knn("image", query, 8, 64, /*exact=*/true);
    EXPECT_EQ(keys_of(before), keys_of(after))
        << "compaction renumbered rows and changed the answer";
    EXPECT_NO_THROW(f.table.validate());
}

TEST(VectorColumn, ValidateCatchesAnEmbeddingWhoseRecordIsGone) {
    TempDir dir("dangling");
    Fixture f(dir, 30, VectorPlan{VectorStructure::Exact});
    // Behind the table's back: RecordStore::del does not know about vectors,
    // which is exactly the state validate() exists to find.
    f.store.del(11);
    EXPECT_THROW(f.table.validate(), std::logic_error);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

TEST(VectorColumn, ReopenReproducesTheIndexExactlyRatherThanApproximately) {
    TempDir dir("reopen");
    std::vector<float> data;
    std::vector<std::vector<VectorMatch>> before;

    {
        Fixture f(dir, 400);
        data = f.data;
        for (std::size_t q = 0; q < 10; ++q) {
            before.push_back(f.table.knn("image", row_of(data, q * 31), 10, 64));
        }
        f.table.checkpoint();
    }

    RecordStore store(dir.str());
    Table table = Table::open(store);
    ASSERT_TRUE(table.has_vector_index("image"));
    EXPECT_EQ(table.vector_info("image").rows, 400u);
    EXPECT_EQ(table.vector_info("image").orphans, 0u);
    EXPECT_TRUE(table.vector_info("image").has_graph);

    for (std::size_t q = 0; q < 10; ++q) {
        const auto after = table.knn("image", row_of(data, q * 31), 10, 64);
        ASSERT_EQ(after.size(), before[q].size());
        for (std::size_t i = 0; i < after.size(); ++i) {
            // Neighbour for neighbour, not recall-within-a-threshold. The
            // graph is not stored; it is replayed, and the seed is fixed, so
            // "the same" is the correct claim and anything weaker would hide a
            // reload that quietly built a different graph.
            EXPECT_EQ(after[i].key, before[q][i].key) << "query " << q << " rank " << i;
            EXPECT_FLOAT_EQ(after[i].score, before[q][i].score);
        }
    }
    EXPECT_NO_THROW(table.validate());
}

TEST(VectorColumn, TheSidecarHoldsLiveRowsOnlyEvenWithoutCompaction) {
    TempDir dir("sidecar_compacts");
    {
        Fixture f(dir, 100, VectorPlan{VectorStructure::Exact});
        for (std::int64_t key : {1, 2, 3}) f.table.erase(key);
        ASSERT_EQ(f.table.vector_info("image").orphans, 3u);
        f.table.checkpoint();
    }

    // 97 vectors of 8 floats, each preceded by an int32 dimension.
    const fs::path fvecs = dir.path() / "image.fvecs";
    ASSERT_TRUE(fs::exists(fvecs));
    EXPECT_EQ(fs::file_size(fvecs), 97u * (sizeof(std::int32_t) + kDim * sizeof(float)));

    RecordStore store(dir.str());
    Table table = Table::open(store);
    EXPECT_EQ(table.vector_info("image").rows, 97u);
    EXPECT_EQ(table.vector_info("image").orphans, 0u)
        << "the reopened column inherited holes the sidecar should not contain";
    for (std::int64_t key : {1, 2, 3}) {
        EXPECT_FALSE(table.vector_column("image").contains(key));
    }
}

TEST(VectorColumn, RefusesASidecarWhoseDimensionDisagreesWithTheSchema) {
    TempDir dir("dim_mismatch");
    {
        Fixture f(dir, 20, VectorPlan{VectorStructure::Exact});
        f.table.checkpoint();
    }

    Schema wider;
    wider.add(ColumnDef("price", LogicalType::Int64));
    wider.add(ColumnDef("band", LogicalType::Int64));
    wider.add(ColumnDef("category", LogicalType::String));
    wider.add(ColumnDef("image", LogicalType::Vector, kDim * 2));

    RecordStore store(dir.str());
    // Reinterpreting 8 floats as 16 would silently pair every row with half of
    // its neighbour, so this has to be a refusal rather than a best effort.
    EXPECT_THROW(Table(store, wider), std::runtime_error);
}

TEST(VectorColumn, EmbeddingsAttachedSinceTheLastSaveDoNotSurviveACrash) {
    TempDir dir("no_wal");
    {
        Fixture f(dir, 50, VectorPlan{VectorStructure::Exact});
        f.table.checkpoint();
        Record late(50);
        late.columns["price"] = "1";
        late.columns["band"] = "0";
        late.columns["category"] = "bags";
        f.table.put(late);
        f.table.put_vector(50, "image", row_of(f.data, 0));
        // No checkpoint, no save_vectors: the process ends here.
    }

    RecordStore store(dir.str());
    Table table = Table::open(store);
    // The record survived, because it went through the write-ahead log.
    EXPECT_TRUE(table.get(50).has_value());
    // The embedding did not, because vectors never do. Asserted rather than
    // left implicit: it is the stated cost of keeping a 128-float payload out
    // of a JSON WAL, and a test is the only thing that keeps it stated.
    EXPECT_EQ(table.vector_info("image").rows, 50u);
    EXPECT_FALSE(table.vector_column("image").contains(50));
}

TEST(VectorColumn, RetuningKeepsEveryEmbeddingAndRefusesAMetricChange) {
    TempDir dir("retune");
    Fixture f(dir, 150, VectorPlan{VectorStructure::Exact});

    const auto before = f.table.knn("image", row_of(f.data, 20), 5);

    VectorPlan graph;
    graph.structure = VectorStructure::Graph;
    graph.M = 8;
    f.table.create_vector_index("image", graph);
    EXPECT_TRUE(f.table.vector_info("image").has_graph);
    EXPECT_EQ(f.table.vector_info("image").rows, 150u);

    const auto after = f.table.knn("image", row_of(f.data, 20), 5, 0, /*exact=*/true);
    EXPECT_EQ(keys_of(before), keys_of(after));

    VectorPlan cosine = graph;
    cosine.metric = Metric::Cosine;
    EXPECT_THROW(f.table.create_vector_index("image", cosine), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Hybrid queries
// ---------------------------------------------------------------------------

TEST(VectorHybrid, EveryPlanReturnsTheSameRows) {
    TempDir dir("plan_agreement");
    Fixture f(dir, 600);
    f.table.create_index("price");
    f.table.create_index_as("band", hylis::index::IndexKind::Bitmap);

    const std::vector<Predicate> predicates{
        Predicate{"band", PredOp::Lt, hylis::index::Datum{std::int64_t{3}},
                  hylis::index::Datum{}}};
    const std::vector<float> query = row_of(f.data, 77);
    // ef high enough that the graph is not losing recall to beam width: the
    // question here is whether the plans agree, not how well HNSW is tuned.
    const std::size_t ef = 400;

    HybridTrace chosen;
    const auto planned = f.table.hybrid(predicates, "image", query, 10, ef, &chosen);
    ASSERT_EQ(planned.size(), 10u);

    for (PlanKind kind : {PlanKind::PreFilter, PlanKind::FilteredGraph,
                          PlanKind::BitmapFilteredGraph}) {
        ASSERT_TRUE(f.table.plan_available(kind, predicates, "image"))
            << hylis::query::to_string(kind);
        const auto forced = f.table.hybrid_with(kind, predicates, "image", query, 10, ef);
        EXPECT_EQ(keys_of(forced), keys_of(planned))
            << "plan " << hylis::query::to_string(kind) << " disagreed";
    }
}

TEST(VectorHybrid, EveryReturnedRowSatisfiesThePredicate) {
    TempDir dir("filter_holds");
    Fixture f(dir, 500);
    f.table.create_index("price");

    const std::vector<Predicate> predicates{
        Predicate{"price", PredOp::Lt, hylis::index::Datum{std::int64_t{200}},
                  hylis::index::Datum{}}};

    for (std::size_t q = 0; q < 8; ++q) {
        const auto matches =
            f.table.hybrid(predicates, "image", row_of(f.data, q * 41), 10, 64);
        for (const VectorMatch& m : matches) {
            const auto record = f.table.get(m.key);
            ASSERT_TRUE(record.has_value());
            EXPECT_LT(std::stoll(record->columns.at("price")), 200);
        }
    }
}

TEST(VectorHybrid, TheBitmapPlanCostsNoRowIdsAndSaysSo) {
    TempDir dir("bitmap_plan");
    Fixture f(dir, 600);
    f.table.create_index_as("band", hylis::index::IndexKind::Bitmap);

    const std::vector<Predicate> predicates{
        Predicate{"band", PredOp::Ge, hylis::index::Datum{std::int64_t{1}},
                  hylis::index::Datum{}}};

    HybridTrace trace;
    const auto matches =
        f.table.hybrid(predicates, "image", row_of(f.data, 5), 10, 64, &trace);
    EXPECT_TRUE(trace.mask_used);
    EXPECT_TRUE(trace.plan.selectivity_was_free);
    EXPECT_EQ(trace.plan.kind, PlanKind::BitmapFilteredGraph);
    EXPECT_EQ(matches.size(), 10u);

    // And the plan stops being reachable the moment the two row spaces part,
    // which is the precondition being enforced rather than assumed.
    f.table.erase(0);
    EXPECT_FALSE(f.table.plan_available(PlanKind::BitmapFilteredGraph, predicates,
                                        "image"));
    EXPECT_THROW(f.table.hybrid_with(PlanKind::BitmapFilteredGraph, predicates,
                                     "image", row_of(f.data, 5), 5, 64),
                 std::invalid_argument);
    // The query still runs; it just takes the other route.
    HybridTrace after;
    EXPECT_NO_THROW(f.table.hybrid(predicates, "image", row_of(f.data, 5), 5, 64,
                                   &after));
    EXPECT_FALSE(after.mask_used);
}

TEST(VectorHybrid, ConjunctionsOverTwoBitmapColumnsReachTheMaskPath) {
    TempDir dir("bitmap_conjunction");
    Fixture f(dir, 600);
    f.table.create_index_as("band", hylis::index::IndexKind::Bitmap);
    f.table.create_index_as("category", hylis::index::IndexKind::Bitmap);

    const std::vector<Predicate> predicates{
        Predicate{"band", PredOp::Ge, hylis::index::Datum{std::int64_t{1}},
                  hylis::index::Datum{}},
        Predicate{"category", PredOp::Eq, hylis::index::Datum{std::string("bags")},
                  hylis::index::Datum{}}};

    HybridTrace trace;
    const auto matches =
        f.table.hybrid(predicates, "image", row_of(f.data, 9), 5, 64, &trace);
    EXPECT_TRUE(trace.mask_used);
    for (const VectorMatch& m : matches) {
        const auto record = f.table.get(m.key);
        ASSERT_TRUE(record.has_value());
        EXPECT_EQ(record->columns.at("category"), "bags");
        EXPECT_GE(std::stoll(record->columns.at("band")), 1);
    }
}

TEST(VectorHybrid, CountsPredicateMatchesThatHaveNoEmbedding) {
    TempDir dir("without_vector");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(200));
    table.create_index("price");
    table.create_vector_index("image", VectorPlan{VectorStructure::Exact});

    // Only the even rows get an embedding.
    const std::vector<float> data = clustered(100);
    std::vector<std::int64_t> keys;
    for (int i = 0; i < 200; i += 2) keys.push_back(i);
    table.put_vectors("image", keys, data);

    const std::vector<Predicate> predicates{
        Predicate{"price", PredOp::Lt, hylis::index::Datum{std::int64_t{500}},
                  hylis::index::Datum{}}};

    HybridTrace trace;
    const auto matches = table.hybrid(predicates, "image", row_of(data, 3), 5, 0,
                                      &trace);
    // The predicate matches all 200; half of them cannot be neighbours, and
    // the count is reported rather than the shortfall being silent.
    EXPECT_EQ(trace.structured.matched, 200u);
    EXPECT_EQ(trace.without_vector, 100u);
    EXPECT_EQ(trace.plan.matched_rows, 100u);
    for (const VectorMatch& m : matches) EXPECT_EQ(m.key % 2, 0);
}

TEST(VectorHybrid, NoPredicateDegeneratesToASimilaritySearch) {
    TempDir dir("no_predicate");
    Fixture f(dir, 200, VectorPlan{VectorStructure::Exact});

    HybridTrace trace;
    const auto matches = f.table.hybrid({}, "image", row_of(f.data, 12), 5, 0,
                                        &trace);
    EXPECT_EQ(trace.plan.kind, PlanKind::NoPredicate);
    EXPECT_EQ(keys_of(matches), keys_of(f.table.knn("image", row_of(f.data, 12), 5)));
}

TEST(VectorHybrid, APredicateNoIndexCanServeStillReachesTheVectorSearch) {
    TempDir dir("scan_predicate");
    Fixture f(dir, 300, VectorPlan{VectorStructure::Exact});

    // Contains has no ordering to exploit, so Table answers it by scanning --
    // and the row ids that fall out are still the vector search's filter. That
    // seam is why the planner grew search_rows().
    const std::vector<Predicate> predicates{
        Predicate{"category", PredOp::Contains,
                  hylis::index::Datum{std::string("ag")}, hylis::index::Datum{}}};

    HybridTrace trace;
    const auto matches =
        f.table.hybrid(predicates, "image", row_of(f.data, 4), 5, 0, &trace);
    EXPECT_FALSE(trace.structured.used_index);
    EXPECT_GT(trace.structured.scanned, 0u);
    ASSERT_FALSE(matches.empty());
    for (const VectorMatch& m : matches) {
        EXPECT_EQ(f.table.get(m.key)->columns.at("category"), "bags");
    }
}

TEST(VectorHybrid, RefusesToPlanOverAColumnWithNoVectorIndex) {
    TempDir dir("no_vector_index");
    RecordStore store(dir.str());
    Table table(store, shop_schema());
    table.put_batch(shop_rows(20));

    EXPECT_THROW(table.knn("image", std::vector<float>(kDim, 0.0f), 3),
                 std::invalid_argument);
    // And a scalar column named where a vector one belongs is a different
    // mistake with a different message.
    EXPECT_THROW(table.knn("price", std::vector<float>(kDim, 0.0f), 3),
                 std::invalid_argument);
}
