// Tests for the hybrid query planner.
//
// One property carries the module: **every plan must return the same rows.**
// A planner chooses between costs, never between answers. If PreFilter and
// FilteredGraph could disagree, the planner would be silently picking which
// results the user gets based on how selective their predicate happened to
// be — the single worst thing a query optimiser can do, and the one a
// correctness-by-inspection review would miss.
//
// Everything else here checks the planner is *correct*. One test at the end
// checks it is *useful*, which is a different question and the actual point.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "index/column_index.hpp"
#include "query/planner.hpp"

using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::ColumnValue;
using hylis::index::Datum;
using hylis::query::PredOp;
using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;
using hylis::query::HybridPlanner;
using hylis::query::PlanKind;
using hylis::query::Predicate;
using hylis::query::QueryPlan;

namespace {

// A corpus of vectors plus one scalar attribute per row, which is what a
// hybrid query needs and what hylis.datasets.make_hybrid generates on the
// Python side.
struct Corpus {
    std::size_t n = 0;
    std::size_t dim = 0;
    std::vector<float> base;
    std::vector<float> query;
    // attribute[row] -> a value in [0, n). Held so a test can compute the
    // true answer without going through the planner.
    std::vector<ColumnKey> attribute;

    FlatIndex exact{1};
    HnswIndex graph{1};
};

// The attribute is a permutation of 0..n-1, so "attribute < c" selects
// exactly c rows. Exact selectivity makes the plan-choice tests statements
// about the planner rather than about the sampling.
Corpus make_corpus(std::size_t n, std::size_t dim, unsigned seed) {
    Corpus c;
    c.n = n;
    c.dim = dim;
    c.base.resize(n * dim);
    c.query.resize(dim);

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss;
    // Clustered rather than uniform noise, so nearest neighbours are actually
    // structured and a filtered search has something to get wrong.
    const std::size_t centres = 32;
    std::vector<float> centre(centres * dim);
    for (float& v : centre) v = gauss(rng) * 4.0f;
    std::uniform_int_distribution<std::size_t> pick(0, centres - 1);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t k = pick(rng);
        for (std::size_t j = 0; j < dim; ++j) {
            c.base[i * dim + j] = centre[k * dim + j] + gauss(rng);
        }
    }
    for (std::size_t j = 0; j < dim; ++j) c.query[j] = centre[j] + gauss(rng);

    c.attribute.resize(n);
    for (std::size_t i = 0; i < n; ++i) c.attribute[i] = static_cast<ColumnKey>(i);
    std::shuffle(c.attribute.begin(), c.attribute.end(), rng);

    c.exact = FlatIndex(dim);
    c.exact.add_batch(c.base.data(), n);
    c.graph = HnswIndex(dim, Metric::L2, 16, 200);
    c.graph.add_batch(c.base.data(), n);
    return c;
}

// The column maps attribute value -> row id, which is the direction a
// predicate is evaluated in.
ColumnIndex attribute_column(const Corpus& c) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    keys.reserve(c.n);
    values.reserve(c.n);
    for (std::size_t row = 0; row < c.n; ++row) {
        keys.push_back(c.attribute[row]);
        values.push_back(static_cast<ColumnValue>(row));
    }
    // range_query wants ascending keys, so sort by attribute.
    std::vector<std::size_t> order(c.n);
    for (std::size_t i = 0; i < c.n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });

    std::vector<ColumnKey> sorted_keys;
    std::vector<ColumnValue> sorted_values;
    sorted_keys.reserve(c.n);
    sorted_values.reserve(c.n);
    for (std::size_t i : order) {
        sorted_keys.push_back(keys[i]);
        sorted_values.push_back(values[i]);
    }
    return ColumnIndex::build(sorted_keys, sorted_values);
}

HybridPlanner make_planner(const Corpus& c, double threshold = 0.5) {
    HybridPlanner planner(threshold);
    planner.set_column("attr", attribute_column(c));
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);
    return planner;
}

std::vector<std::int64_t> ids_of(const std::vector<Neighbor>& ns) {
    std::vector<std::int64_t> out;
    out.reserve(ns.size());
    for (const Neighbor& n : ns) out.push_back(n.id);
    return out;
}

// The true answer, computed without the planner: filter by hand, then scan
// the survivors exhaustively.
std::vector<std::int64_t> oracle(const Corpus& c, PredOp op, ColumnKey value,
                                 std::size_t k) {
    std::vector<std::int64_t> allowed;
    for (std::size_t row = 0; row < c.n; ++row) {
        const ColumnKey a = c.attribute[row];
        const bool hit = (op == PredOp::Eq && a == value) ||
                         (op == PredOp::Lt && a < value) ||
                         (op == PredOp::Le && a <= value) ||
                         (op == PredOp::Gt && a > value) ||
                         (op == PredOp::Ge && a >= value);
        if (hit) allowed.push_back(static_cast<std::int64_t>(row));
    }
    return ids_of(c.exact.search_filtered(c.query.data(), k, allowed));
}

}  // namespace

// ------------------------------------------------------ the join exists

TEST(HybridPlanner, AnswersAQueryNeitherIndexCouldAnswerAlone) {
    const Corpus c = make_corpus(4000, 16, 1);
    const HybridPlanner planner = make_planner(c);

    Predicate p{"attr", PredOp::Lt, Datum{static_cast<std::int64_t>(400)}};  // 10% of rows
    QueryPlan plan;
    const auto got = planner.search(p, c.query.data(), 10, 64, &plan);

    ASSERT_EQ(got.size(), 10u);
    EXPECT_EQ(plan.matched_rows, 400u);
    EXPECT_NEAR(plan.selectivity, 0.1, 1e-9);

    // Every returned row satisfies the predicate. Without this the planner
    // could be returning nearest neighbours that the WHERE clause excludes.
    for (const Neighbor& n : got) {
        EXPECT_LT(c.attribute[static_cast<std::size_t>(n.id)], 400)
            << "row " << n.id << " violates the predicate";
    }
    EXPECT_EQ(ids_of(got), oracle(c, PredOp::Lt, 400, 10));
}

TEST(HybridPlanner, TheStructuredHalfIsAvailableOnItsOwn) {
    const Corpus c = make_corpus(2000, 8, 2);
    const HybridPlanner planner = make_planner(c);
    const auto rows = planner.matching_rows({"attr", PredOp::Lt, 250});
    EXPECT_EQ(rows.size(), 250u);
    for (ColumnValue row : rows) {
        EXPECT_LT(c.attribute[static_cast<std::size_t>(row)], 250);
    }
}

// ------------------------------------------- plans agree, always

class PlanAgreement : public ::testing::TestWithParam<double> {};

TEST_P(PlanAgreement, EveryPlanReturnsTheSameRows) {
    const double selectivity = GetParam();
    const Corpus c = make_corpus(4000, 16, 3);
    const HybridPlanner planner = make_planner(c);

    const auto cut = static_cast<ColumnKey>(selectivity * static_cast<double>(c.n));
    Predicate p{"attr", PredOp::Lt, Datum{static_cast<std::int64_t>(cut)}};
    const std::size_t k = 10;

    // ef high enough that the graph is not losing recall to beam width — the
    // question here is whether the plans agree, not how well HNSW is tuned.
    const auto pre = planner.search_with(PlanKind::PreFilter, p, c.query.data(), k, 400);
    const auto graph = planner.search_with(PlanKind::FilteredGraph, p, c.query.data(), k, 400);
    const auto truth = oracle(c, PredOp::Lt, cut, k);

    EXPECT_EQ(ids_of(pre), truth) << "the exact plan disagreed with the oracle";
    EXPECT_EQ(ids_of(graph), truth)
        << "the graph plan returned different rows at selectivity "
        << selectivity << "; the planner would be choosing answers, not costs";
}

TEST_P(PlanAgreement, ThePostFilterPlanIsAPrefixOfTheTruth) {
    // The trap plan, held to the weaker standard it can actually meet: it may
    // return fewer than k rows, but whatever it returns must be the *right*
    // rows in the right order. Letting it off entirely would hide a real bug.
    const double selectivity = GetParam();
    const Corpus c = make_corpus(4000, 16, 4);
    const HybridPlanner planner = make_planner(c);

    const auto cut = static_cast<ColumnKey>(selectivity * static_cast<double>(c.n));
    Predicate p{"attr", PredOp::Lt, Datum{static_cast<std::int64_t>(cut)}};
    const std::size_t k = 10;

    const auto post = ids_of(
        planner.search_with(PlanKind::PostFilter, p, c.query.data(), k, 400));
    const auto truth = oracle(c, PredOp::Lt, cut, k);

    ASSERT_LE(post.size(), truth.size());
    for (std::size_t i = 0; i < post.size(); ++i) {
        EXPECT_EQ(post[i], truth[i]) << "post-filter diverged at position " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(AcrossSelectivities, PlanAgreement,
                         ::testing::Values(0.001, 0.01, 0.1, 0.5, 0.9, 1.0),
                         [](const auto& info) {
                             const int permille =
                                 static_cast<int>(info.param * 1000.0 + 0.5);
                             return "sel_" + std::to_string(permille) + "permille";
                         });

TEST(HybridPlanner, EveryPredicateAgreesWithTheOracle) {
    const Corpus c = make_corpus(3000, 12, 5);
    const HybridPlanner planner = make_planner(c);
    const std::size_t k = 8;

    for (ColumnKey cut : {ColumnKey{50}, ColumnKey{1500}, ColumnKey{2900}}) {
        for (PredOp op : {PredOp::Lt, PredOp::Le, PredOp::Gt,
                             PredOp::Ge, PredOp::Eq}) {
            Predicate p{"attr", op, Datum{static_cast<std::int64_t>(cut)}};
            const auto got = ids_of(planner.search(p, c.query.data(), k, 400));
            EXPECT_EQ(got, oracle(c, op, cut, k))
                << "op " << static_cast<int>(op) << " cut " << cut;
        }
    }
}

// -------------------------------------------------------- plan choice

TEST(PlanChoice, TightPredicatesPreFilterAndLooseOnesDoNot) {
    const Corpus c = make_corpus(4000, 16, 6);
    const HybridPlanner planner = make_planner(c, /*threshold=*/0.5);

    // Well below the crossover: scanning 40 rows cannot lose to traversing a
    // graph that has to step through 3,960 rejects to stay connected.
    EXPECT_EQ(planner.explain({"attr", PredOp::Lt, 40}, 10).kind,
              PlanKind::PreFilter);

    // Well above it: the graph rejects almost nothing, so its speedup stands.
    EXPECT_EQ(planner.explain({"attr", PredOp::Lt, 3800}, 10).kind,
              PlanKind::FilteredGraph);
}

TEST(PlanChoice, FewerMatchesThanKIsAlwaysAScan) {
    const Corpus c = make_corpus(2000, 8, 7);
    const HybridPlanner planner = make_planner(c, /*threshold=*/0.0);

    // threshold 0 would normally force the graph, but when every match is
    // going into the answer anyway there is nothing for a graph to prune.
    const QueryPlan plan = planner.explain({"attr", PredOp::Lt, 5}, 10);
    EXPECT_EQ(plan.kind, PlanKind::PreFilter);
    EXPECT_NE(plan.reason.find("do not exceed k"), std::string::npos);
}

TEST(PlanChoice, TheThresholdIsWhatMoves) {
    const Corpus c = make_corpus(3000, 8, 8);
    Predicate p{"attr", PredOp::Lt, Datum{static_cast<std::int64_t>(900)}};  // 30%

    EXPECT_EQ(make_planner(c, 0.5).explain(p, 10).kind, PlanKind::PreFilter);
    EXPECT_EQ(make_planner(c, 0.1).explain(p, 10).kind, PlanKind::FilteredGraph);
}

TEST(PlanChoice, ExplainsItself) {
    const Corpus c = make_corpus(2000, 8, 9);
    const HybridPlanner planner = make_planner(c);
    const QueryPlan plan = planner.explain({"attr", PredOp::Lt, 100}, 10);
    // A planner that cannot say why is not defensible in a report.
    EXPECT_FALSE(plan.reason.empty());
    EXPECT_NE(plan.reason.find("crossover"), std::string::npos);
}

// ------------------------------------------------------------ degenerate

TEST(HybridPlanner, APredicateMatchingNothingSkipsTheVectorWorkEntirely) {
    const Corpus c = make_corpus(1000, 8, 10);
    const HybridPlanner planner = make_planner(c);

    QueryPlan plan;
    const auto got = planner.search({"attr", PredOp::Lt, 0}, c.query.data(),
                                    10, 64, &plan);
    EXPECT_TRUE(got.empty());
    EXPECT_EQ(plan.matched_rows, 0u);
    EXPECT_NE(plan.reason.find("no vector search needed"), std::string::npos);
}

TEST(HybridPlanner, KLargerThanTheMatchCountReturnsEveryMatch) {
    const Corpus c = make_corpus(1000, 8, 11);
    const HybridPlanner planner = make_planner(c);
    const auto got = planner.search({"attr", PredOp::Lt, 7}, c.query.data(),
                                    50, 64);
    EXPECT_EQ(got.size(), 7u);
}

TEST(HybridPlanner, KOfZeroReturnsNothingRatherThanEverything) {
    const Corpus c = make_corpus(500, 8, 12);
    const HybridPlanner planner = make_planner(c);
    EXPECT_TRUE(planner.search({"attr", PredOp::Lt, 100}, c.query.data(), 0).empty());
}

TEST(HybridPlanner, AnUnknownColumnSaysWhichOnesExist) {
    const Corpus c = make_corpus(500, 8, 13);
    const HybridPlanner planner = make_planner(c);
    try {
        planner.matching_rows({"nosuch", PredOp::Lt, 10});
        FAIL() << "an unknown column should not be silently empty";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("attr"), std::string::npos)
            << "the error should list the columns that do exist";
    }
}

TEST(HybridPlanner, WorksWithNoGraphAttached) {
    const Corpus c = make_corpus(1500, 8, 14);
    HybridPlanner planner(0.5);
    planner.set_column("attr", attribute_column(c));
    planner.set_exact(&c.exact);
    // No graph: every plan must fall back to the exact index rather than
    // dereferencing a null one.
    const QueryPlan plan = planner.explain({"attr", PredOp::Lt, 1400}, 10);
    EXPECT_EQ(plan.kind, PlanKind::PreFilter);
    EXPECT_EQ(planner.search({"attr", PredOp::Lt, 1400}, c.query.data(), 10).size(),
              10u);
}

TEST(HybridPlanner, AsksForAGraphPlanWithoutAGraphAndIsTold) {
    const Corpus c = make_corpus(500, 8, 15);
    HybridPlanner planner(0.5);
    planner.set_column("attr", attribute_column(c));
    planner.set_exact(&c.exact);
    EXPECT_THROW(planner.search_with(PlanKind::FilteredGraph,
                                     {"attr", PredOp::Lt, 400},
                                     c.query.data(), 10),
                 std::logic_error);
}

TEST(HybridPlanner, HoldsAnyIndexKindBehindTheSameInterface) {
    // The CompareOp contract, paying off one level up: the planner never
    // learns whether a B+ tree, an RMI or a dynamic RMI answered.
    const Corpus c = make_corpus(2000, 8, 16);

    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    for (std::size_t row = 0; row < c.n; ++row) {
        keys.push_back(static_cast<ColumnKey>(row));
        values.push_back(static_cast<ColumnValue>(row));
    }

    std::vector<std::int64_t> answers[3];
    const hylis::index::IndexKind kinds[3] = {
        hylis::index::IndexKind::BPlusTree,
        hylis::index::IndexKind::RMI,
        hylis::index::IndexKind::DynamicRMI,
    };
    for (int i = 0; i < 3; ++i) {
        hylis::index::IndexPlan plan;
        plan.kind = kinds[i];
        plan.rmi_models = 64;

        HybridPlanner planner(0.5);
        planner.set_column("attr", ColumnIndex::build_with(keys, values, plan));
        planner.set_exact(&c.exact);
        planner.set_graph(&c.graph);
        answers[i] = ids_of(planner.search({"attr", PredOp::Lt, 500},
                                           c.query.data(), 10, 400));
    }
    EXPECT_EQ(answers[0], answers[1]) << "B+ tree and RMI gave different rows";
    EXPECT_EQ(answers[0], answers[2]) << "B+ tree and dynamic RMI gave different rows";
    EXPECT_EQ(answers[0].size(), 10u);
}

// ----------------------------------------------------------- usefulness

// Everything above tests that the planner is *correct*. This tests that it is
// *useful*, which is the actual point of the module and a different question.
TEST(PlannerUsefulness, ChoosesTheFasterPlanAtBothEnds) {
    const Corpus c = make_corpus(20000, 16, 17);
    const HybridPlanner planner = make_planner(c);
    const std::size_t k = 10;

    auto time_plan = [&](PlanKind kind, ColumnKey cut) {
        Predicate p{"attr", kind == PlanKind::PostFilter ? PredOp::Lt
                                                         : PredOp::Lt,
                    Datum{static_cast<std::int64_t>(cut)}};
        // Warm, then measure: the first call pays for cold caches on
        // whichever plan happens to run first.
        planner.search_with(kind, p, c.query.data(), k, 64);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) planner.search_with(kind, p, c.query.data(), k, 64);
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start).count();
    };

    // Tight predicate: 0.5% of rows. The scan should win, and the planner
    // should have said so.
    const ColumnKey tight = 100;
    const double tight_scan = time_plan(PlanKind::PreFilter, tight);
    const double tight_graph = time_plan(PlanKind::FilteredGraph, tight);
    EXPECT_LT(tight_scan, tight_graph)
        << "the measured crossover says a scan wins at 0.5% selectivity; if "
           "this fails the planner's threshold is calibrated to the wrong "
           "machine, not merely unlucky";
    EXPECT_EQ(planner.explain({"attr", PredOp::Lt, tight}, k).kind,
              PlanKind::PreFilter);

    // Loose predicate: 95% of rows. The graph should win.
    const ColumnKey loose = 19000;
    const double loose_scan = time_plan(PlanKind::PreFilter, loose);
    const double loose_graph = time_plan(PlanKind::FilteredGraph, loose);
    EXPECT_LT(loose_graph, loose_scan)
        << "the graph lost even at 95% selectivity, where it rejects almost "
           "nothing — that would mean the graph is never worth choosing";
    EXPECT_EQ(planner.explain({"attr", PredOp::Lt, loose}, k).kind,
              PlanKind::FilteredGraph);
}

// ---------------------------------------------------------------------------
// The bitmap-filtered graph plan
// ---------------------------------------------------------------------------
//
// Two claims, and only the second is about speed.
//
// **The rows are the same.** A plan that returned different rows would not be
// a cheaper way of answering the query, it would be a different query.
//
// **The selectivity was free.** planner.hpp states its own weakness at the
// top: it knows selectivity exactly because it *executes* the predicate first,
// so "a predicate matching nearly everything is paid for in full before the
// planner can discover it should have post-filtered". A bitmap column answers
// by popcount and materialises nothing, and `selectivity_was_free` is how a
// caller can tell that case apart.

namespace {

// A bitmap column over a low-cardinality attribute, built over the corpus's
// whole row space so its bit positions are the vector index's row ids.
ColumnIndex bucket_column(const Corpus& c, std::size_t buckets) {
    std::vector<std::pair<ColumnKey, ColumnValue>> pairs;
    for (std::size_t row = 0; row < c.n; ++row) {
        pairs.emplace_back(
            static_cast<ColumnKey>(c.attribute[row] % static_cast<ColumnKey>(buckets)),
            static_cast<ColumnValue>(row));
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> rows;
    std::vector<ColumnValue> space;
    for (const auto& [value, row] : pairs) {
        keys.push_back(value);
        rows.push_back(row);
    }
    for (std::size_t row = 0; row < c.n; ++row) {
        space.push_back(static_cast<ColumnValue>(row));
    }

    hylis::index::IndexPlan plan;
    plan.kind = hylis::index::IndexKind::Bitmap;
    plan.type = hylis::index::LogicalType::Int64;
    plan.encoding = hylis::index::KeyEncoding::Dictionary;
    return ColumnIndex::build_typed_with(hylis::index::LogicalType::Int64, keys,
                                         rows, plan, &space);
}

}  // namespace

TEST(BitmapPlan, SelectivityIsFreeAndThePlanSaysSo) {
    Corpus c = make_corpus(4000, 16, 11);
    HybridPlanner planner(0.2);
    planner.set_column("bucket", bucket_column(c, 10));
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);

    // 8 of 10 buckets: 80% selectivity, well above the crossover.
    const Predicate loose{"bucket", PredOp::Lt, Datum{static_cast<std::int64_t>(8)}};
    const QueryPlan plan = planner.explain(loose, 10);

    EXPECT_TRUE(plan.selectivity_was_free);
    EXPECT_EQ(plan.kind, PlanKind::BitmapFilteredGraph);
    EXPECT_NEAR(plan.selectivity, 0.8, 0.02);
    EXPECT_NE(plan.reason.find("no row id is materialised"), std::string::npos)
        << plan.reason;
}

TEST(BitmapPlan, AnOrderedColumnHasToExecuteToKnowItsSelectivity) {
    // The contrast that makes the flag mean something.
    Corpus c = make_corpus(4000, 16, 11);
    HybridPlanner planner(0.2);
    planner.set_column("attr", attribute_column(c));
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);

    const Predicate loose{"attr", PredOp::Lt,
                          Datum{static_cast<std::int64_t>(3200)}};
    const QueryPlan plan = planner.explain(loose, 10);
    EXPECT_FALSE(plan.selectivity_was_free);
    EXPECT_EQ(plan.kind, PlanKind::FilteredGraph);
}

TEST(BitmapPlan, TheRowsMatchTheIdListPlan) {
    Corpus c = make_corpus(4000, 16, 12);
    HybridPlanner planner(0.2);
    planner.set_column("bucket", bucket_column(c, 10));
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);

    for (std::int64_t cut : {1, 4, 8, 10}) {
        const Predicate p{"bucket", PredOp::Lt, Datum{cut}};
        const auto masked =
            planner.search_with(PlanKind::BitmapFilteredGraph, p, c.query.data(), 10, 64);
        const auto listed =
            planner.search_with(PlanKind::FilteredGraph, p, c.query.data(), 10, 64);
        EXPECT_EQ(ids_of(masked), ids_of(listed)) << "cut " << cut;
    }
}

TEST(BitmapPlan, EveryPlanStillAgreesWithTheOracle) {
    Corpus c = make_corpus(3000, 16, 13);
    HybridPlanner planner(0.2);
    planner.set_column("bucket", bucket_column(c, 8));
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);

    for (std::int64_t cut : {2, 6}) {
        const Predicate p{"bucket", PredOp::Lt, Datum{cut}};
        std::vector<std::int64_t> allowed;
        for (std::size_t row = 0; row < c.n; ++row) {
            if (c.attribute[row] % 8 < cut) {
                allowed.push_back(static_cast<std::int64_t>(row));
            }
        }
        const auto want = ids_of(c.exact.search_filtered(c.query.data(), 10, allowed));
        const auto got =
            planner.search_with(PlanKind::PreFilter, p, c.query.data(), 10, 64);
        EXPECT_EQ(ids_of(got), want) << "cut " << cut;
    }
}

TEST(BitmapPlan, AnOrderedColumnCannotRunItAndSaysSo) {
    // The bit set has to line up with the vector index's row ids. A tree has
    // none to give, and answering with a silently different filter would be
    // far worse than refusing.
    Corpus c = make_corpus(1000, 8, 14);
    HybridPlanner planner(0.2);
    planner.set_column("attr", attribute_column(c));
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);

    const Predicate p{"attr", PredOp::Lt, Datum{static_cast<std::int64_t>(500)}};
    EXPECT_FALSE(planner.plan_available(PlanKind::BitmapFilteredGraph, p));
    EXPECT_THROW(
        planner.search_with(PlanKind::BitmapFilteredGraph, p, c.query.data(), 10),
        std::invalid_argument);
    EXPECT_TRUE(planner.plan_available(PlanKind::FilteredGraph, p));
}

TEST(BitmapPlan, RowsSuppliedByTheCallerAreStillPlanned) {
    // The seam for everything the planner cannot express: a conjunction Table
    // resolved, or a Contains it had to scan for.
    Corpus c = make_corpus(2000, 16, 15);
    HybridPlanner planner(0.2);
    planner.set_exact(&c.exact);
    planner.set_graph(&c.graph);

    std::vector<std::int64_t> few;
    for (std::size_t row = 0; row < 40; ++row) {
        few.push_back(static_cast<std::int64_t>(row));
    }
    QueryPlan plan;
    const auto got = planner.search_rows(few, c.query.data(), 10, 64, &plan);
    EXPECT_EQ(plan.kind, PlanKind::PreFilter) << plan.reason;
    EXPECT_EQ(ids_of(got),
              ids_of(c.exact.search_filtered(c.query.data(), 10, few)));
}

TEST(BitmapPlan, MaskedSearchMatchesTheIdListOnTheGraphItself) {
    // One level down: the two HnswIndex entry points must agree before any
    // plan built on them can.
    Corpus c = make_corpus(2000, 16, 16);
    std::vector<std::int64_t> allowed;
    hylis::index::Bitset mask(c.n);
    for (std::size_t row = 0; row < c.n; row += 3) {
        allowed.push_back(static_cast<std::int64_t>(row));
        mask.set(row);
    }

    for (std::size_t ef : {16u, 64u}) {
        const auto listed = c.graph.search_filtered(c.query.data(), 10, allowed, ef);
        const auto masked = c.graph.search_masked(c.query.data(), 10, mask, ef);
        EXPECT_EQ(ids_of(listed), ids_of(masked)) << "ef " << ef;
    }

    const auto flat_listed = c.exact.search_filtered(c.query.data(), 10, allowed);
    const auto flat_masked = c.exact.search_masked(c.query.data(), 10, mask);
    EXPECT_EQ(ids_of(flat_listed), ids_of(flat_masked));
}
