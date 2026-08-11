// Tests for per-column index selection and the persisted catalog.
//
// The property that makes this layer worth having is *indistinguishability*:
// a caller must get identical answers whichever structure was chosen, or the
// query planner would be choosing between different results rather than
// different costs. Most of what follows is that claim in various forms.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "index/column_index.hpp"
#include "index/index_catalog.hpp"

using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::ColumnValue;
using hylis::index::CompareOp;
using hylis::index::IndexCatalog;
using hylis::index::IndexKind;
using hylis::index::IndexPlan;
using hylis::index::choose_index;

namespace {

std::vector<ColumnKey> sequential_keys(std::size_t n, unsigned seed = 1) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<ColumnKey> gap(1, 7);
    std::vector<ColumnKey> keys;
    ColumnKey k = 0;
    for (std::size_t i = 0; i < n; ++i) keys.push_back(k += gap(rng));
    return keys;
}

std::vector<ColumnKey> clustered_keys(std::size_t n, unsigned seed = 2) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<ColumnKey> centre(0, ColumnKey{1} << 40);
    std::uniform_int_distribution<ColumnKey> offset(0, 10000);
    std::vector<ColumnKey> centres;
    for (int i = 0; i < 64; ++i) centres.push_back(centre(rng));
    std::uniform_int_distribution<std::size_t> pick(0, centres.size() - 1);

    std::vector<ColumnKey> keys;
    for (std::size_t i = 0; i < n * 2; ++i) keys.push_back(centres[pick(rng)] + offset(rng));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() > n) keys.resize(n);
    return keys;
}

std::vector<ColumnValue> values_for(const std::vector<ColumnKey>& keys) {
    std::vector<ColumnValue> v(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) v[i] = static_cast<ColumnValue>(i) * 10;
    return v;
}

std::string temp_path(const char* name) {
    return std::string(HYLIS_TEST_TMP) + "/" + name;
}

}  // namespace

// --------------------------------------------------------------------------
// Indistinguishability
// --------------------------------------------------------------------------

TEST(ColumnIndex, AnsweringIsIdenticalWhicheverStructureWasChosen) {
    const auto keys = sequential_keys(4000);
    const auto vals = values_for(keys);

    IndexPlan force_tree;
    force_tree.kind = IndexKind::BPlusTree;
    IndexPlan force_rmi;
    force_rmi.kind = IndexKind::RMI;
    force_rmi.rmi_models = 256;

    const ColumnIndex tree = ColumnIndex::build_with(keys, vals, force_tree);
    const ColumnIndex rmi = ColumnIndex::build_with(keys, vals, force_rmi);
    ASSERT_EQ(tree.kind(), IndexKind::BPlusTree);
    ASSERT_EQ(rmi.kind(), IndexKind::RMI);

    std::mt19937_64 rng(7);
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
    for (int t = 0; t < 300; ++t) {
        const ColumnKey v = (t % 2) ? keys[pick(rng)] : keys[pick(rng)] + 3;

        const ColumnValue* a = tree.find(v);
        const ColumnValue* b = rmi.find(v);
        ASSERT_EQ(a == nullptr, b == nullptr) << "v=" << v;
        if (a != nullptr) { EXPECT_EQ(*a, *b); }

        for (CompareOp op : {CompareOp::Lt, CompareOp::Le, CompareOp::Gt,
                             CompareOp::Ge, CompareOp::Eq}) {
            EXPECT_EQ(tree.range_query(op, v), rmi.range_query(op, v))
                << "op=" << static_cast<int>(op) << " v=" << v;
        }
    }
}

TEST(ColumnIndex, AutoTunedIndexMatchesAMapReference) {
    const auto keys = clustered_keys(3000);
    const auto vals = values_for(keys);
    const ColumnIndex index = ColumnIndex::build(keys, vals);
    ASSERT_NO_THROW(index.validate());

    std::map<ColumnKey, ColumnValue> ref;
    for (std::size_t i = 0; i < keys.size(); ++i) ref[keys[i]] = vals[i];

    EXPECT_EQ(index.size(), ref.size());
    for (const auto& [k, v] : ref) {
        const ColumnValue* got = index.find(k);
        ASSERT_NE(got, nullptr) << "k=" << k;
        EXPECT_EQ(*got, v);
    }
    EXPECT_EQ(index.find(keys.back() + 1), nullptr);
}

TEST(ColumnIndex, EmptyColumnIsUsable) {
    const ColumnIndex index = ColumnIndex::build({}, {});
    EXPECT_EQ(index.size(), 0u);
    EXPECT_TRUE(index.empty());
    EXPECT_EQ(index.find(1), nullptr);
    EXPECT_TRUE(index.range_query(CompareOp::Ge, 0).empty());
    ASSERT_NO_THROW(index.validate());
}

TEST(ColumnIndex, MismatchedKeyValueLengthsRejected) {
    EXPECT_THROW(ColumnIndex::build({1, 2, 3}, {1, 2}), std::invalid_argument);
}

// --------------------------------------------------------------------------
// Selection
// --------------------------------------------------------------------------

TEST(ChooseIndex, RecordsTheEvidenceForItsChoice) {
    const auto keys = sequential_keys(20000);
    const IndexPlan plan = choose_index(keys, values_for(keys));

    EXPECT_GT(plan.ns_per_lookup, 0.0) << "the choice must be measured, not assumed";
    EXPECT_GT(plan.index_bytes, 0u);
    EXPECT_EQ(plan.n_keys, keys.size());
    EXPECT_EQ(plan.key_min, keys.front());
    EXPECT_EQ(plan.key_max, keys.back());
}

TEST(ChooseIndex, ModelCountAdaptsToColumnSize) {
    // Selection must be a function of the data, not a constant. Column size
    // is the axis where that shows up in the *structure*: a 200-key column
    // has no use for 16384 models, and the candidate list reflects that.
    const auto tiny = sequential_keys(200);
    const auto big = sequential_keys(30000);

    const IndexPlan small_plan = choose_index(tiny, values_for(tiny));
    const IndexPlan big_plan = choose_index(big, values_for(big));

    EXPECT_LT(small_plan.rmi_models, big_plan.rmi_models)
        << "a tiny column should not be given the same model budget as a large one";
    EXPECT_LT(small_plan.index_bytes, big_plan.index_bytes);
}

TEST(ChooseIndex, EvidenceDiffersSharplyBetweenEasyAndHardColumns) {
    // At equal size, an easy and a hard distribution may well land on the
    // same *structure* — but the measurements behind that choice must not be
    // interchangeable, or the plan would not be evidence about anything.
    //
    // Worth recording what this actually shows: on static data the learned
    // index wins across every distribution generated here, and the B+ tree's
    // real claim is mutability rather than lookup speed. The interesting
    // per-column variable turns out to be *how much* it wins by.
    const auto easy = sequential_keys(30000);
    const auto hard = clustered_keys(30000);

    const IndexPlan a = choose_index(easy, values_for(easy));
    const IndexPlan b = choose_index(hard, values_for(hard));

    EXPECT_LT(a.max_error, b.max_error)
        << "a near-linear CDF must fit better than a stepped one";
    EXPECT_LT(a.ns_per_lookup, b.ns_per_lookup)
        << "and that better fit must show up as a faster measured lookup";
}

TEST(ChooseIndex, RespectsASizeBudget) {
    const auto keys = sequential_keys(20000);
    const auto vals = values_for(keys);

    const IndexPlan unconstrained = choose_index(keys, vals);
    ASSERT_GT(unconstrained.index_bytes, 0u);

    const std::size_t budget = unconstrained.index_bytes - 1;
    const IndexPlan constrained = choose_index(keys, vals, budget);

    // Either something smaller fit, or nothing did and it fell back to a tree.
    const bool honoured = constrained.index_bytes <= budget ||
                          constrained.kind == IndexKind::BPlusTree;
    EXPECT_TRUE(honoured) << "budget " << budget << " but chose "
                          << constrained.index_bytes << " bytes";
}

TEST(ChooseIndex, ImpossibleBudgetStillProducesAWorkingIndex) {
    const auto keys = sequential_keys(2000);
    const auto vals = values_for(keys);
    const ColumnIndex index = ColumnIndex::build(keys, vals, /*size_budget=*/1);

    ASSERT_NO_THROW(index.validate());
    for (std::size_t i = 0; i < keys.size(); i += 41) {
        ASSERT_NE(index.find(keys[i]), nullptr) << "i=" << i;
    }
}

TEST(ChooseIndex, SkipsModelCountsLargerThanTheColumn) {
    const auto keys = sequential_keys(100);
    const IndexPlan plan = choose_index(keys, values_for(keys));
    if (plan.kind == IndexKind::RMI) {
        EXPECT_LE(plan.rmi_models, keys.size());
    }
}

// --------------------------------------------------------------------------
// Plans as evidence: fingerprints and staleness
// --------------------------------------------------------------------------

TEST(IndexPlan, MatchesOnlyTheDataItWasChosenFor) {
    const auto keys = sequential_keys(500);
    ASSERT_FALSE(keys.empty());
    const IndexPlan plan = choose_index(keys, values_for(keys));

    EXPECT_TRUE(plan.matches(keys));

    std::vector<ColumnKey> shorter(keys.begin(), keys.end() - 1);
    EXPECT_FALSE(plan.matches(shorter)) << "a different length must not match";

    std::vector<ColumnKey> shifted = keys;
    shifted[shifted.size() - 1] += 1;
    EXPECT_FALSE(plan.matches(shifted)) << "a different last key must not match";

    EXPECT_FALSE(plan.matches({})) << "an empty column must not match a full plan";
}

TEST(ColumnIndex, ReplayingAPlanRefreshesItsMeasurementsForTheNewData) {
    const auto small = sequential_keys(500, 3);
    IndexPlan plan = choose_index(small, values_for(small));

    const auto large = sequential_keys(5000, 3);
    const ColumnIndex index = ColumnIndex::build_with(large, values_for(large), plan);

    EXPECT_EQ(index.plan().kind, plan.kind) << "the structural choice is obeyed";
    EXPECT_EQ(index.plan().n_keys, large.size()) << "but the fingerprint is refreshed";
    EXPECT_EQ(index.plan().key_max, large.back());
    ASSERT_NO_THROW(index.validate());
}

TEST(ColumnIndex, AStalePlanStillProducesACorrectIndex) {
    // The safety property the whole persistence story rests on: a plan chosen
    // for entirely different data can only make the index slower, never wrong.
    const auto other = clustered_keys(1000, 11);
    IndexPlan wrong = choose_index(other, values_for(other));
    wrong.rmi_models = 3;  // and deliberately mangle it further

    const auto keys = sequential_keys(4000);
    const auto vals = values_for(keys);
    ASSERT_FALSE(wrong.matches(keys));

    const ColumnIndex index = ColumnIndex::build_with(keys, vals, wrong);
    ASSERT_NO_THROW(index.validate());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const ColumnValue* got = index.find(keys[i]);
        ASSERT_NE(got, nullptr) << "i=" << i;
        EXPECT_EQ(*got, vals[i]);
    }
}

// --------------------------------------------------------------------------
// Catalog
// --------------------------------------------------------------------------

TEST(IndexCatalog, RoundTripsThroughJson) {
    IndexCatalog catalog;

    IndexPlan a;
    a.kind = IndexKind::RMI;
    a.rmi_models = 16384;
    a.search_threshold = 32;
    a.ns_per_lookup = 123.456;
    a.max_error = 77;
    a.index_bytes = 4242;
    a.n_keys = 1000;
    a.key_min = -50;
    a.key_max = 99999;
    catalog.set("price", a);

    IndexPlan b;
    b.kind = IndexKind::BPlusTree;
    b.btree_order = 64;
    b.n_keys = 7;
    catalog.set("user_id", b);

    const IndexCatalog back = IndexCatalog::parse(catalog.serialize());
    ASSERT_EQ(back.size(), 2u);

    const auto price = back.get("price");
    ASSERT_TRUE(price.has_value());
    EXPECT_EQ(price->kind, IndexKind::RMI);
    EXPECT_EQ(price->rmi_models, 16384u);
    EXPECT_EQ(price->search_threshold, 32u);
    EXPECT_NEAR(price->ns_per_lookup, 123.456, 1e-3);
    EXPECT_EQ(price->max_error, 77u);
    EXPECT_EQ(price->index_bytes, 4242u);
    EXPECT_EQ(price->n_keys, 1000u);
    EXPECT_EQ(price->key_min, -50);
    EXPECT_EQ(price->key_max, 99999);

    const auto user = back.get("user_id");
    ASSERT_TRUE(user.has_value());
    EXPECT_EQ(user->kind, IndexKind::BPlusTree);
    EXPECT_EQ(user->btree_order, 64u);
}

TEST(IndexCatalog, EmptyCatalogRoundTrips) {
    const IndexCatalog back = IndexCatalog::parse(IndexCatalog{}.serialize());
    EXPECT_TRUE(back.empty());
}

TEST(IndexCatalog, ColumnNamesWithAwkwardCharactersSurvive) {
    IndexCatalog catalog;
    catalog.set("a\"quoted\"\\name\nwith\tcontrol", IndexPlan{});
    const IndexCatalog back = IndexCatalog::parse(catalog.serialize());
    EXPECT_TRUE(back.get("a\"quoted\"\\name\nwith\tcontrol").has_value());
}

TEST(IndexCatalog, UnknownFieldsAreIgnoredNotRejected) {
    // Forward compatibility: a catalog written by a later build must still load.
    const std::string blob =
        R"({"version":1,"future":"stuff","plans":[)"
        R"({"column":"c","kind":"rmi","rmi_models":64,"invented_field":123}]})";
    const IndexCatalog back = IndexCatalog::parse(blob);
    const auto plan = back.get("c");
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->kind, IndexKind::RMI);
    EXPECT_EQ(plan->rmi_models, 64u);
}

TEST(IndexCatalog, SavesAndLoadsFromDisk) {
    const std::string path = temp_path("catalog_roundtrip.json");
    std::remove(path.c_str());

    IndexCatalog catalog;
    IndexPlan plan;
    plan.kind = IndexKind::RMI;
    plan.rmi_models = 1024;
    plan.n_keys = 12345;
    plan.key_min = 1;
    plan.key_max = 999;
    catalog.set("price", plan);
    catalog.save(path);

    const IndexCatalog loaded = IndexCatalog::load(path);
    ASSERT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded.get("price")->rmi_models, 1024u);
    EXPECT_EQ(loaded.get("price")->n_keys, 12345u);
}

TEST(IndexCatalog, LoadingAMissingFileIsNotAnError) {
    const IndexCatalog catalog = IndexCatalog::load(temp_path("definitely_absent.json"));
    EXPECT_TRUE(catalog.empty());
}

TEST(IndexCatalog, LoadingACorruptFileThrows) {
    // A corrupt catalog silently discarded would show up only as an
    // unexplained slowdown; better to say so.
    const std::string path = temp_path("catalog_corrupt.json");
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("{\"plans\":[{\"column\":\"x\",\"kind\":", f);
    std::fclose(f);

    EXPECT_THROW(IndexCatalog::load(path), std::runtime_error);
}

TEST(IndexCatalog, SaveIsAtomicAndLeavesNoTempBehind) {
    const std::string path = temp_path("catalog_atomic.json");
    IndexCatalog catalog;
    catalog.set("a", IndexPlan{});
    catalog.save(path);
    catalog.set("b", IndexPlan{});
    catalog.save(path);  // overwrite an existing file

    EXPECT_EQ(IndexCatalog::load(path).size(), 2u);
    std::FILE* leftover = std::fopen((path + ".tmp").c_str(), "rb");
    EXPECT_EQ(leftover, nullptr) << "the temp file should have been renamed away";
    if (leftover) std::fclose(leftover);
}

TEST(IndexCatalog, ReportsFreshnessAgainstTheActualColumn) {
    const auto keys = sequential_keys(1000);
    IndexCatalog catalog;

    EXPECT_EQ(catalog.freshness("price", keys), IndexCatalog::Freshness::Missing);

    catalog.set("price", choose_index(keys, values_for(keys)));
    EXPECT_EQ(catalog.freshness("price", keys), IndexCatalog::Freshness::Fresh);

    auto changed = keys;
    changed.push_back(changed.back() + 1000);
    EXPECT_EQ(catalog.freshness("price", changed), IndexCatalog::Freshness::Stale);
}

TEST(IndexCatalog, BuildColumnReplaysAFreshPlanAndRetunesAStaleOne) {
    const auto keys = sequential_keys(5000);
    const auto vals = values_for(keys);

    IndexCatalog catalog;
    const ColumnIndex first = catalog.build_column("price", keys, vals);
    ASSERT_EQ(catalog.freshness("price", keys), IndexCatalog::Freshness::Fresh);

    // Replayed: same structural decision, still correct.
    const ColumnIndex second = catalog.build_column("price", keys, vals);
    EXPECT_EQ(second.plan().kind, first.plan().kind);
    EXPECT_EQ(second.plan().rmi_models, first.plan().rmi_models);
    ASSERT_NO_THROW(second.validate());

    // Different data: the plan is no longer evidence, so it re-tunes and the
    // catalog ends up describing what was actually built.
    const auto other = clustered_keys(5000);
    const ColumnIndex third = catalog.build_column("price", other, values_for(other));
    ASSERT_NO_THROW(third.validate());
    EXPECT_EQ(catalog.freshness("price", other), IndexCatalog::Freshness::Fresh);
    EXPECT_EQ(catalog.get("price")->key_max, other.back());
}

TEST(IndexCatalog, EraseAndColumns) {
    IndexCatalog catalog;
    catalog.set("a", IndexPlan{});
    catalog.set("b", IndexPlan{});
    EXPECT_EQ(catalog.columns(), (std::vector<std::string>{"a", "b"}));
    EXPECT_TRUE(catalog.erase("a"));
    EXPECT_FALSE(catalog.erase("a"));
    EXPECT_EQ(catalog.size(), 1u);
    catalog.clear();
    EXPECT_TRUE(catalog.empty());
}

// ------------------------------------------------- workload-aware selection

// The reason Workload exists. On lookups alone the static RMI wins nearly
// everything, and it is build-only — so without the write rate the selector
// would hand an immutable structure to a column that is about to be written
// to, and the first insert would throw.
TEST(WorkloadAwareSelection, NeverOffersAnImmutableIndexToAWrittenColumn) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    for (int i = 0; i < 50000; ++i) {
        keys.push_back(static_cast<ColumnKey>(i) * 3);
        values.push_back(i);
    }

    hylis::index::Workload read_only;
    const IndexPlan cold = choose_index(keys, values, SIZE_MAX, read_only);
    EXPECT_EQ(cold.kind, IndexKind::RMI)
        << "a read-only column should still get the fastest thing available";

    for (double fraction : {0.01, 0.5, 1.0}) {
        hylis::index::Workload writes;
        writes.write_fraction = fraction;
        const IndexPlan plan = choose_index(keys, values, SIZE_MAX, writes);
        EXPECT_TRUE(hylis::index::is_mutable(plan.kind))
            << "chose " << hylis::index::to_string(plan.kind)
            << " for a column with write_fraction " << fraction;
        EXPECT_GT(plan.ns_per_write, 0.0) << "writes were not actually timed";
    }
}

TEST(WorkloadAwareSelection, AStaticRMIRefusesWritesClearly) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    for (int i = 0; i < 1000; ++i) {
        keys.push_back(static_cast<ColumnKey>(i) * 3);
        values.push_back(i);
    }
    IndexPlan plan;
    plan.kind = IndexKind::RMI;
    ColumnIndex index = ColumnIndex::build_with(keys, values, plan);

    EXPECT_FALSE(index.is_mutable());
    // Not "returns false" — that would read as "the key was already there".
    EXPECT_THROW(index.insert(7, 7), std::logic_error);
    EXPECT_THROW(index.erase(0), std::logic_error);
}

TEST(WorkloadAwareSelection, TheDynamicIndexAnswersLikeEveryOtherKind) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    std::map<ColumnKey, ColumnValue> oracle;
    for (int i = 0; i < 20000; ++i) {
        keys.push_back(static_cast<ColumnKey>(i) * 7);
        values.push_back(i * 2);
        oracle[keys.back()] = values.back();
    }

    IndexPlan plan;
    plan.kind = IndexKind::DynamicRMI;
    plan.rmi_models = 256;
    ColumnIndex index = ColumnIndex::build_with(keys, values, plan);

    ASSERT_TRUE(index.is_mutable());
    EXPECT_TRUE(index.insert(5, 999));
    oracle[5] = 999;
    EXPECT_TRUE(index.erase(keys[100]));
    oracle.erase(keys[100]);

    for (const auto& [key, value] : oracle) {
        const ColumnValue* got = index.find(key);
        ASSERT_NE(got, nullptr) << key;
        EXPECT_EQ(*got, value);
    }
    EXPECT_EQ(index.find(keys[100]), nullptr);
    EXPECT_EQ(index.size(), oracle.size());
    EXPECT_NO_THROW(index.validate());
}

// ------------------------------------------------------- the retune policy

// Re-tuning means rebuilding and timing every candidate. Any change to a
// column makes its plan stale, and most changes do not make it *wrong*, so
// re-tuning on staleness alone spends the expensive path on columns that
// would have reached the same answer.
TEST(RetunePolicy, ATrivialChangeReplaysRatherThanRetuning) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    for (int i = 0; i < 50000; ++i) {
        keys.push_back(static_cast<ColumnKey>(i) * 3);
        values.push_back(i);
    }

    IndexCatalog catalog;
    IndexCatalog::Decision decision;
    catalog.build_column("c", keys, values, hylis::index::Workload{}, &decision);
    EXPECT_EQ(decision.action, IndexCatalog::Action::Chosen);
    EXPECT_EQ(decision.freshness, IndexCatalog::Freshness::Missing);

    catalog.build_column("c", keys, values, hylis::index::Workload{}, &decision);
    EXPECT_EQ(decision.action, IndexCatalog::Action::Replayed);
    EXPECT_EQ(decision.freshness, IndexCatalog::Freshness::Fresh);

    keys.pop_back();
    values.pop_back();
    catalog.build_column("c", keys, values, hylis::index::Workload{}, &decision);
    EXPECT_EQ(decision.freshness, IndexCatalog::Freshness::Stale)
        << "one row fewer is a different column by the fingerprint";
    EXPECT_EQ(decision.action, IndexCatalog::Action::Replayed)
        << "but not a different *answer*, so it should not have re-tuned";
    EXPECT_GT(decision.measured_ns, 0.0) << "it should still have been re-timed";
}

TEST(RetunePolicy, RealDegradationDoesTriggerARetune) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    for (int i = 0; i < 20000; ++i) {
        keys.push_back(static_cast<ColumnKey>(i) * 3);
        values.push_back(i);
    }

    IndexCatalog catalog;
    catalog.build_column("c", keys, values);

    // An impossible bar: anything at all counts as degraded. Checks the
    // branch is reachable, rather than that any particular workload trips it.
    catalog.set_retune_threshold(0.0);
    keys.pop_back();
    values.pop_back();

    IndexCatalog::Decision decision;
    catalog.build_column("c", keys, values, hylis::index::Workload{}, &decision);
    EXPECT_EQ(decision.action, IndexCatalog::Action::Retuned);
}

TEST(RetunePolicy, AChangeOfWorkloadIsNotSomethingAPlanCanSurvive) {
    std::vector<ColumnKey> keys;
    std::vector<ColumnValue> values;
    for (int i = 0; i < 20000; ++i) {
        keys.push_back(static_cast<ColumnKey>(i) * 3);
        values.push_back(i);
    }

    IndexCatalog catalog;
    ColumnIndex cold = catalog.build_column("c", keys, values);
    ASSERT_EQ(cold.kind(), IndexKind::RMI);

    // Same keys, so the fingerprint still matches — but the plan is now for
    // the wrong job, and matching data does not make an immutable structure
    // able to take writes.
    hylis::index::Workload writes;
    writes.write_fraction = 0.5;
    IndexCatalog::Decision decision;
    ColumnIndex warm =
        catalog.build_column("c", keys, values, writes, &decision);
    EXPECT_EQ(decision.action, IndexCatalog::Action::Chosen);
    EXPECT_TRUE(warm.is_mutable());
    EXPECT_NO_THROW(warm.insert(1, 1));
}

TEST(RetunePolicy, TheNewPlanFieldsSurviveARoundTrip) {
    IndexPlan plan;
    plan.kind = IndexKind::DynamicRMI;
    plan.rmi_models = 512;
    plan.merge_ratio = 0.02;
    plan.write_fraction = 0.25;
    plan.ns_per_lookup = 41.5;
    plan.ns_per_write = 250.75;
    plan.n_keys = 1234;

    IndexCatalog catalog;
    catalog.set("c", plan);
    const IndexCatalog reloaded = IndexCatalog::parse(catalog.serialize());
    const IndexPlan back = *reloaded.get("c");

    EXPECT_EQ(back.kind, IndexKind::DynamicRMI);
    EXPECT_EQ(back.rmi_models, 512u);
    EXPECT_NEAR(back.merge_ratio, 0.02, 1e-9);
    EXPECT_NEAR(back.write_fraction, 0.25, 1e-9);
    EXPECT_NEAR(back.ns_per_lookup, 41.5, 1e-3);
    EXPECT_NEAR(back.ns_per_write, 250.75, 1e-3);
    // Without write_fraction round-tripping, every reloaded plan would look
    // like it was chosen for a different workload and be re-tuned on sight.
    EXPECT_EQ(back.n_keys, 1234u);
}
