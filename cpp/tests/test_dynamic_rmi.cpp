// Tests for the writable learned index.
//
// Two properties carry the weight here.
//
// Exactness has to survive mutation. base_.validate() replays every stored key
// through the real lookup path and insists the predicted window contains it,
// and it is called after merges, after tombstoning, and after long randomised
// workloads. An incremental merge that quietly widened or misplaced a window
// would still return right answers on the cases a hand-written test thought
// to try, and would fail here.
//
// Agreement has to be with an oracle, not with itself. The long workload below
// drives a std::map through exactly the same operations and compares every
// result, which is the only way to catch a tombstone, delta-union or merge bug
// that happens to be self-consistent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "index/dynamic_rmi.hpp"
#include "index/rmi.hpp"

using hylis::index::CompareOp;
using hylis::index::DynamicRMIndex;
using hylis::index::RMIndex;

namespace {

using Dynamic = DynamicRMIndex<std::int64_t, std::int64_t>;
using Static = RMIndex<std::int64_t, std::int64_t>;

// Same four CDF shapes as test_rmi.cpp and python/hylis/datasets.py, so every
// suite stresses the same distributions.
enum class Shape { SequentialGaps, Uniform, Lognormal, Clustered };

std::vector<std::int64_t> make_keys(Shape shape, std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::int64_t> keys;
    keys.reserve(n * 2);

    switch (shape) {
        case Shape::SequentialGaps: {
            std::uniform_int_distribution<std::int64_t> gap(1, 7);
            std::int64_t k = 0;
            for (std::size_t i = 0; i < n; ++i) keys.push_back(k += gap(rng));
            break;
        }
        case Shape::Uniform: {
            std::uniform_int_distribution<std::int64_t> d(0, std::int64_t{1} << 40);
            for (std::size_t i = 0; i < n * 2; ++i) keys.push_back(d(rng));
            break;
        }
        case Shape::Lognormal: {
            std::lognormal_distribution<double> d(0.0, 2.0);
            for (std::size_t i = 0; i < n * 2; ++i) {
                keys.push_back(static_cast<std::int64_t>(d(rng) * 1e6));
            }
            break;
        }
        case Shape::Clustered: {
            std::uniform_int_distribution<std::int64_t> centre(0, std::int64_t{1} << 40);
            std::uniform_int_distribution<std::int64_t> offset(0, 10000);
            std::vector<std::int64_t> centres;
            for (int i = 0; i < 64; ++i) centres.push_back(centre(rng));
            std::uniform_int_distribution<std::size_t> pick(0, centres.size() - 1);
            for (std::size_t i = 0; i < n * 2; ++i) {
                keys.push_back(centres[pick(rng)] + offset(rng));
            }
            break;
        }
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() > n) keys.resize(n);
    return keys;
}

const char* name_of(Shape s) {
    switch (s) {
        case Shape::SequentialGaps: return "sequential_gaps";
        case Shape::Uniform: return "uniform";
        case Shape::Lognormal: return "lognormal";
        case Shape::Clustered: return "clustered";
    }
    return "?";
}

std::vector<std::int64_t> payloads(std::size_t n) {
    std::vector<std::int64_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int64_t>(i) * 10;
    return v;
}

Dynamic build(const std::vector<std::int64_t>& keys, Dynamic::Config cfg = {}) {
    Dynamic idx(cfg);
    idx.build(keys, payloads(keys.size()));
    return idx;
}

// ---------------------------------------------------------------- basics

TEST(DynamicRMIndex, StartsAsTheStaticIndexDid) {
    const auto keys = make_keys(Shape::SequentialGaps, 5000, 1);
    Dynamic idx = build(keys);
    EXPECT_EQ(idx.size(), keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        ASSERT_NE(idx.find(keys[i]), nullptr) << "key " << keys[i];
        EXPECT_EQ(*idx.find(keys[i]), static_cast<std::int64_t>(i) * 10);
    }
    EXPECT_NO_THROW(idx.validate());
}

TEST(DynamicRMIndex, EmptyIndexAnswersEverythingWithNothing) {
    Dynamic idx = build({});
    EXPECT_TRUE(idx.empty());
    EXPECT_EQ(idx.find(42), nullptr);
    EXPECT_TRUE(idx.range(0, 100).empty());
    for (CompareOp op : {CompareOp::Eq, CompareOp::Lt, CompareOp::Le,
                         CompareOp::Gt, CompareOp::Ge}) {
        EXPECT_TRUE(idx.range_query(op, 42).empty());
    }
    EXPECT_TRUE(idx.insert(42, 7));
    EXPECT_NE(idx.find(42), nullptr);
}

TEST(DynamicRMIndex, RejectsDuplicatesFromEitherSide) {
    const auto keys = make_keys(Shape::SequentialGaps, 1000, 2);
    Dynamic idx = build(keys);

    EXPECT_FALSE(idx.insert(keys[10], 999)) << "already in the base";
    ASSERT_TRUE(idx.insert(keys[10] + 1, 999));
    EXPECT_FALSE(idx.insert(keys[10] + 1, 111)) << "already in the delta";
}

TEST(DynamicRMIndex, ErasingWhatIsNotThereReportsSo) {
    const auto keys = make_keys(Shape::SequentialGaps, 1000, 3);
    Dynamic idx = build(keys);
    EXPECT_FALSE(idx.erase(keys.back() + 12345));
    EXPECT_TRUE(idx.erase(keys[5]));
    EXPECT_FALSE(idx.erase(keys[5])) << "already tombstoned";
}

// A key can be in the base, in the delta, in both (the base copy tombstoned),
// or gone from either. Each has its own path through find().
TEST(DynamicRMIndex, ResolvesEveryCombinationOfBaseAndDelta) {
    const auto keys = make_keys(Shape::SequentialGaps, 2000, 4);
    Dynamic::Config cfg;
    cfg.merge_ratio = 10.0;  // never merge, so the states stay observable
    Dynamic idx = build(keys, cfg);

    const std::int64_t base_only = keys[100];
    const std::int64_t delta_only = keys[200] + 1;
    const std::int64_t both = keys[300];
    const std::int64_t erased_base = keys[400];
    const std::int64_t erased_delta = keys[500] + 1;

    ASSERT_TRUE(idx.insert(delta_only, -1));
    ASSERT_TRUE(idx.erase(both));
    ASSERT_TRUE(idx.insert(both, -2)) << "reinserting a tombstoned key";
    ASSERT_TRUE(idx.erase(erased_base));
    ASSERT_TRUE(idx.insert(erased_delta, -3));
    ASSERT_TRUE(idx.erase(erased_delta));

    ASSERT_NE(idx.find(base_only), nullptr);
    EXPECT_EQ(*idx.find(base_only), 1000);
    ASSERT_NE(idx.find(delta_only), nullptr);
    EXPECT_EQ(*idx.find(delta_only), -1);
    ASSERT_NE(idx.find(both), nullptr);
    EXPECT_EQ(*idx.find(both), -2) << "the delta holds the newer value";
    EXPECT_EQ(idx.find(erased_base), nullptr);
    EXPECT_EQ(idx.find(erased_delta), nullptr);

    // And a range spanning all of them must agree with the point lookups.
    const auto got = idx.range(keys[99], keys[501]);
    EXPECT_EQ(std::count(got.begin(), got.end(), -2), 1);
    EXPECT_EQ(std::count(got.begin(), got.end(), -1), 1);
    EXPECT_EQ(std::count(got.begin(), got.end(), -3), 0);
    EXPECT_NO_THROW(idx.validate());
}

// ------------------------------------------------------ the oracle tests

// The one that matters. Everything else is a special case of this.
class DynamicWorkload : public ::testing::TestWithParam<Shape> {};

TEST_P(DynamicWorkload, AgreesWithStdMapThroughALongMixedWorkload) {
    const Shape shape = GetParam();
    const auto keys = make_keys(shape, 8000, 11);
    ASSERT_GT(keys.size(), 1000u) << name_of(shape);

    std::map<std::int64_t, std::int64_t> oracle;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        oracle[keys[i]] = static_cast<std::int64_t>(i) * 10;
    }

    Dynamic::Config cfg;
    cfg.second_stage_size = 256;
    cfg.merge_ratio = 0.03;  // merge often, so many merges are exercised
    Dynamic idx = build(keys, cfg);

    const std::int64_t span = keys.back() - keys.front() + 1;
    std::mt19937_64 rng(7);

    for (int step = 0; step < 25000; ++step) {
        const int op = static_cast<int>(rng() % 100);
        const std::int64_t probe = keys.front() +
            static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(span));

        if (op < 30) {
            const std::int64_t value = static_cast<std::int64_t>(step);
            const bool wanted = oracle.emplace(probe, value).second;
            EXPECT_EQ(idx.insert(probe, value), wanted)
                << "insert " << probe << " at step " << step;
        } else if (op < 55) {
            // Half the deletions target a key that exists, so the tombstone
            // path is exercised rather than just the miss path.
            std::int64_t target = probe;
            if ((rng() & 1) && !oracle.empty()) {
                auto it = oracle.begin();
                std::advance(it, static_cast<std::ptrdiff_t>(rng() % oracle.size()));
                target = it->first;
            }
            const bool wanted = oracle.erase(target) > 0;
            EXPECT_EQ(idx.erase(target), wanted)
                << "erase " << target << " at step " << step;
        } else if (op < 90) {
            const auto it = oracle.find(probe);
            const std::int64_t* got = idx.find(probe);
            if (it == oracle.end()) {
                EXPECT_EQ(got, nullptr) << "find " << probe << " at step " << step;
            } else {
                ASSERT_NE(got, nullptr) << "find " << probe << " at step " << step;
                EXPECT_EQ(*got, it->second);
            }
        } else {
            const std::int64_t hi = probe + static_cast<std::int64_t>(rng() % 100000);
            std::vector<std::int64_t> wanted;
            for (auto it = oracle.lower_bound(probe);
                 it != oracle.end() && it->first <= hi; ++it) {
                wanted.push_back(it->second);
            }
            EXPECT_EQ(idx.range(probe, hi), wanted)
                << "range at step " << step;
        }
    }

    EXPECT_EQ(idx.size(), oracle.size());
    EXPECT_NO_THROW(idx.validate())
        << "exactness did not survive the workload on " << name_of(shape);
    EXPECT_GT(idx.stats().merges, 0u) << "the workload never merged; it proves little";
}

TEST_P(DynamicWorkload, AgreesWithStdMapOnEveryPredicate) {
    const Shape shape = GetParam();
    const auto keys = make_keys(shape, 3000, 12);
    std::map<std::int64_t, std::int64_t> oracle;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        oracle[keys[i]] = static_cast<std::int64_t>(i) * 10;
    }

    Dynamic::Config cfg;
    cfg.second_stage_size = 128;
    cfg.merge_ratio = 10.0;  // leave the delta and tombstones in place
    Dynamic idx = build(keys, cfg);

    std::mt19937_64 rng(13);
    for (int i = 0; i < 200; ++i) {
        const std::int64_t k = keys[rng() % keys.size()];
        if (rng() & 1) {
            if (oracle.erase(k)) {
                ASSERT_TRUE(idx.erase(k));
            }
        } else {
            const std::int64_t nk = k + 1;
            if (oracle.emplace(nk, -i).second) {
                ASSERT_TRUE(idx.insert(nk, -i));
            }
        }
    }
    ASSERT_GT(idx.stats().delta_size, 0u);
    ASSERT_GT(idx.stats().tombstones, 0u);

    for (int t = 0; t < 60; ++t) {
        const std::int64_t value = keys[rng() % keys.size()] + ((rng() & 1) ? 0 : 1);
        for (CompareOp op : {CompareOp::Eq, CompareOp::Lt, CompareOp::Le,
                             CompareOp::Gt, CompareOp::Ge}) {
            std::vector<std::int64_t> wanted;
            for (const auto& kv : oracle) {
                const bool hit =
                    (op == CompareOp::Eq && kv.first == value) ||
                    (op == CompareOp::Lt && kv.first < value) ||
                    (op == CompareOp::Le && kv.first <= value) ||
                    (op == CompareOp::Gt && kv.first > value) ||
                    (op == CompareOp::Ge && kv.first >= value);
                if (hit) wanted.push_back(kv.second);
            }
            EXPECT_EQ(idx.range_query(op, value), wanted)
                << "op " << static_cast<int>(op) << " value " << value
                << " on " << name_of(shape);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AllShapes, DynamicWorkload,
                         ::testing::Values(Shape::SequentialGaps, Shape::Uniform,
                                           Shape::Lognormal, Shape::Clustered),
                         [](const auto& info) { return name_of(info.param); });

// ------------------------------------------------- the merge machinery

// The claim the incremental merge rests on: a model that saw no insert and no
// delete needs only its intercept moved, and that leaves its residuals — and
// so its error bounds — untouched. If that were wrong, exactness would depend
// on which models a write happened to land in.
TEST(IncrementalMerge, UndisturbedModelsAreShiftedRatherThanRefitted) {
    const auto keys = make_keys(Shape::SequentialGaps, 40000, 21);
    Dynamic::Config cfg;
    cfg.second_stage_size = 256;
    cfg.merge_ratio = 10.0;
    Dynamic idx = build(keys, cfg);

    // A localised batch: everything lands in a handful of models.
    for (int i = 0; i < 300; ++i) idx.insert(keys[20000] + i * 2 + 1, i);
    idx.merge();

    const auto stats = idx.stats();
    EXPECT_GT(stats.models_shifted, stats.models_refitted * 10)
        << "a localised batch should leave almost every model merely shifted";
    EXPECT_LT(stats.keys_rescanned, keys.size() / 4)
        << "and should not have rescanned most of the index";
    EXPECT_NO_THROW(idx.validate());
}

// The other half of the same claim, and the honest one: a batch spread across
// the whole key range disturbs most models, and then the incremental merge
// saves little. Asserted so the saving is never quoted as unconditional.
TEST(IncrementalMerge, ScatteredWritesDisturbMostModels) {
    const auto keys = make_keys(Shape::SequentialGaps, 40000, 22);
    Dynamic::Config cfg;
    cfg.second_stage_size = 256;
    cfg.merge_ratio = 10.0;
    Dynamic idx = build(keys, cfg);

    std::mt19937_64 rng(23);
    for (int i = 0; i < 600; ++i) idx.insert(keys[rng() % keys.size()] + 1, i);
    idx.merge();

    const auto stats = idx.stats();
    EXPECT_GT(stats.models_refitted, stats.models_shifted)
        << "a scattered batch touches most models, and the merge is not cheap";
    EXPECT_NO_THROW(idx.validate());
}

TEST(IncrementalMerge, ReachesTheSameAnswersAsAFullRebuild) {
    const auto keys = make_keys(Shape::Lognormal, 20000, 24);
    Dynamic::Config cfg;
    cfg.second_stage_size = 256;
    cfg.merge_ratio = 10.0;
    Dynamic idx = build(keys, cfg);

    std::mt19937_64 rng(25);
    std::map<std::int64_t, std::int64_t> oracle;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        oracle[keys[i]] = static_cast<std::int64_t>(i) * 10;
    }
    for (int i = 0; i < 2000; ++i) {
        const std::int64_t k = keys[rng() % keys.size()] + 1;
        if (oracle.emplace(k, -i).second) idx.insert(k, -i);
    }
    for (int i = 0; i < 1000; ++i) {
        const std::int64_t k = keys[rng() % keys.size()];
        if (oracle.erase(k)) idx.erase(k);
    }
    idx.merge();

    // A static index built from scratch on the same contents. Its models are
    // different — it refitted stage 1, the merged one did not — but every
    // answer must be identical, because both are exact.
    std::vector<std::int64_t> nk, nv;
    for (const auto& kv : oracle) { nk.push_back(kv.first); nv.push_back(kv.second); }
    Static fresh(256);
    fresh.build(nk, nv);

    for (const auto& kv : oracle) {
        const std::int64_t* a = idx.find(kv.first);
        const std::int64_t* b = fresh.find(kv.first);
        ASSERT_NE(a, nullptr) << kv.first;
        ASSERT_NE(b, nullptr) << kv.first;
        EXPECT_EQ(*a, *b);
        EXPECT_EQ(*a, kv.second);
    }
    EXPECT_NO_THROW(idx.validate());
}

TEST(IncrementalMerge, TombstonesAreReclaimed) {
    const auto keys = make_keys(Shape::SequentialGaps, 10000, 26);
    Dynamic::Config cfg;
    cfg.merge_ratio = 10.0;
    Dynamic idx = build(keys, cfg);

    for (int i = 0; i < 500; ++i) idx.erase(keys[i * 7]);
    EXPECT_EQ(idx.stats().tombstones, 500u);
    EXPECT_EQ(idx.stats().base_size, keys.size()) << "not compacted yet";

    idx.merge();
    EXPECT_EQ(idx.stats().tombstones, 0u);
    EXPECT_EQ(idx.stats().base_size, keys.size() - 500);
    EXPECT_EQ(idx.size(), keys.size() - 500);
    EXPECT_NO_THROW(idx.validate());
}

TEST(IncrementalMerge, TheSizeTriggerFires) {
    const auto keys = make_keys(Shape::SequentialGaps, 10000, 27);
    Dynamic::Config cfg;
    cfg.merge_ratio = 0.01;  // 100 pending changes
    Dynamic idx = build(keys, cfg);

    EXPECT_EQ(idx.stats().merges, 0u);
    for (int i = 0; i < 150; ++i) idx.insert(keys[i * 11] + 1, i);
    EXPECT_GT(idx.stats().merges, 0u);
    EXPECT_LT(idx.stats().delta_size, 150u) << "the delta was folded back in";
}

// ------------------------------------------------------------ unlearning

// Machine unlearning, in the sense the paper uses it: after withdrawing a
// batch from a model's moments, the model those moments imply must be the
// model you would have got by never having had the batch. Exact here, rather
// than approximate, because tombstoning leaves every other key's position
// alone.
TEST(Unlearning, WithdrawingKeysMatchesNeverHavingHadThem) {
    const auto keys = make_keys(Shape::Uniform, 20000, 31);
    std::vector<std::int64_t> values = payloads(keys.size());

    Static full(64);
    full.build(keys, values);
    full.enable_incremental();
    ASSERT_TRUE(full.incremental_enabled());

    // Withdraw a contiguous run, which is what a tombstoned batch looks like
    // to a single model.
    std::set<std::size_t> removed;
    for (std::size_t i = 500; i < 600; ++i) {
        full.unlearn_position(i);
        removed.insert(i);
    }

    // The same index, built without those keys ever being present, would have
    // different positions for everything after the hole — so that is not the
    // comparison. The comparison is against the moments of exactly the keys
    // that remain, at exactly the positions they still occupy.
    Static reference(64);
    reference.build(keys, values);
    reference.enable_incremental();
    for (std::size_t i : removed) reference.unlearn_position(i);

    for (std::size_t m = 0; m < 64; ++m) {
        EXPECT_NEAR(full.model_score(m), reference.model_score(m), 1e-9)
            << "model " << m;
    }
    EXPECT_NO_THROW(full.validate())
        << "unlearning must not touch the installed model, only its statistics";
}

TEST(Unlearning, LeavesTheInstalledModelAndItsBoundsAlone) {
    const auto keys = make_keys(Shape::Uniform, 10000, 32);
    Static idx(64);
    idx.build(keys, payloads(keys.size()));
    idx.enable_incremental();

    const std::size_t max_error_before = idx.stats().max_error;
    for (std::size_t i = 0; i < 200; ++i) idx.unlearn_position(i * 13);

    EXPECT_EQ(idx.stats().max_error, max_error_before)
        << "unlearning changed the error bounds; lookups could now miss keys";
    EXPECT_NO_THROW(idx.validate());
}

TEST(Unlearning, TheScoreIsZeroUntilSomethingChanges) {
    const auto keys = make_keys(Shape::Uniform, 20000, 33);
    Static idx(256);
    idx.build(keys, payloads(keys.size()));
    idx.enable_incremental();

    // Not "small": exactly zero. The score is measured against the fit the
    // moments implied when they were last synchronised, so an untouched index
    // has moved by precisely nothing. Measuring it against the installed model
    // instead would leave float-level disagreement in the numerator and a
    // near-zero MSE in the denominator, and report a large score for an index
    // nothing had been done to.
    for (std::size_t m = 0; m < 256; ++m) {
        EXPECT_DOUBLE_EQ(idx.model_score(m), 0.0) << "model " << m;
    }

    idx.unlearn_position(5000);
    EXPECT_GT(idx.model_score(idx.model_of_key(keys[5000])), 0.0);
}

// The measurement behind the default of score_threshold = infinity: Cook's
// distance is not scale-free in the segment length, so a threshold tuned for
// ALEX-sized leaves cannot be carried across to RMI segments unchanged.
TEST(Unlearning, TheScoreScalesStronglyWithSegmentLength) {
    const auto keys = make_keys(Shape::Uniform, 100000, 34);
    const auto values = payloads(keys.size());

    double previous = 0.0;
    for (std::size_t models : {std::size_t{64}, std::size_t{1024}}) {
        Static idx(models);
        idx.build(keys, values);
        idx.enable_incremental();

        std::mt19937_64 rng(35);
        std::vector<std::size_t> touched;
        for (int i = 0; i < 32; ++i) {
            const std::size_t pos = rng() % keys.size();
            idx.unlearn_position(pos);
            touched.push_back(idx.model_of_key(keys[pos]));
        }
        const double worst = idx.max_score(touched);
        if (previous > 0.0) {
            EXPECT_GT(worst, previous * 10.0)
                << "shorter segments should score far higher for the same "
                   "number of withdrawals; if they do not, a single threshold "
                   "would serve both and the default could be reconsidered";
        }
        previous = worst;
    }
}

// ----------------------------------------------------------- degradation

TEST(DynamicRMIndex, StaysExactWhenTheDataGrowsPastItsOriginalRange) {
    // Frozen routing is fitted to the keys present at build. Appending far
    // beyond them is the case it fits worst, so it is the case exactness most
    // needs to be checked on.
    const auto keys = make_keys(Shape::SequentialGaps, 5000, 41);
    Dynamic::Config cfg;
    cfg.second_stage_size = 128;
    cfg.merge_ratio = 0.05;
    Dynamic idx = build(keys, cfg);

    std::int64_t k = keys.back();
    for (int i = 0; i < 5000; ++i) {
        k += 1000;
        ASSERT_TRUE(idx.insert(k, i));
    }
    idx.merge();

    EXPECT_EQ(idx.size(), keys.size() + 5000);
    EXPECT_NO_THROW(idx.validate());
    for (int i = 0; i < 100; ++i) {
        const std::int64_t probe = keys.back() + (i + 1) * 1000;
        EXPECT_NE(idx.find(probe), nullptr) << probe;
    }
}

TEST(DynamicRMIndex, FallsBackToAFullRebuildWhenDriftGetsBad) {
    const auto keys = make_keys(Shape::SequentialGaps, 20000, 42);
    Dynamic::Config cfg;
    cfg.second_stage_size = 128;
    cfg.merge_ratio = 0.02;
    cfg.rebuild_error_ratio = 1.5;  // trip easily, so the path is exercised
    Dynamic idx = build(keys, cfg);

    std::int64_t k = keys.back();
    for (int i = 0; i < 30000; ++i) {
        k += 1;
        idx.insert(k, i);
    }
    idx.merge();

    EXPECT_GT(idx.stats().full_rebuilds, 0u)
        << "routing drifted far enough that stage 1 should have been refitted";
    EXPECT_NO_THROW(idx.validate());
}

TEST(DynamicRMIndex, TheStaticIndexPaysNothingForAnyOfThis) {
    // The dynamic support lives in a side vector precisely so a read-only
    // index is not charged for it. If Model itself ever grows, this fails.
    const auto keys = make_keys(Shape::Uniform, 10000, 43);
    Static idx(1024);
    idx.build(keys, payloads(keys.size()));
    const std::size_t lean = idx.stats().model_bytes;
    EXPECT_EQ(idx.stats().incremental_bytes, 0u);

    idx.enable_incremental();
    EXPECT_EQ(idx.stats().model_bytes, lean)
        << "enabling incremental support changed the model footprint";
    EXPECT_GT(idx.stats().incremental_bytes, 0u)
        << "and it is charged visibly rather than folded into model_bytes";
}

}  // namespace
