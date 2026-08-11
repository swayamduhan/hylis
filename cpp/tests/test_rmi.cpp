// Tests for the Recursive Model Index.
//
// The load-bearing property is that the index is *exact* despite being built
// on an approximation. validate() replays every key through the real lookup
// path and asserts its true position lies inside the predicted window, so a
// bad model can only cost time. Most tests below end by calling it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <vector>

#include "index/btree.hpp"
#include "index/rmi.hpp"

using hylis::index::BPlusTree;
using hylis::index::CompareOp;
using hylis::index::LinearModel;
using hylis::index::RMIndex;

namespace {

using Tree = BPlusTree<std::int64_t, std::int64_t>;
using Index = RMIndex<std::int64_t, std::int64_t>;

// The same four CDF shapes python/hylis/datasets.py generates, so the C++ and
// Python suites stress the same cases.
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
            // Tight clumps separated by enormous gaps: the SOSD `fb` shape,
            // and the case piecewise-linear fitting cannot converge on.
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

std::vector<std::int64_t> positions(std::size_t n) {
    std::vector<std::int64_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int64_t>(i) * 10;
    return v;
}

Index build(const std::vector<std::int64_t>& keys, std::size_t models = 1024) {
    Index idx(models);
    idx.build(keys, positions(keys.size()));
    return idx;
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

const Shape kAllShapes[] = {Shape::SequentialGaps, Shape::Uniform,
                            Shape::Lognormal, Shape::Clustered};

}  // namespace

// --------------------------------------------------------------------------
// LinearModel
// --------------------------------------------------------------------------

TEST(LinearModel, FitsAnExactLineExactly) {
    std::vector<double> xs{0, 1, 2, 3, 4}, ys{1, 3, 5, 7, 9};  // y = 2x + 1
    const LinearModel m = LinearModel::fit(xs.data(), ys.data(), xs.size());
    EXPECT_NEAR(m.slope(), 2.0, 1e-12);
    EXPECT_NEAR(m.intercept(), 1.0, 1e-12);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(m.predict(xs[i]), ys[i], 1e-9);
    }
}

TEST(LinearModel, HandlesDegenerateInputs) {
    EXPECT_NEAR(LinearModel::fit(nullptr, nullptr, 0).predict(5.0), 0.0, 1e-12);

    std::vector<double> one_x{7}, one_y{42};
    EXPECT_NEAR(LinearModel::fit(one_x.data(), one_y.data(), 1).predict(99.0), 42.0, 1e-12);

    // Every x identical: no slope is determined, so it must predict the mean
    // rather than divide by zero.
    std::vector<double> flat_x{3, 3, 3}, flat_y{1, 2, 3};
    const LinearModel m = LinearModel::fit(flat_x.data(), flat_y.data(), 3);
    EXPECT_NEAR(m.slope(), 0.0, 1e-12);
    EXPECT_NEAR(m.predict(3.0), 2.0, 1e-12);
}

TEST(LinearModel, StaysAccurateOnLargeMagnitudeInputs) {
    // The case that motivates mean-centring: raw sums of squares here would be
    // ~1e24 and lose the precision the fit depends on.
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < 1000; ++i) {
        xs.push_back(1e12 + static_cast<double>(i) * 1e6);
        ys.push_back(static_cast<double>(i));
    }
    const LinearModel m = LinearModel::fit(xs.data(), ys.data(), xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(m.predict(xs[i]), ys[i], 1e-3) << "i=" << i;
    }
}

// --------------------------------------------------------------------------
// build() input validation and edge cases
// --------------------------------------------------------------------------

TEST(RMIndex, ZeroModelsRejected) {
    EXPECT_THROW(Index(0), std::invalid_argument);
}

TEST(RMIndex, MismatchedKeyValueLengthsRejected) {
    Index idx;
    EXPECT_THROW(idx.build({1, 2, 3}, {10, 20}), std::invalid_argument);
}

TEST(RMIndex, UnsortedKeysRejected) {
    Index idx;
    EXPECT_THROW(idx.build({3, 1, 2}, {1, 2, 3}), std::invalid_argument);
}

TEST(RMIndex, DuplicateKeysRejected) {
    Index idx;
    EXPECT_THROW(idx.build({1, 2, 2, 3}, {1, 2, 3, 4}), std::invalid_argument);
}

TEST(RMIndex, EmptyIndex) {
    Index idx;
    idx.build({}, {});
    EXPECT_EQ(idx.size(), 0u);
    EXPECT_TRUE(idx.empty());
    EXPECT_EQ(idx.find(1), nullptr);
    EXPECT_TRUE(idx.range(0, 100).empty());
    EXPECT_TRUE(idx.range_query(CompareOp::Ge, 0).empty());
    EXPECT_NO_THROW(idx.validate());
}

TEST(RMIndex, SingleKey) {
    Index idx = build({42});
    EXPECT_EQ(idx.size(), 1u);
    ASSERT_NE(idx.find(42), nullptr);
    EXPECT_EQ(*idx.find(42), 0);
    EXPECT_EQ(idx.find(41), nullptr);
    EXPECT_EQ(idx.find(43), nullptr);
    EXPECT_NO_THROW(idx.validate());
}

TEST(RMIndex, TwoKeys) {
    Index idx = build({10, 20});
    EXPECT_EQ(*idx.find(10), 0);
    EXPECT_EQ(*idx.find(20), 10);
    EXPECT_EQ(idx.find(15), nullptr);
    EXPECT_NO_THROW(idx.validate());
}

TEST(RMIndex, ClearEmptiesTheIndex) {
    Index idx = build(make_keys(Shape::Uniform, 500, 1));
    idx.clear();
    EXPECT_EQ(idx.size(), 0u);
    EXPECT_EQ(idx.find(0), nullptr);
    EXPECT_NO_THROW(idx.validate());
}

// --------------------------------------------------------------------------
// Exactness across distributions -- the central claim
// --------------------------------------------------------------------------

TEST(RMIndex, FindsEveryKeyOnEveryDistribution) {
    for (Shape shape : kAllShapes) {
        const auto keys = make_keys(shape, 20000, 7);
        Index idx = build(keys);
        ASSERT_NO_THROW(idx.validate()) << name_of(shape);

        for (std::size_t i = 0; i < keys.size(); ++i) {
            const std::int64_t* v = idx.find(keys[i]);
            ASSERT_NE(v, nullptr) << name_of(shape) << " key index " << i;
            EXPECT_EQ(*v, static_cast<std::int64_t>(i) * 10);
        }
    }
}

TEST(RMIndex, AbsentKeysReturnNull) {
    for (Shape shape : kAllShapes) {
        const auto keys = make_keys(shape, 5000, 3);
        Index idx = build(keys);

        std::mt19937_64 rng(99);
        std::uniform_int_distribution<std::int64_t> d(-1000, std::int64_t{1} << 41);
        int checked = 0;
        for (int t = 0; t < 2000 && checked < 500; ++t) {
            const std::int64_t probe = d(rng);
            if (std::binary_search(keys.begin(), keys.end(), probe)) continue;
            ++checked;
            EXPECT_EQ(idx.find(probe), nullptr) << name_of(shape) << " probe " << probe;
        }
        EXPECT_GT(checked, 0) << "no absent keys were actually exercised";
    }
}

TEST(RMIndex, ExtremeKeyMagnitudesStillExact) {
    // Keys spanning nearly the whole int64 range. Beyond 2^53 a double cannot
    // represent every integer, so the models lose resolution -- correctness
    // must not depend on that.
    std::vector<std::int64_t> keys{
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min() / 2,
        -1, 0, 1,
        std::numeric_limits<std::int64_t>::max() / 2,
        std::numeric_limits<std::int64_t>::max(),
    };
    Index idx = build(keys, 16);
    ASSERT_NO_THROW(idx.validate());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        ASSERT_NE(idx.find(keys[i]), nullptr) << "i=" << i;
    }
}

TEST(RMIndex, AllKeysInOneTightClusterStillExact) {
    std::vector<std::int64_t> keys;
    for (std::int64_t i = 0; i < 5000; ++i) keys.push_back(1'000'000'000'000LL + i);
    Index idx = build(keys, 1024);
    ASSERT_NO_THROW(idx.validate());
    EXPECT_NE(idx.find(1'000'000'002'500LL), nullptr);
}

// --------------------------------------------------------------------------
// Range queries
// --------------------------------------------------------------------------

TEST(RMIndex, RangeMatchesAMapReference) {
    const auto keys = make_keys(Shape::Lognormal, 4000, 11);
    Index idx = build(keys);

    std::map<std::int64_t, std::int64_t> ref;
    for (std::size_t i = 0; i < keys.size(); ++i) ref[keys[i]] = static_cast<std::int64_t>(i) * 10;

    std::mt19937_64 rng(5);
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
    for (int t = 0; t < 200; ++t) {
        std::int64_t lo = keys[pick(rng)], hi = keys[pick(rng)];
        if (hi < lo) std::swap(lo, hi);

        std::vector<std::int64_t> expected;
        for (auto it = ref.lower_bound(lo); it != ref.end() && it->first <= hi; ++it) {
            expected.push_back(it->second);
        }
        EXPECT_EQ(idx.range(lo, hi), expected) << "trial " << t;
    }
}

TEST(RMIndex, InvertedRangeIsEmpty) {
    Index idx = build(make_keys(Shape::Uniform, 500, 2));
    EXPECT_TRUE(idx.range(100, 10).empty());
}

TEST(RMIndex, RangeQueryMatchesAMapReferenceForEveryOperator) {
    const auto keys = make_keys(Shape::Clustered, 3000, 13);
    Index idx = build(keys);

    std::map<std::int64_t, std::int64_t> ref;
    for (std::size_t i = 0; i < keys.size(); ++i) ref[keys[i]] = static_cast<std::int64_t>(i) * 10;

    std::mt19937_64 rng(17);
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
    for (int t = 0; t < 150; ++t) {
        // Alternate between probing a key that exists and one that does not,
        // since the boundary handling differs.
        const std::int64_t v = (t % 2) ? keys[pick(rng)] : keys[pick(rng)] + 1;

        std::vector<std::int64_t> lt, le, gt, ge, eq;
        for (const auto& [k, val] : ref) {
            if (k < v) lt.push_back(val);
            if (k <= v) le.push_back(val);
            if (k > v) gt.push_back(val);
            if (k >= v) ge.push_back(val);
            if (k == v) eq.push_back(val);
        }
        EXPECT_EQ(idx.range_query(CompareOp::Lt, v), lt) << "t=" << t;
        EXPECT_EQ(idx.range_query(CompareOp::Le, v), le) << "t=" << t;
        EXPECT_EQ(idx.range_query(CompareOp::Gt, v), gt) << "t=" << t;
        EXPECT_EQ(idx.range_query(CompareOp::Ge, v), ge) << "t=" << t;
        EXPECT_EQ(idx.range_query(CompareOp::Eq, v), eq) << "t=" << t;
    }
}

TEST(RMIndex, RangeQueryHandlesValuesOutsideTheKeyRange) {
    const auto keys = make_keys(Shape::Uniform, 1000, 21);
    Index idx = build(keys);
    const std::int64_t below = keys.front() - 1000;
    const std::int64_t above = keys.back() + 1000;

    EXPECT_TRUE(idx.range_query(CompareOp::Lt, below).empty());
    EXPECT_EQ(idx.range_query(CompareOp::Ge, below).size(), keys.size());
    EXPECT_EQ(idx.range_query(CompareOp::Le, above).size(), keys.size());
    EXPECT_TRUE(idx.range_query(CompareOp::Gt, above).empty());
}

TEST(RMIndex, ItemsAndKeysAreAscending) {
    const auto keys = make_keys(Shape::Lognormal, 1000, 23);
    Index idx = build(keys);
    EXPECT_EQ(idx.keys(), keys);
    const auto items = idx.items();
    ASSERT_EQ(items.size(), keys.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        EXPECT_EQ(items[i].first, keys[i]);
        EXPECT_EQ(items[i].second, static_cast<std::int64_t>(i) * 10);
    }
}

// --------------------------------------------------------------------------
// Differential against std::map and against the B+ tree
// --------------------------------------------------------------------------

TEST(RMIndex, DifferentialFuzzAgainstStdMap) {
    for (Shape shape : kAllShapes) {
        for (std::size_t models : {std::size_t{1}, std::size_t{16}, std::size_t{1024}}) {
            const auto keys = make_keys(shape, 3000, 31);
            Index idx = build(keys, models);
            ASSERT_NO_THROW(idx.validate()) << name_of(shape) << " M=" << models;

            std::map<std::int64_t, std::int64_t> ref;
            for (std::size_t i = 0; i < keys.size(); ++i) {
                ref[keys[i]] = static_cast<std::int64_t>(i) * 10;
            }

            std::mt19937_64 rng(41);
            std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
            for (int t = 0; t < 500; ++t) {
                const std::int64_t probe = (t % 3) ? keys[pick(rng)] : keys[pick(rng)] + 1;
                const auto it = ref.find(probe);
                const std::int64_t* got = idx.find(probe);
                if (it == ref.end()) {
                    EXPECT_EQ(got, nullptr)
                        << name_of(shape) << " M=" << models << " probe=" << probe;
                } else {
                    ASSERT_NE(got, nullptr)
                        << name_of(shape) << " M=" << models << " probe=" << probe;
                    EXPECT_EQ(*got, it->second);
                }
            }
        }
    }
}

TEST(RMIndex, AgreesWithTheBPlusTreeOnEveryDistribution) {
    // The two structures share the CompareOp interface precisely so the query
    // planner can swap one for the other. That is only safe if they are
    // indistinguishable through it.
    for (Shape shape : kAllShapes) {
        const auto keys = make_keys(shape, 5000, 53);
        const auto vals = positions(keys.size());

        Index rmi = build(keys);
        Tree tree(32);
        for (std::size_t i = 0; i < keys.size(); ++i) tree.insert(keys[i], vals[i]);

        ASSERT_EQ(rmi.keys(), tree.keys()) << name_of(shape);
        ASSERT_EQ(rmi.items(), tree.items()) << name_of(shape);

        std::mt19937_64 rng(59);
        std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
        for (int t = 0; t < 300; ++t) {
            const std::int64_t v = (t % 2) ? keys[pick(rng)] : keys[pick(rng)] + 7;

            const std::int64_t* a = rmi.find(v);
            const std::int64_t* b = tree.find(v);
            ASSERT_EQ(a == nullptr, b == nullptr) << name_of(shape) << " v=" << v;
            if (a != nullptr) { EXPECT_EQ(*a, *b); }

            for (CompareOp op : {CompareOp::Lt, CompareOp::Le, CompareOp::Gt,
                                 CompareOp::Ge, CompareOp::Eq}) {
                EXPECT_EQ(rmi.range_query(op, v), tree.range_query(op, v))
                    << name_of(shape) << " op=" << static_cast<int>(op) << " v=" << v;
            }
        }

        std::int64_t lo = keys[pick(rng)], hi = keys[pick(rng)];
        if (hi < lo) std::swap(lo, hi);
        EXPECT_EQ(rmi.range(lo, hi), tree.range(lo, hi)) << name_of(shape);
    }
}

// --------------------------------------------------------------------------
// Curvature: what more models actually buy
// --------------------------------------------------------------------------

TEST(RMIndex, MoreModelsReduceErrorOnASmoothlyCurvedDistribution) {
    // lognormal is curved but continuous, so piecewise-linear fitting should
    // converge on it: this is the claim that the second stage handles
    // curvature at all.
    //
    // Asserted on *mean* error, which is what expected lookup cost is
    // proportional to. Max error is a tail statistic and behaves differently
    // -- see the test below, which pins that separately.
    const auto keys = make_keys(Shape::Lognormal, 50000, 61);

    const double mean_1 = build(keys, 1).stats().mean_error;
    const double mean_64 = build(keys, 64).stats().mean_error;
    const double mean_4096 = build(keys, 4096).stats().mean_error;

    EXPECT_LT(mean_64, mean_1 / 100.0) << "64 models should be orders better than 1";
    EXPECT_LT(mean_4096, mean_64) << "convergence should not stall at 64";
}

TEST(RMIndex, MaxErrorPlateausOnASkewedDistribution) {
    // A real limitation, pinned rather than hidden.
    //
    // Stage 1 routes by *predicted position*. On a heavily skewed CDF the
    // dense region all predicts a position near zero and piles into the first
    // few second-stage models however large M is, so the worst-fitted spot
    // stops improving even while the average keeps getting better. This is
    // the weakness that motivates asking whether a non-linear stage 1 would
    // route better -- see scripts/experiment_stage1.py.
    const auto keys = make_keys(Shape::Lognormal, 50000, 61);

    const auto s64 = build(keys, 64).stats();
    const auto s16384 = build(keys, 16384).stats();

    EXPECT_LT(s16384.mean_error, s64.mean_error) << "the average must still improve";
    EXPECT_GT(s16384.max_error * 2, s64.max_error)
        << "max error is expected to plateau; if it now converges too, the "
           "stage-1 routing has been improved and this test should be revisited";
}

TEST(RMIndex, MoreModelsDoNotRescueASteppedDistribution) {
    // clustered has cliffs, not curves. There are only 64 clusters, so once
    // each has its own model every further model is empty and *nothing*
    // changes -- not approximately, exactly. That hard floor is precisely why
    // a distribution-free B+ tree still has a job.
    const auto keys = make_keys(Shape::Clustered, 50000, 67);

    const auto s1024 = build(keys, 1024).stats();
    const auto s16384 = build(keys, 16384).stats();
    const auto s65536 = build(keys, 65536).stats();

    EXPECT_EQ(s16384.max_error, s1024.max_error);
    EXPECT_EQ(s65536.max_error, s1024.max_error);
    EXPECT_DOUBLE_EQ(s65536.mean_error, s1024.mean_error)
        << "16x more models changed nothing at all: the error floor is set by "
           "the gaps in the data, not by the model budget";

    // And the reason: almost every extra model is routed nothing.
    EXPECT_GT(s65536.empty_models * 100, s65536.models * 99);
}

TEST(RMIndex, StatsAreSelfConsistent) {
    const auto keys = make_keys(Shape::Uniform, 10000, 71);
    Index idx = build(keys, 256);
    const auto s = idx.stats();

    EXPECT_EQ(s.size, keys.size());
    EXPECT_EQ(s.models, 256u);
    EXPECT_LT(s.max_error, keys.size());
    EXPECT_LE(s.mean_error, static_cast<double>(s.max_error));
    EXPECT_GE(s.max_window, 1u);
    EXPECT_LE(s.empty_models, s.models);
    EXPECT_GT(s.total_bytes, s.model_bytes);
}

TEST(RMIndex, ModelOverheadIsIndependentOfKeyCount) {
    // The headline memory argument: a B+ tree's internal nodes grow with n,
    // while the RMI's models do not.
    const std::size_t small = build(make_keys(Shape::Uniform, 1000, 73), 1024)
                                  .stats().model_bytes;
    const std::size_t large = build(make_keys(Shape::Uniform, 100000, 73), 1024)
                                  .stats().model_bytes;
    EXPECT_EQ(small, large);
}

// --------------------------------------------------------------------------
// Graceful degradation
// --------------------------------------------------------------------------

TEST(RMIndex, ASingleUselessModelIsStillCorrectAndStillLogarithmic) {
    // M=1 on clustered keys is close to the worst model obtainable: one line
    // through a step function. Correctness must survive it, and the probe
    // count must stay logarithmic rather than collapsing to a linear scan.
    const auto keys = make_keys(Shape::Clustered, 20000, 79);
    Index idx = build(keys, 1);
    ASSERT_NO_THROW(idx.validate());

    const double log2n = std::log2(static_cast<double>(keys.size()));
    for (std::size_t i = 0; i < keys.size(); i += 97) {
        ASSERT_NE(idx.find(keys[i]), nullptr) << "i=" << i;
        EXPECT_LE(static_cast<double>(idx.probes(keys[i])), log2n + 8.0)
            << "probe count at i=" << i << " is not logarithmic";
    }
}

TEST(RMIndex, SearchThresholdDoesNotChangeResults) {
    const auto keys = make_keys(Shape::Lognormal, 5000, 83);
    const auto vals = positions(keys.size());

    for (std::size_t threshold : {std::size_t{0}, std::size_t{1}, std::size_t{64},
                                  std::size_t{100000}}) {
        Index idx(256, threshold);
        idx.build(keys, vals);
        ASSERT_NO_THROW(idx.validate()) << "threshold=" << threshold;
        for (std::size_t i = 0; i < keys.size(); i += 37) {
            const std::int64_t* v = idx.find(keys[i]);
            ASSERT_NE(v, nullptr) << "threshold=" << threshold << " i=" << i;
            EXPECT_EQ(*v, vals[i]);
        }
        EXPECT_EQ(idx.find(keys.back() + 1), nullptr);
    }
}
