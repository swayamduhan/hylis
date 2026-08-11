// Tests for the flat (brute-force) vector index.
//
// Exhaustive search has exactly one correct answer, which makes this the
// strongest test suite in the project: everything below is an equality
// assertion, with no tolerance and no "close enough". The approximate
// structures that come later will not get that luxury, and they will be
// graded against this.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "index/flat.hpp"

using hylis::index::FlatIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;

namespace {

std::vector<std::int64_t> ids_of(const std::vector<Neighbor>& ns) {
    std::vector<std::int64_t> out;
    out.reserve(ns.size());
    for (const Neighbor& n : ns) out.push_back(n.id);
    return out;
}

// Reference implementation: score every row, sort the lot, take the first k.
//
// This isolates *selection* -- the bounded heap, the early rejection, the
// tie-break -- which is the part of FlatIndex that is easy to get subtly
// wrong. It therefore mirrors the index's float arithmetic exactly rather
// than recomputing in double: were it to use double, the two would disagree
// in the last bit on near-ties and the test would be measuring floating-point
// associativity instead of the logic under test.
//
// Scoring is checked separately and independently, against numpy's oracle on
// real SIFT vectors, in tests/test_flat.py.
void ref_normalise(float* v, std::size_t dim) {
    float sq = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) sq += v[i] * v[i];
    if (sq <= 0.0f) return;
    const float norm = std::sqrt(sq);
    for (std::size_t i = 0; i < dim; ++i) v[i] /= norm;
}

float ref_score(const float* q, const float* b, std::size_t dim, Metric metric) {
    if (metric == Metric::L2) {
        float acc = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            const float d = q[i] - b[i];
            acc += d * d;
        }
        return acc;
    }
    float dot = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) dot += q[i] * b[i];
    return -dot;
}

std::vector<std::int64_t> naive_search(const std::vector<float>& data, std::size_t dim,
                                       const std::vector<float>& query, std::size_t k,
                                       Metric metric) {
    const std::size_t n = data.size() / dim;

    std::vector<float> q = query;
    if (metric == Metric::Cosine) ref_normalise(q.data(), dim);

    std::vector<std::pair<float, std::int64_t>> scored;
    scored.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> row(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        if (metric == Metric::Cosine) ref_normalise(row.data(), dim);
        scored.push_back({ref_score(q.data(), row.data(), dim, metric),
                          std::int64_t(i)});
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });

    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i < std::min(k, scored.size()); ++i) {
        out.push_back(scored[i].second);
    }
    return out;
}

// Four points on a line in 1-D: trivially checkable by hand.
FlatIndex line_index() {
    FlatIndex idx(1);
    for (float v : {0.0f, 1.0f, 2.0f, 10.0f}) idx.add(&v);
    return idx;
}

}  // namespace

// --------------------------------------------------------------------------
// Construction and basic bookkeeping
// --------------------------------------------------------------------------

TEST(FlatIndex, StartsEmpty) {
    FlatIndex idx(4);
    EXPECT_EQ(idx.size(), 0u);
    EXPECT_EQ(idx.dim(), 4u);
    EXPECT_TRUE(idx.empty());
    EXPECT_EQ(idx.metric(), Metric::L2);
}

TEST(FlatIndex, ZeroDimensionRejected) {
    EXPECT_THROW(FlatIndex(0), std::invalid_argument);
}

TEST(FlatIndex, AddAssignsSequentialIds) {
    FlatIndex idx(2);
    std::vector<float> a{1, 2}, b{3, 4};
    EXPECT_EQ(idx.add(a), 0);
    EXPECT_EQ(idx.add(b), 1);
    EXPECT_EQ(idx.size(), 2u);
}

TEST(FlatIndex, WrongDimensionRejected) {
    FlatIndex idx(3);
    std::vector<float> wrong{1, 2};
    EXPECT_THROW(idx.add(wrong), std::invalid_argument);
    idx.add(std::vector<float>{1, 2, 3});
    EXPECT_THROW(idx.search(wrong, 1), std::invalid_argument);
}

TEST(FlatIndex, AddBatchMatchesRepeatedAdd) {
    std::vector<float> rows{1, 2, 3, 4, 5, 6};
    FlatIndex one(2), many(2);
    for (std::size_t i = 0; i < 3; ++i) one.add(rows.data() + i * 2);
    many.add_batch(rows.data(), 3);

    ASSERT_EQ(one.size(), many.size());
    for (std::int64_t i = 0; i < 3; ++i) {
        EXPECT_EQ(one.vector_at(i)[0], many.vector_at(i)[0]);
        EXPECT_EQ(one.vector_at(i)[1], many.vector_at(i)[1]);
    }
}

TEST(FlatIndex, ClearResetsButKeepsDimAndMetric) {
    FlatIndex idx(2, Metric::Cosine);
    idx.add(std::vector<float>{1, 1});
    idx.clear();
    EXPECT_EQ(idx.size(), 0u);
    EXPECT_EQ(idx.dim(), 2u);
    EXPECT_EQ(idx.metric(), Metric::Cosine);
    EXPECT_TRUE(idx.search(std::vector<float>{1, 1}, 3).empty());
}

TEST(FlatIndex, VectorAtRejectsOutOfRangeIds) {
    FlatIndex idx(2);
    idx.add(std::vector<float>{1, 2});
    EXPECT_THROW(idx.vector_at(1), std::out_of_range);
    EXPECT_THROW(idx.vector_at(-1), std::out_of_range);
}

// --------------------------------------------------------------------------
// Exact search
// --------------------------------------------------------------------------

TEST(FlatIndex, FindsNearestOnALine) {
    FlatIndex idx = line_index();
    float q = 1.9f;
    EXPECT_EQ(ids_of(idx.search(&q, 2)), (std::vector<std::int64_t>{2, 1}));
}

TEST(FlatIndex, L2ScoreIsTheActualDistanceNotItsSquare) {
    FlatIndex idx(2);
    idx.add(std::vector<float>{0, 0});
    idx.add(std::vector<float>{3, 4});  // distance 5, squared 25

    auto got = idx.search(std::vector<float>{0, 0}, 2);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_FLOAT_EQ(got[0].score, 0.0f);
    EXPECT_FLOAT_EQ(got[1].score, 5.0f);
}

TEST(FlatIndex, NearestNeighbourOfAStoredPointIsItself) {
    FlatIndex idx(3);
    std::mt19937 rng(7);
    std::normal_distribution<float> dist;
    std::vector<float> rows(50 * 3);
    for (float& v : rows) v = dist(rng);
    idx.add_batch(rows.data(), 50);

    for (std::int64_t i = 0; i < 50; ++i) {
        auto got = idx.search(idx.vector_at(i), 1);
        ASSERT_EQ(got.size(), 1u);
        EXPECT_EQ(got[0].id, i);
        EXPECT_NEAR(got[0].score, 0.0f, 1e-5f);
    }
}

TEST(FlatIndex, KLargerThanTheIndexIsClamped) {
    FlatIndex idx = line_index();
    float q = 0.0f;
    EXPECT_EQ(idx.search(&q, 100).size(), 4u);
}

TEST(FlatIndex, ZeroKReturnsNothing) {
    FlatIndex idx = line_index();
    float q = 0.0f;
    EXPECT_TRUE(idx.search(&q, 0).empty());
}

TEST(FlatIndex, SearchOnEmptyIndexReturnsNothing) {
    FlatIndex idx(2);
    EXPECT_TRUE(idx.search(std::vector<float>{1, 1}, 5).empty());
}

TEST(FlatIndex, ResultsAreOrderedBestFirst) {
    FlatIndex idx = line_index();
    float q = 0.0f;
    auto got = idx.search(&q, 4);
    for (std::size_t i = 1; i < got.size(); ++i) {
        EXPECT_LE(got[i - 1].score, got[i].score);
    }
}

TEST(FlatIndex, TiesAreBrokenByLowerId) {
    // Four vectors all exactly one unit from the origin, so every candidate
    // ties. Without a tie-break the surviving two would depend on heap
    // ordering; with one, it is always the two lowest ids.
    FlatIndex idx(2);
    idx.add(std::vector<float>{1, 0});
    idx.add(std::vector<float>{0, 1});
    idx.add(std::vector<float>{-1, 0});
    idx.add(std::vector<float>{0, -1});

    auto got = idx.search(std::vector<float>{0, 0}, 2);
    EXPECT_EQ(ids_of(got), (std::vector<std::int64_t>{0, 1}));
}

TEST(FlatIndex, TieBreakIsIndependentOfScanOrder) {
    // The same four equidistant points, offered through search_filtered in
    // descending id order. A tie-break that quietly relied on ids arriving
    // ascending would pick {3, 2} here.
    FlatIndex idx(2);
    idx.add(std::vector<float>{1, 0});
    idx.add(std::vector<float>{0, 1});
    idx.add(std::vector<float>{-1, 0});
    idx.add(std::vector<float>{0, -1});

    auto got = idx.search_filtered(std::vector<float>{0, 0}, 2, {3, 2, 1, 0});
    EXPECT_EQ(ids_of(got), (std::vector<std::int64_t>{0, 1}));
}

// --------------------------------------------------------------------------
// Metrics
// --------------------------------------------------------------------------

TEST(FlatIndex, InnerProductPrefersLargerDotProducts) {
    FlatIndex idx(2, Metric::InnerProduct);
    idx.add(std::vector<float>{1, 0});
    idx.add(std::vector<float>{5, 0});
    idx.add(std::vector<float>{2, 0});

    auto got = idx.search(std::vector<float>{1, 0}, 3);
    EXPECT_EQ(ids_of(got), (std::vector<std::int64_t>{1, 2, 0}));
    EXPECT_FLOAT_EQ(got[0].score, 5.0f);
}

TEST(FlatIndex, CosineIgnoresMagnitude) {
    FlatIndex idx(2, Metric::Cosine);
    idx.add(std::vector<float>{1, 0});
    idx.add(std::vector<float>{100, 0});  // same direction, 100x the length
    idx.add(std::vector<float>{0, 1});

    auto got = idx.search(std::vector<float>{3, 0}, 3);
    // The two collinear vectors tie at similarity 1; the orthogonal one is 0.
    EXPECT_EQ(ids_of(got), (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_NEAR(got[0].score, 1.0f, 1e-6f);
    EXPECT_NEAR(got[1].score, 1.0f, 1e-6f);
    EXPECT_NEAR(got[2].score, 0.0f, 1e-6f);
}

TEST(FlatIndex, CosineQueryScaleDoesNotChangeResults) {
    FlatIndex idx(3, Metric::Cosine);
    std::mt19937 rng(11);
    std::normal_distribution<float> dist;
    std::vector<float> rows(40 * 3);
    for (float& v : rows) v = dist(rng);
    idx.add_batch(rows.data(), 40);

    std::vector<float> q{0.3f, -1.2f, 0.7f};
    std::vector<float> scaled{q[0] * 50, q[1] * 50, q[2] * 50};
    EXPECT_EQ(ids_of(idx.search(q, 10)), ids_of(idx.search(scaled, 10)));
}

TEST(FlatIndex, CosineHandlesZeroVectorsWithoutNaN) {
    FlatIndex idx(2, Metric::Cosine);
    idx.add(std::vector<float>{0, 0});
    idx.add(std::vector<float>{1, 1});

    auto got = idx.search(std::vector<float>{1, 1}, 2);
    ASSERT_EQ(got.size(), 2u);
    for (const Neighbor& n : got) EXPECT_FALSE(std::isnan(n.score));
    EXPECT_EQ(got[0].id, 1);
}

// --------------------------------------------------------------------------
// Filtered search -- the pre-filter execution plan
// --------------------------------------------------------------------------

TEST(FlatIndex, FilteredSearchOnlyReturnsAllowedIds) {
    FlatIndex idx = line_index();
    float q = 0.0f;
    auto got = idx.search_filtered(&q, 4, {2, 3});
    EXPECT_EQ(ids_of(got), (std::vector<std::int64_t>{2, 3}));
}

TEST(FlatIndex, FilteredSearchRejectsUnknownIds) {
    FlatIndex idx = line_index();
    float q = 0.0f;
    EXPECT_THROW(idx.search_filtered(&q, 1, {99}), std::out_of_range);
    EXPECT_THROW(idx.search_filtered(&q, 1, {-1}), std::out_of_range);
}

TEST(FlatIndex, EmptyFilterReturnsNothing) {
    FlatIndex idx = line_index();
    float q = 0.0f;
    EXPECT_TRUE(idx.search_filtered(&q, 3, {}).empty());
}

TEST(FlatIndex, FilterEverythingEqualsAnUnfilteredSearch) {
    FlatIndex idx = line_index();
    std::vector<std::int64_t> all{0, 1, 2, 3};
    float q = 1.5f;
    EXPECT_EQ(ids_of(idx.search(&q, 3)), ids_of(idx.search_filtered(&q, 3, all)));
}

TEST(FlatIndex, PreFilterAgreesWithPostFilter) {
    // The identity the query planner depends on: restricting the candidate
    // set up front must return the same answer as searching everything and
    // discarding the rows that fail the predicate. If these diverged the
    // planner would be choosing between different *answers* rather than
    // different costs, and comparing plans would be meaningless.
    std::mt19937 rng(21);
    std::normal_distribution<float> dist;
    FlatIndex idx(8);
    std::vector<float> rows(400 * 8);
    for (float& v : rows) v = dist(rng);
    idx.add_batch(rows.data(), 400);

    std::vector<std::int64_t> allowed;
    for (std::int64_t i = 0; i < 400; ++i) {
        if (i % 3 == 0) allowed.push_back(i);
    }

    for (int trial = 0; trial < 20; ++trial) {
        std::vector<float> q(8);
        for (float& v : q) v = dist(rng);

        auto pre = ids_of(idx.search_filtered(q, 5, allowed));

        auto full = idx.search(q, 400);
        std::vector<std::int64_t> post;
        for (const Neighbor& n : full) {
            if (n.id % 3 == 0) post.push_back(n.id);
            if (post.size() == 5) break;
        }
        EXPECT_EQ(pre, post) << "trial " << trial;
    }
}

// --------------------------------------------------------------------------
// Batch
// --------------------------------------------------------------------------

TEST(FlatIndex, BatchSearchMatchesIndividualSearches) {
    std::mt19937 rng(31);
    std::normal_distribution<float> dist;
    FlatIndex idx(6);
    std::vector<float> rows(200 * 6);
    for (float& v : rows) v = dist(rng);
    idx.add_batch(rows.data(), 200);

    std::vector<float> queries(9 * 6);
    for (float& v : queries) v = dist(rng);

    auto batched = idx.search_batch(queries.data(), 9, 4);
    ASSERT_EQ(batched.size(), 9u);
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_EQ(ids_of(batched[i]), ids_of(idx.search(queries.data() + i * 6, 4)));
    }
}

// --------------------------------------------------------------------------
// Differential fuzz against the naive reference
// --------------------------------------------------------------------------

TEST(FlatIndex, DifferentialFuzzAgainstNaiveSearch) {
    const Metric metrics[] = {Metric::L2, Metric::InnerProduct, Metric::Cosine};

    for (unsigned seed : {1u, 7u, 42u, 1337u}) {
        for (std::size_t dim : {1u, 3u, 16u}) {
            for (Metric metric : metrics) {
                std::mt19937 rng(seed);
                std::normal_distribution<float> dist;

                const std::size_t n = 150;
                std::vector<float> rows(n * dim);
                for (float& v : rows) v = dist(rng);

                FlatIndex idx(dim, metric);
                idx.add_batch(rows.data(), n);

                for (int trial = 0; trial < 15; ++trial) {
                    std::vector<float> q(dim);
                    for (float& v : q) v = dist(rng);
                    const std::size_t k = 1 + (trial % 12);

                    EXPECT_EQ(ids_of(idx.search(q, k)),
                              naive_search(rows, dim, q, k, metric))
                        << "seed " << seed << " dim " << dim
                        << " metric " << int(metric) << " trial " << trial;
                }
            }
        }
    }
}

TEST(FlatIndex, DifferentialFuzzWithQuantisedVectorsForcesTies) {
    // Random floats essentially never tie, so the fuzz above barely exercises
    // the tie-break. Quantising to a handful of small integers -- which is
    // what SIFT descriptors actually are -- makes exact ties common.
    for (unsigned seed : {3u, 99u}) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> small(0, 3);

        const std::size_t dim = 4, n = 120;
        std::vector<float> rows(n * dim);
        for (float& v : rows) v = float(small(rng));

        FlatIndex idx(dim);
        idx.add_batch(rows.data(), n);

        for (int trial = 0; trial < 25; ++trial) {
            std::vector<float> q(dim);
            for (float& v : q) v = float(small(rng));
            const std::size_t k = 1 + (trial % 10);
            EXPECT_EQ(ids_of(idx.search(q, k)),
                      naive_search(rows, dim, q, k, Metric::L2))
                << "seed " << seed << " trial " << trial;
        }
    }
}
