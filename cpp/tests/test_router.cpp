// Tests for the neural router: weight loading, the forward pass, and its
// integration into HnswIndex.
//
// Weights are built by hand here rather than produced by the Python trainer,
// so these tests stay independent of it: a bug in training must not be able to
// make a loading bug invisible, and vice versa. The cross-language agreement
// between the two implementations is checked separately in tests/test_router.py,
// where both are available at once.

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "index/flat.hpp"
#include "index/hnsw.hpp"
#include "index/router.hpp"

using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;
using hylis::index::NeuralRouter;

namespace {

std::string join(const std::vector<double>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(values[i]);
    }
    return out;
}

// A router whose weights are chosen so the answer is checkable by hand.
std::string router_blob(std::size_t dim, std::size_t hidden, std::size_t clusters,
                        const std::vector<double>& w1, const std::vector<double>& b1,
                        const std::vector<double>& w2, const std::vector<double>& b2,
                        const std::vector<double>& medoids) {
    return "{\"version\":1,\"dim\":" + std::to_string(dim) +
           ",\"hidden\":" + std::to_string(hidden) +
           ",\"clusters\":" + std::to_string(clusters) +
           ",\"w1\":[" + join(w1) + "],\"b1\":[" + join(b1) +
           "],\"w2\":[" + join(w2) + "],\"b2\":[" + join(b2) +
           "],\"medoids\":[" + join(medoids) + "]}";
}

// dim=2, hidden=2, clusters=2. The first layer is the identity, so the logits
// are just a relu'd copy of the input and the predicted cluster is simply
// whichever coordinate is larger.
NeuralRouter identity_router() {
    return NeuralRouter::from_json(router_blob(
        /*dim=*/2, /*hidden=*/2, /*clusters=*/2,
        /*w1=*/{1, 0, 0, 1}, /*b1=*/{0, 0},
        /*w2=*/{1, 0, 0, 1}, /*b2=*/{0, 0},
        /*medoids=*/{0, 1}));
}

struct Corpus {
    std::vector<float> base, queries;
    std::size_t n, dim, n_queries;
};

Corpus make_corpus(std::size_t n, std::size_t dim, std::size_t n_queries,
                   unsigned seed, std::size_t clusters = 20) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss;
    std::vector<float> centres(clusters * dim);
    for (float& v : centres) v = gauss(rng);
    std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);

    Corpus c{std::vector<float>(n * dim), std::vector<float>(n_queries * dim),
             n, dim, n_queries};
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t k = pick(rng);
        for (std::size_t j = 0; j < dim; ++j) {
            c.base[i * dim + j] = centres[k * dim + j] + 0.15f * gauss(rng);
        }
    }
    for (std::size_t i = 0; i < n_queries; ++i) {
        const std::size_t k = pick(rng);
        for (std::size_t j = 0; j < dim; ++j) {
            c.queries[i * dim + j] = centres[k * dim + j] + 0.15f * gauss(rng);
        }
    }
    return c;
}

// A router that sends every query to cluster 0, whose medoid is node 0. Useless
// but valid — exactly what is needed to prove correctness does not depend on
// routing quality.
NeuralRouter constant_router(std::size_t dim, std::size_t clusters,
                             std::uint32_t medoid) {
    std::vector<double> w1(dim * 2, 0.0), b1{0, 0};
    std::vector<double> w2(2 * clusters, 0.0), b2(clusters, 0.0);
    b2[0] = 1.0;  // cluster 0 always wins
    std::vector<double> medoids(clusters, static_cast<double>(medoid));
    return NeuralRouter::from_json(
        router_blob(dim, 2, clusters, w1, b1, w2, b2, medoids));
}

std::vector<std::int64_t> ids_of(const std::vector<Neighbor>& ns) {
    std::vector<std::int64_t> out;
    for (const Neighbor& n : ns) out.push_back(n.id);
    return out;
}

}  // namespace

// --------------------------------------------------------------------------
// Loading
// --------------------------------------------------------------------------

TEST(NeuralRouter, DefaultConstructedIsEmpty) {
    NeuralRouter router;
    EXPECT_TRUE(router.empty());
    EXPECT_EQ(router.clusters(), 0u);
}

TEST(NeuralRouter, ParsesShapeAndWeights) {
    const NeuralRouter router = identity_router();
    EXPECT_EQ(router.dim(), 2u);
    EXPECT_EQ(router.hidden(), 2u);
    EXPECT_EQ(router.clusters(), 2u);
    EXPECT_EQ(router.medoid(0), 0u);
    EXPECT_EQ(router.medoid(1), 1u);
    EXPECT_FALSE(router.empty());
}

TEST(NeuralRouter, RejectsMisShapedWeights) {
    // The failure this guards against is silent: a weight matrix of the wrong
    // size would still multiply, still produce logits, and route badly forever.
    EXPECT_THROW(NeuralRouter::from_json(router_blob(2, 2, 2, {1, 0, 0}, {0, 0},
                                                     {1, 0, 0, 1}, {0, 0}, {0, 1})),
                 std::runtime_error);
    EXPECT_THROW(NeuralRouter::from_json(router_blob(2, 2, 2, {1, 0, 0, 1}, {0},
                                                     {1, 0, 0, 1}, {0, 0}, {0, 1})),
                 std::runtime_error);
    EXPECT_THROW(NeuralRouter::from_json(router_blob(2, 2, 2, {1, 0, 0, 1}, {0, 0},
                                                     {1, 0, 0, 1}, {0, 0}, {0})),
                 std::runtime_error);
}

TEST(NeuralRouter, RejectsZeroDimensions) {
    EXPECT_THROW(NeuralRouter::from_json(router_blob(0, 2, 2, {}, {0, 0},
                                                     {1, 0, 0, 1}, {0, 0}, {0, 1})),
                 std::runtime_error);
}

TEST(NeuralRouter, RejectsMalformedJson) {
    EXPECT_THROW(NeuralRouter::from_json("{\"dim\":2,"), std::runtime_error);
    EXPECT_THROW(NeuralRouter::from_json("{}"), std::runtime_error);
    EXPECT_THROW(NeuralRouter::from_json("not json at all"), std::runtime_error);
}

TEST(NeuralRouter, IgnoresUnknownFields) {
    // Forward compatibility: a file written by a later trainer must still load.
    const std::string blob =
        "{\"version\":2,\"trained_at\":\"yesterday\",\"notes\":[1,2,3],"
        "\"dim\":2,\"hidden\":2,\"clusters\":2,"
        "\"w1\":[1,0,0,1],\"b1\":[0,0],\"w2\":[1,0,0,1],\"b2\":[0,0],"
        "\"medoids\":[0,1]}";
    const NeuralRouter router = NeuralRouter::from_json(blob);
    EXPECT_EQ(router.clusters(), 2u);
}

TEST(NeuralRouter, LoadingAMissingFileThrows) {
    EXPECT_THROW(NeuralRouter::load(std::string(HYLIS_TEST_TMP) + "/no_such_router.json"),
                 std::runtime_error);
}

// --------------------------------------------------------------------------
// Forward pass
// --------------------------------------------------------------------------

TEST(NeuralRouter, PredictsTheLargerCoordinate) {
    const NeuralRouter router = identity_router();
    std::vector<std::uint32_t> out;

    const std::vector<float> a{3.0f, 1.0f};
    router.predict(a.data(), 1, out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 0u);

    const std::vector<float> b{1.0f, 5.0f};
    router.predict(b.data(), 1, out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 1u);
}

TEST(NeuralRouter, ReluClampsNegativeInputs) {
    // Both coordinates negative means both hidden units are clamped to zero,
    // so the logits tie and the tie-break must still yield a valid cluster.
    const NeuralRouter router = identity_router();
    std::vector<std::uint32_t> out;
    const std::vector<float> q{-4.0f, -9.0f};
    router.predict(q.data(), 1, out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_LT(out[0], 2u);
    EXPECT_FLOAT_EQ(router.last_logits()[0], 0.0f);
    EXPECT_FLOAT_EQ(router.last_logits()[1], 0.0f);
}

TEST(NeuralRouter, TopPReturnsDistinctClustersBestFirst) {
    const NeuralRouter router = identity_router();
    std::vector<std::uint32_t> out;
    const std::vector<float> q{7.0f, 2.0f};

    router.predict(q.data(), 2, out);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0u);
    EXPECT_EQ(out[1], 1u);
    EXPECT_NE(out[0], out[1]);
}

TEST(NeuralRouter, TopPIsClampedToTheClusterCount) {
    const NeuralRouter router = identity_router();
    std::vector<std::uint32_t> out;
    const std::vector<float> q{1.0f, 2.0f};
    router.predict(q.data(), 99, out);
    EXPECT_EQ(out.size(), 2u);

    router.predict(q.data(), 0, out);
    EXPECT_TRUE(out.empty());
}

TEST(NeuralRouter, EntryPointsDeduplicateSharedMedoids) {
    // Two clusters can legitimately share a medoid on degenerate data; seeding
    // the beam with the same node twice would waste an entry slot.
    const NeuralRouter router = NeuralRouter::from_json(router_blob(
        2, 2, 2, {1, 0, 0, 1}, {0, 0}, {1, 0, 0, 1}, {0, 0}, {5, 5}));
    std::vector<std::uint32_t> out;
    const std::vector<float> q{1.0f, 2.0f};
    router.entry_points(q.data(), 2, out);
    EXPECT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 5u);
}

TEST(NeuralRouter, MedoidOutOfRangeThrows) {
    const NeuralRouter router = identity_router();
    EXPECT_THROW(router.medoid(99), std::out_of_range);
}

// --------------------------------------------------------------------------
// Compatibility with the index it is attached to
// --------------------------------------------------------------------------

TEST(NeuralRouter, RejectsAnIndexOfADifferentDimension) {
    const Corpus c = make_corpus(200, 8, 1, 1, 5);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_THROW(graph.set_router(identity_router()), std::invalid_argument);
}

TEST(NeuralRouter, RejectsMedoidsBeyondTheCorpus) {
    // The signature of a router trained on a different, larger corpus.
    const Corpus c = make_corpus(50, 4, 1, 2, 5);
    HnswIndex graph(4);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_THROW(graph.set_router(constant_router(4, 4, /*medoid=*/9999)),
                 std::invalid_argument);
}

TEST(NeuralRouter, AttachingAndClearing) {
    const Corpus c = make_corpus(300, 4, 1, 3, 5);
    HnswIndex graph(4);
    graph.add_batch(c.base.data(), c.n);

    EXPECT_FALSE(graph.has_router());
    graph.set_router(constant_router(4, 8, 0));
    EXPECT_TRUE(graph.has_router());
    graph.clear_router();
    EXPECT_FALSE(graph.has_router());
}

// --------------------------------------------------------------------------
// Behaviour inside a search
// --------------------------------------------------------------------------

TEST(RoutedSearch, WithoutARouterUseRouterFallsBackToDescent) {
    // Adding the feature must not be able to change the baseline.
    const Corpus c = make_corpus(2000, 8, 20, 5);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    ASSERT_FALSE(graph.has_router());

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const float* q = c.queries.data() + i * c.dim;
        EXPECT_EQ(ids_of(graph.search(q, 10, 50, /*use_router=*/true)),
                  ids_of(graph.search(q, 10, 50, /*use_router=*/false)));
    }
}

TEST(RoutedSearch, RemainsCorrectWithADeliberatelyUselessRouter) {
    // The safety property the whole design rests on: routing decides where the
    // search *starts*, never what it is allowed to return. A router that sends
    // every query to the same arbitrary node may cost recall; it must not
    // return anything invalid, unsorted, or duplicated.
    const Corpus c = make_corpus(3000, 8, 30, 7);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.set_router(constant_router(8, 16, /*medoid=*/0));

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const auto found = graph.search(c.queries.data() + i * c.dim, 10, 50, true);
        EXPECT_EQ(found.size(), 10u);
        for (std::size_t j = 0; j < found.size(); ++j) {
            EXPECT_GE(found[j].id, 0);
            EXPECT_LT(found[j].id, static_cast<std::int64_t>(c.n));
            if (j > 0) { EXPECT_LE(found[j - 1].score, found[j].score); }
        }
    }
}

TEST(RoutedSearch, IsDeterministic) {
    const Corpus c = make_corpus(1500, 8, 10, 11);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.set_router(constant_router(8, 16, 3));

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const float* q = c.queries.data() + i * c.dim;
        EXPECT_EQ(ids_of(graph.search(q, 10, 50, true)),
                  ids_of(graph.search(q, 10, 50, true)));
    }
}

TEST(RoutedSearch, ReportsZeroRoutingVisitsWhereDescentReportsSome) {
    // The measurement that makes the comparison interpretable: the descent
    // spends graph visits getting to layer 0, the router spends arithmetic.
    const Corpus c = make_corpus(5000, 8, 5, 13);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.set_router(constant_router(8, 16, 0));

    const float* q = c.queries.data();
    graph.search(q, 10, 50, /*use_router=*/false);
    const std::size_t descent_routing = graph.last_routing_visited();
    graph.search(q, 10, 50, /*use_router=*/true);
    EXPECT_EQ(graph.last_routing_visited(), 0u);
    EXPECT_GT(descent_routing, 0u);
}

TEST(RoutedSearch, KeepingTheGlobalEntryAddsAFallbackStart) {
    const Corpus c = make_corpus(2000, 8, 20, 17);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.set_router(constant_router(8, 16, 0));
    graph.set_router_keeps_global_entry(true);

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const auto found = graph.search(c.queries.data() + i * c.dim, 10, 50, true);
        EXPECT_EQ(found.size(), 10u);
    }
}

TEST(RoutedSearch, FilteredSearchStillHonoursTheFilter) {
    const Corpus c = make_corpus(2000, 8, 10, 19);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.set_router(constant_router(8, 16, 0));

    std::vector<std::int64_t> allowed;
    for (std::int64_t i = 0; i < 2000; i += 5) allowed.push_back(i);

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const auto found =
            graph.search_filtered(c.queries.data() + i * c.dim, 10, allowed, 100, true);
        for (const Neighbor& n : found) {
            EXPECT_EQ(n.id % 5, 0) << "returned an id outside the filter";
        }
    }
}

TEST(RoutedSearch, TopPIsSettable) {
    const Corpus c = make_corpus(1000, 8, 5, 23);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.set_router(constant_router(8, 16, 0));

    EXPECT_EQ(graph.router_top_p(), 2u);
    graph.set_router_top_p(4);
    EXPECT_EQ(graph.router_top_p(), 4u);
    graph.set_router_top_p(0);
    EXPECT_EQ(graph.router_top_p(), 1u) << "top_p must stay at least 1";
}

// --------------------------------------------------------------------------
// flat_only: the full-replacement configuration
// --------------------------------------------------------------------------

TEST(FlatOnlyGraph, HasNoLayersAboveZero) {
    const Corpus c = make_corpus(3000, 8, 5, 29);
    HnswIndex graph(8, Metric::L2, 16, 200, 100, /*flat_only=*/true);
    graph.add_batch(c.base.data(), c.n);

    const auto s = graph.stats();
    EXPECT_EQ(s.levels, 1u);
    ASSERT_EQ(s.layer_population.size(), 1u);
    EXPECT_EQ(s.layer_population[0], c.n);
    EXPECT_EQ(s.edges, s.layer0_edges) << "no edge may live above layer 0";
    EXPECT_NO_THROW(graph.validate());
}

TEST(FlatOnlyGraph, IsCheaperThanAFullHierarchy) {
    const Corpus c = make_corpus(5000, 8, 5, 31);
    HnswIndex full(8, Metric::L2, 16, 200, 100, false);
    HnswIndex flat(8, Metric::L2, 16, 200, 100, true);
    full.add_batch(c.base.data(), c.n);
    flat.add_batch(c.base.data(), c.n);

    EXPECT_LT(flat.stats().graph_bytes, full.stats().graph_bytes)
        << "dropping the upper layers must actually save memory";
}

TEST(FlatOnlyGraph, StillAnswersCorrectlyWithoutARouter) {
    // With no hierarchy and no router there is only the global entry point, so
    // search still works — just from a fixed start.
    const Corpus c = make_corpus(2000, 8, 20, 37);
    FlatIndex exact(8);
    exact.add_batch(c.base.data(), c.n);
    HnswIndex graph(8, Metric::L2, 16, 200, 100, true);
    graph.add_batch(c.base.data(), c.n);

    std::size_t hits = 0;
    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const float* q = c.queries.data() + i * c.dim;
        const auto truth = ids_of(exact.search(q, 10));
        for (const Neighbor& n : graph.search(q, 10, 200)) {
            if (std::find(truth.begin(), truth.end(), n.id) != truth.end()) ++hits;
        }
    }
    EXPECT_GT(static_cast<double>(hits) / (c.n_queries * 10), 0.8);
}
