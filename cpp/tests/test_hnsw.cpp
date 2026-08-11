// Tests for the HNSW graph index.
//
// This is the first structure in the project whose answers are *approximate*,
// and the suite has to change shape for it. There is no equality assertion to
// make about search results: "is it correct?" has no meaning for an index
// that is allowed to miss neighbours. What can be asserted is
//
//   * the graph's structure -- degrees, levels, reachability -- which is
//     still strictly yes/no, and
//   * bounds and monotonicity of search quality: recall must clear a floor,
//     and must not fall when ef rises.
//
// Recall is measured against FlatIndex, which module 3 built to be the exact
// oracle for exactly this purpose.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "index/flat.hpp"
#include "index/hnsw.hpp"

using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;

namespace {

// Clustered vectors, because uniform random points in high dimensions are all
// roughly equidistant -- there is no neighbourhood structure for a graph index
// to exploit, and recall measured on them means nothing.
struct Corpus {
    std::vector<float> base;
    std::vector<float> queries;
    std::size_t n, dim, n_queries;
};

Corpus make_corpus(std::size_t n, std::size_t dim, std::size_t n_queries,
                   unsigned seed, std::size_t clusters = 50) {
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

std::vector<std::int64_t> ids_of(const std::vector<Neighbor>& ns) {
    std::vector<std::int64_t> out;
    out.reserve(ns.size());
    for (const Neighbor& n : ns) out.push_back(n.id);
    return out;
}

// Mean fraction of the true k nearest that the graph actually returned.
double recall_at_k(const HnswIndex& graph, const FlatIndex& exact,
                   const Corpus& corpus, std::size_t k, std::size_t ef) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < corpus.n_queries; ++i) {
        const float* q = corpus.queries.data() + i * corpus.dim;
        const std::vector<std::int64_t> truth = ids_of(exact.search(q, k));
        const std::set<std::int64_t> truth_set(truth.begin(), truth.end());
        for (const Neighbor& n : graph.search(q, k, ef)) {
            if (truth_set.count(n.id)) ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(corpus.n_queries * k);
}

}  // namespace

// --------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------

TEST(Hnsw, StartsEmpty) {
    HnswIndex graph(8);
    EXPECT_EQ(graph.size(), 0u);
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.dim(), 8u);
    EXPECT_EQ(graph.entry_point(), -1);
    EXPECT_TRUE(graph.search(std::vector<float>(8, 0.0f), 5).empty());
    EXPECT_NO_THROW(graph.validate());
}

TEST(Hnsw, RejectsNonsenseParameters) {
    EXPECT_THROW(HnswIndex(0), std::invalid_argument);
    EXPECT_THROW(HnswIndex(4, Metric::L2, 1), std::invalid_argument);
    EXPECT_THROW(HnswIndex(4, Metric::L2, 16, 0), std::invalid_argument);
}

TEST(Hnsw, WrongDimensionRejected) {
    HnswIndex graph(3);
    EXPECT_THROW(graph.add(std::vector<float>{1, 2}), std::invalid_argument);
    graph.add(std::vector<float>{1, 2, 3});
    EXPECT_THROW(graph.search(std::vector<float>{1, 2}, 1), std::invalid_argument);
}

TEST(Hnsw, SingleVector) {
    HnswIndex graph(4);
    graph.add(std::vector<float>{1, 2, 3, 4});
    EXPECT_EQ(graph.size(), 1u);
    EXPECT_EQ(graph.entry_point(), 0);
    EXPECT_NO_THROW(graph.validate());

    const auto found = graph.search(std::vector<float>{1, 2, 3, 4}, 5);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0].id, 0);
    EXPECT_NEAR(found[0].score, 0.0f, 1e-5f);
}

TEST(Hnsw, ClearResetsEverything) {
    const Corpus c = make_corpus(300, 8, 1, 1);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    graph.clear();

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.entry_point(), -1);
    EXPECT_TRUE(graph.search(c.queries, 5).empty());
    EXPECT_NO_THROW(graph.validate());
}

TEST(Hnsw, CanKeepAddingAfterSearching) {
    const Corpus c = make_corpus(400, 8, 1, 2);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), 200);
    EXPECT_FALSE(graph.search(c.queries, 5).empty());

    graph.add_batch(c.base.data() + 200 * 8, 200);
    EXPECT_EQ(graph.size(), 400u);
    EXPECT_NO_THROW(graph.validate());
}

TEST(Hnsw, MinimumMStillBuildsAValidGraph) {
    const Corpus c = make_corpus(500, 8, 1, 3);
    HnswIndex graph(8, Metric::L2, /*M=*/2, /*ef_construction=*/50);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_NO_THROW(graph.validate());
    EXPECT_EQ(graph.search(c.queries, 5).size(), 5u);
}

TEST(Hnsw, IdenticalVectorsDoNotBreakTheGraph) {
    // Every distance is zero, so every tie-break and every heuristic
    // comparison hits its degenerate path at once.
    HnswIndex graph(4);
    for (int i = 0; i < 200; ++i) graph.add(std::vector<float>{1, 1, 1, 1});
    EXPECT_NO_THROW(graph.validate());

    const auto found = graph.search(std::vector<float>{1, 1, 1, 1}, 10);
    EXPECT_EQ(found.size(), 10u);
    for (const Neighbor& n : found) EXPECT_NEAR(n.score, 0.0f, 1e-5f);
}

TEST(Hnsw, ReachabilityIsCompleteAtSensibleParameters) {
    // A node with no in-edges can never be returned, whatever ef is, so this
    // is the practical ceiling on recall. At M=16 it should be everything.
    const Corpus c = make_corpus(5000, 16, 1, 101);
    HnswIndex graph(16, Metric::L2, 16, 200);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_EQ(graph.reachable(), c.n);
    EXPECT_EQ(graph.stats().reachable, c.n);
}

TEST(Hnsw, ReachabilityDegradesAtVerySmallM) {
    // Documents a real limitation rather than asserting it away. Links are
    // added in both directions, but pruning a full neighbour list can drop the
    // back-link — and with M=2 the lists are tiny and prune constantly, so a
    // few percent of nodes end up stranded with no in-edges at all. Nothing in
    // the algorithm prevents this; it is why M is not a free parameter.
    const Corpus c = make_corpus(500, 8, 1, 3);

    HnswIndex tiny(8, Metric::L2, /*M=*/2, /*ef_construction=*/50);
    tiny.add_batch(c.base.data(), c.n);

    HnswIndex sensible(8, Metric::L2, /*M=*/16, /*ef_construction=*/50);
    sensible.add_batch(c.base.data(), c.n);

    EXPECT_LT(tiny.reachable(), c.n) << "M=2 is expected to strand some nodes";
    EXPECT_EQ(sensible.reachable(), c.n);
}

// --------------------------------------------------------------------------
// Graph structure
// --------------------------------------------------------------------------

TEST(Hnsw, ValidateHoldsAcrossSizesAndParameters) {
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{17},
                          std::size_t{500}, std::size_t{2000}}) {
        for (std::size_t m : {std::size_t{2}, std::size_t{8}, std::size_t{16}}) {
            const Corpus c = make_corpus(n, 8, 1, 11, std::min<std::size_t>(n, 20));
            HnswIndex graph(8, Metric::L2, m, 100);
            graph.add_batch(c.base.data(), c.n);
            ASSERT_NO_THROW(graph.validate()) << "n=" << n << " M=" << m;
            EXPECT_EQ(graph.size(), n);
        }
    }
}

TEST(Hnsw, LayerPopulationsDecayGeometrically) {
    // Levels are drawn from an exponential distribution with mL = 1/ln(M), so
    // each layer should hold roughly 1/M of the one below. That decay is what
    // makes the upper layers a cheap highway rather than a second full index.
    const Corpus c = make_corpus(20000, 8, 1, 13);
    HnswIndex graph(8, Metric::L2, /*M=*/16, /*ef_construction=*/64);
    graph.add_batch(c.base.data(), c.n);

    const auto s = graph.stats();
    ASSERT_GE(s.layer_population.size(), 3u) << "20k nodes should reach 3+ layers";
    EXPECT_EQ(s.layer_population[0], c.n) << "layer 0 must hold every node";

    for (std::size_t layer = 1; layer < s.layer_population.size(); ++layer) {
        EXPECT_LT(s.layer_population[layer], s.layer_population[layer - 1])
            << "layer " << layer << " is not sparser than the one below";
    }
    // ~1/16 expected; allow a wide band so this fails on a broken
    // distribution rather than on sampling noise.
    const double ratio = static_cast<double>(s.layer_population[1]) /
                         static_cast<double>(s.layer_population[0]);
    EXPECT_GT(ratio, 1.0 / 64.0);
    EXPECT_LT(ratio, 1.0 / 4.0);
}

TEST(Hnsw, DegreesStayWithinTheirCaps) {
    const Corpus c = make_corpus(3000, 8, 1, 17);
    HnswIndex graph(8, Metric::L2, /*M=*/8, /*ef_construction=*/100);
    graph.add_batch(c.base.data(), c.n);

    const auto s = graph.stats();
    EXPECT_LE(s.max_degree_l0, 16u) << "layer 0 cap is 2*M";
    EXPECT_GT(s.mean_degree_l0, 1.0) << "a graph this size should be well connected";
    EXPECT_NO_THROW(graph.validate());
}

TEST(Hnsw, GraphOverheadIsReportedSeparatelyFromVectors) {
    const Corpus c = make_corpus(2000, 16, 1, 19);
    HnswIndex graph(16);
    graph.add_batch(c.base.data(), c.n);

    const auto s = graph.stats();
    EXPECT_GT(s.graph_bytes, 0u);
    EXPECT_GT(s.total_bytes, s.graph_bytes);
    EXPECT_EQ(s.nodes, c.n);
    EXPECT_GT(s.edges, s.layer0_edges) << "some edges must live above layer 0";
}

TEST(Hnsw, IsDeterministicForASeed) {
    const Corpus c = make_corpus(800, 8, 5, 23);

    auto build = [&](std::uint64_t seed) {
        HnswIndex graph(8, Metric::L2, 16, 100, seed);
        graph.add_batch(c.base.data(), c.n);
        return ids_of(graph.search(c.queries.data(), 10, 40));
    };

    EXPECT_EQ(build(7), build(7));
    // Different seeds give different level assignments, hence a different
    // graph. Results may still coincide, so compare the structure instead.
    HnswIndex a(8, Metric::L2, 16, 100, 7);
    HnswIndex b(8, Metric::L2, 16, 100, 8);
    a.add_batch(c.base.data(), c.n);
    b.add_batch(c.base.data(), c.n);
    EXPECT_NE(a.entry_point(), b.entry_point());
}

// --------------------------------------------------------------------------
// Search quality -- bounds and monotonicity, never equality
// --------------------------------------------------------------------------

TEST(Hnsw, SelfRetrievalReturnsTheVectorItself) {
    const Corpus c = make_corpus(2000, 16, 1, 29);
    HnswIndex graph(16);
    graph.add_batch(c.base.data(), c.n);

    std::size_t found_self = 0;
    for (std::size_t i = 0; i < c.n; i += 17) {
        const auto got = graph.search(graph.vector_at(static_cast<std::int64_t>(i)), 1, 50);
        ASSERT_EQ(got.size(), 1u);
        if (got[0].id == static_cast<std::int64_t>(i)) ++found_self;
    }
    const std::size_t probed = (c.n + 16) / 17;
    EXPECT_GT(found_self * 100, probed * 99)
        << "a stored vector should almost always retrieve itself";
}

TEST(Hnsw, RecallRisesMonotonicallyWithEf) {
    // The one property a user actually relies on: ef is the quality knob, and
    // turning it up must never make results worse.
    const Corpus c = make_corpus(5000, 16, 100, 31);
    FlatIndex exact(16);
    exact.add_batch(c.base.data(), c.n);
    HnswIndex graph(16);
    graph.add_batch(c.base.data(), c.n);

    double previous = 0.0;
    for (std::size_t ef : {std::size_t{10}, std::size_t{20}, std::size_t{50},
                           std::size_t{100}, std::size_t{200}}) {
        const double recall = recall_at_k(graph, exact, c, 10, ef);
        EXPECT_GE(recall, previous - 1e-9)
            << "recall fell when ef rose to " << ef;
        previous = recall;
    }
    EXPECT_GT(previous, 0.99) << "ef=200 should be very close to exact";
}

TEST(Hnsw, ClearsARecallFloorAtDefaultParameters) {
    // M=16, efConstruction=200, ef=100 is the configuration the literature
    // reports ~0.95+ recall@10 for. A floor rather than an equality, so it
    // catches regressions without being brittle.
    const Corpus c = make_corpus(5000, 32, 100, 37);
    FlatIndex exact(32);
    exact.add_batch(c.base.data(), c.n);
    HnswIndex graph(32, Metric::L2, 16, 200);
    graph.add_batch(c.base.data(), c.n);

    EXPECT_GE(recall_at_k(graph, exact, c, 10, 100), 0.95);
}

TEST(Hnsw, VisitsFarFewerNodesThanAFullScan) {
    // The entire justification for the structure.
    const Corpus c = make_corpus(10000, 16, 20, 41);
    HnswIndex graph(16);
    graph.add_batch(c.base.data(), c.n);

    std::size_t visited = 0;
    for (std::size_t i = 0; i < c.n_queries; ++i) {
        graph.search(c.queries.data() + i * c.dim, 10, 50);
        visited += graph.last_visited();
    }
    const double mean = static_cast<double>(visited) / static_cast<double>(c.n_queries);
    EXPECT_LT(mean, static_cast<double>(c.n) / 5.0)
        << "visited " << mean << " of " << c.n << " nodes -- no better than scanning";
}

TEST(Hnsw, ResultsAreOrderedBestFirst) {
    const Corpus c = make_corpus(1000, 8, 5, 43);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const auto found = graph.search(c.queries.data() + i * c.dim, 10, 50);
        for (std::size_t j = 1; j < found.size(); ++j) {
            EXPECT_LE(found[j - 1].score, found[j].score);
        }
    }
}

TEST(Hnsw, KLargerThanTheCorpusIsClamped) {
    const Corpus c = make_corpus(40, 8, 1, 47, 5);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_LE(graph.search(c.queries, 500, 500).size(), 40u);
    EXPECT_TRUE(graph.search(c.queries, 0).empty());
}

TEST(Hnsw, SelectionHeuristicBeatsNaiveNearestAtEqualEf) {
    // The claim Algorithm 4 makes, turned into a measurement. Keeping the M
    // nearest candidates fills a node's links with mutual near-duplicates and
    // leaves it no long-range edges to escape its own cluster on; the
    // heuristic trades some of that closeness for reach.
    const Corpus c = make_corpus(5000, 16, 100, 53);
    FlatIndex exact(16);
    exact.add_batch(c.base.data(), c.n);

    HnswIndex heuristic(16, Metric::L2, 8, 100);
    heuristic.add_batch(c.base.data(), c.n);

    HnswIndex naive(16, Metric::L2, 8, 100);
    naive.set_use_heuristic(false);
    naive.add_batch(c.base.data(), c.n);

    const double with_heuristic = recall_at_k(heuristic, exact, c, 10, 20);
    const double without = recall_at_k(naive, exact, c, 10, 20);
    EXPECT_GE(with_heuristic, without)
        << "heuristic " << with_heuristic << " vs naive " << without;
}

// --------------------------------------------------------------------------
// Metrics
// --------------------------------------------------------------------------

TEST(Hnsw, InnerProductPrefersLargerDots) {
    HnswIndex graph(2, Metric::InnerProduct);
    graph.add(std::vector<float>{1, 0});
    graph.add(std::vector<float>{5, 0});
    graph.add(std::vector<float>{2, 0});

    const auto found = graph.search(std::vector<float>{1, 0}, 3, 10);
    ASSERT_FALSE(found.empty());
    EXPECT_EQ(found[0].id, 1);
    EXPECT_FLOAT_EQ(found[0].score, 5.0f);
}

TEST(Hnsw, CosineIgnoresMagnitude) {
    const Corpus c = make_corpus(1000, 16, 1, 59);
    HnswIndex graph(16, Metric::Cosine);
    graph.add_batch(c.base.data(), c.n);

    std::vector<float> query(c.queries.begin(), c.queries.begin() + 16);
    std::vector<float> scaled(query);
    for (float& v : scaled) v *= 47.0f;

    EXPECT_EQ(ids_of(graph.search(query, 10, 50)),
              ids_of(graph.search(scaled, 10, 50)));
}

TEST(Hnsw, CosineMatchesTheExactIndexClosely) {
    const Corpus c = make_corpus(3000, 16, 50, 61);
    FlatIndex exact(16, Metric::Cosine);
    exact.add_batch(c.base.data(), c.n);
    HnswIndex graph(16, Metric::Cosine);
    graph.add_batch(c.base.data(), c.n);

    EXPECT_GE(recall_at_k(graph, exact, c, 10, 100), 0.9);
}

// --------------------------------------------------------------------------
// Filtered search
// --------------------------------------------------------------------------

TEST(Hnsw, FilteredSearchReturnsOnlyAllowedIds) {
    const Corpus c = make_corpus(2000, 8, 10, 67);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);

    std::vector<std::int64_t> allowed;
    for (std::int64_t i = 0; i < 2000; i += 3) allowed.push_back(i);
    const std::set<std::int64_t> allowed_set(allowed.begin(), allowed.end());

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const auto found =
            graph.search_filtered(c.queries.data() + i * c.dim, 10, allowed, 100);
        EXPECT_FALSE(found.empty());
        for (const Neighbor& n : found) {
            EXPECT_TRUE(allowed_set.count(n.id)) << "returned disallowed id " << n.id;
        }
        for (std::size_t j = 1; j < found.size(); ++j) {
            EXPECT_LE(found[j - 1].score, found[j].score);
        }
    }
}

TEST(Hnsw, EmptyFilterReturnsNothing) {
    const Corpus c = make_corpus(500, 8, 1, 71);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_TRUE(graph.search_filtered(c.queries, 5, {}).empty());
}

TEST(Hnsw, FilteredSearchRejectsUnknownIds) {
    const Corpus c = make_corpus(100, 8, 1, 73, 5);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);
    EXPECT_THROW(graph.search_filtered(c.queries, 5, {9999}), std::out_of_range);
    EXPECT_THROW(graph.search_filtered(c.queries, 5, {-1}), std::out_of_range);
}

TEST(Hnsw, FilteringEverythingMatchesAnUnfilteredSearch) {
    const Corpus c = make_corpus(1000, 8, 5, 79);
    HnswIndex graph(8);
    graph.add_batch(c.base.data(), c.n);

    std::vector<std::int64_t> all(1000);
    for (std::int64_t i = 0; i < 1000; ++i) all[i] = i;

    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const float* q = c.queries.data() + i * c.dim;
        EXPECT_EQ(ids_of(graph.search(q, 10, 50)),
                  ids_of(graph.search_filtered(q, 10, all, 50)));
    }
}

TEST(Hnsw, FilteredRecallDegradesAsThePredicateTightens) {
    // Not a defect -- the mechanism the query planner exists to exploit. The
    // graph must traverse non-matching nodes to stay connected, so a tighter
    // filter means visiting more of the corpus to collect the same k, while a
    // filtered exhaustive scan gets *cheaper* in proportion. Somewhere those
    // cross; scripts/bench_vector.py measures where.
    const Corpus c = make_corpus(5000, 16, 20, 83);
    FlatIndex exact(16);
    exact.add_batch(c.base.data(), c.n);
    HnswIndex graph(16);
    graph.add_batch(c.base.data(), c.n);

    std::size_t visited_loose = 0, visited_tight = 0;
    for (std::size_t i = 0; i < c.n_queries; ++i) {
        const float* q = c.queries.data() + i * c.dim;

        std::vector<std::int64_t> loose, tight;
        for (std::int64_t j = 0; j < 5000; ++j) {
            if (j % 2 == 0) loose.push_back(j);
            if (j % 100 == 0) tight.push_back(j);
        }
        graph.search_filtered(q, 10, loose, 50);
        visited_loose += graph.last_visited();
        graph.search_filtered(q, 10, tight, 50);
        visited_tight += graph.last_visited();
    }
    EXPECT_GT(visited_tight, visited_loose)
        << "a tighter filter must force the traversal to work harder";
}
