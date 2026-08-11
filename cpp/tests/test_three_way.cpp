// Cross-implementation tests: exact scan, our HNSW, our HNSW with a router,
// and hnswlib.
//
// Two distinct jobs here.
//
// The first is *shared correctness*. Anything true of "a vector index" rather
// than of one particular index gets asserted for all of them: results are a
// subset of the corpus, sorted best-first, free of duplicates; k>n clamps;
// filters are honoured. Writing those once and running them over every
// implementation is what stops one of them drifting quietly.
//
// The second is *agreement*. Each index is separately graded against the exact
// oracle, but that is not sufficient — two indexes can each score 0.96 against
// the truth while overlapping far less with each other, which would mean they
// are missing different things and something is wrong. So they are also
// compared pairwise.
//
// hnswlib is a benchmark baseline only. It is here to answer whether our HNSW
// is competitive or a strawman, since a router that beats a slow baseline has
// proved nothing. All hnswlib tests compile out when it was not fetched.

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "index/flat.hpp"
#include "index/hnsw.hpp"
#include "index/router.hpp"

#ifdef HYLIS_HAS_HNSWLIB
#include "index/hnswlib_adapter.hpp"
using hylis::index::HnswlibIndex;
#endif

using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;
using hylis::index::NeuralRouter;

namespace {

constexpr std::size_t kN = 4000;
constexpr std::size_t kDim = 16;
constexpr std::size_t kQueries = 60;
constexpr std::size_t kClusters = 32;

struct Corpus {
    std::vector<float> base, queries;
};

const Corpus& corpus() {
    static const Corpus c = [] {
        std::mt19937 rng(4242);
        std::normal_distribution<float> gauss;
        std::vector<float> centres(kClusters * kDim);
        for (float& v : centres) v = gauss(rng);
        std::uniform_int_distribution<std::size_t> pick(0, kClusters - 1);

        Corpus out{std::vector<float>(kN * kDim), std::vector<float>(kQueries * kDim)};
        for (std::size_t i = 0; i < kN; ++i) {
            const std::size_t k = pick(rng);
            for (std::size_t j = 0; j < kDim; ++j) {
                out.base[i * kDim + j] = centres[k * kDim + j] + 0.15f * gauss(rng);
            }
        }
        for (std::size_t i = 0; i < kQueries; ++i) {
            const std::size_t k = pick(rng);
            for (std::size_t j = 0; j < kDim; ++j) {
                out.queries[i * kDim + j] = centres[k * kDim + j] + 0.15f * gauss(rng);
            }
        }
        return out;
    }();
    return c;
}

std::vector<std::int64_t> ids_of(const std::vector<Neighbor>& ns) {
    std::vector<std::int64_t> out;
    for (const Neighbor& n : ns) out.push_back(n.id);
    return out;
}

// A router built directly from the corpus: cluster centres are just evenly
// spaced base vectors, and the first layer projects onto them so the predicted
// cluster is the nearest centre. Crude next to the trained one, but it makes
// these tests independent of the Python trainer.
NeuralRouter geometric_router() {
    const Corpus& c = corpus();
    const std::size_t clusters = 32;
    const std::size_t stride = kN / clusters;

    std::string w1, b1, w2, b2, medoids;
    // hidden == clusters: hidden unit j scores the query against centre j,
    // as -(|centre|^2/2 - q.centre), then w2 is the identity.
    for (std::size_t i = 0; i < kDim; ++i) {
        for (std::size_t j = 0; j < clusters; ++j) {
            if (!w1.empty()) w1 += ",";
            w1 += std::to_string(c.base[(j * stride) * kDim + i]);
        }
    }
    for (std::size_t j = 0; j < clusters; ++j) {
        double sq = 0.0;
        for (std::size_t i = 0; i < kDim; ++i) {
            const double v = c.base[(j * stride) * kDim + i];
            sq += v * v;
        }
        if (!b1.empty()) b1 += ",";
        b1 += std::to_string(-0.5 * sq);
        if (!b2.empty()) b2 += ",";
        b2 += "0";
        if (!medoids.empty()) medoids += ",";
        medoids += std::to_string(j * stride);
    }
    for (std::size_t h = 0; h < clusters; ++h) {
        for (std::size_t k = 0; k < clusters; ++k) {
            if (!w2.empty()) w2 += ",";
            w2 += (h == k ? "1" : "0");
        }
    }

    return NeuralRouter::from_json(
        "{\"dim\":" + std::to_string(kDim) + ",\"hidden\":" + std::to_string(clusters) +
        ",\"clusters\":" + std::to_string(clusters) + ",\"w1\":[" + w1 + "],\"b1\":[" +
        b1 + "],\"w2\":[" + w2 + "],\"b2\":[" + b2 + "],\"medoids\":[" + medoids + "]}");
}

// Mean fraction of `truth` that `found` recovered, compared as sets per query.
double overlap(const std::vector<std::vector<std::int64_t>>& found,
               const std::vector<std::vector<std::int64_t>>& truth, std::size_t k) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < found.size(); ++i) {
        const std::set<std::int64_t> want(truth[i].begin(), truth[i].end());
        for (std::int64_t id : found[i]) {
            if (want.count(id)) ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(found.size() * k);
}

// Every implementation answers through this, so the shared assertions below
// can be written once.
using Searcher = std::function<std::vector<Neighbor>(const float*, std::size_t)>;

std::vector<std::pair<std::string, Searcher>> implementations() {
    static FlatIndex flat(kDim);
    static HnswIndex graph(kDim);
    static HnswIndex routed(kDim);
    static bool built = false;
    if (!built) {
        const Corpus& c = corpus();
        flat.add_batch(c.base.data(), kN);
        graph.add_batch(c.base.data(), kN);
        routed.add_batch(c.base.data(), kN);
        routed.set_router(geometric_router());
        built = true;
    }

    std::vector<std::pair<std::string, Searcher>> out;
    out.emplace_back("flat", [](const float* q, std::size_t k) {
        return flat.search(q, k);
    });
    out.emplace_back("hnsw", [](const float* q, std::size_t k) {
        return graph.search(q, k, 100);
    });
    out.emplace_back("routed", [](const float* q, std::size_t k) {
        return routed.search(q, k, 100, /*use_router=*/true);
    });
#ifdef HYLIS_HAS_HNSWLIB
    static HnswlibIndex lib(kDim, Metric::L2, kN);
    static bool lib_built = false;
    if (!lib_built) {
        lib.add_batch(corpus().base.data(), kN);
        lib_built = true;
    }
    out.emplace_back("hnswlib", [](const float* q, std::size_t k) {
        return lib.search(q, k, 100);
    });
#endif
    return out;
}

}  // namespace

// --------------------------------------------------------------------------
// Shared correctness: true of any vector index, asserted for all of them
// --------------------------------------------------------------------------

TEST(ThreeWay, ResultsAreValidSortedAndDistinct) {
    const Corpus& c = corpus();
    for (const auto& [name, search] : implementations()) {
        for (std::size_t i = 0; i < kQueries; ++i) {
            const auto found = search(c.queries.data() + i * kDim, 10);
            ASSERT_EQ(found.size(), 10u) << name << " query " << i;

            std::set<std::int64_t> seen;
            for (std::size_t j = 0; j < found.size(); ++j) {
                EXPECT_GE(found[j].id, 0) << name;
                EXPECT_LT(found[j].id, static_cast<std::int64_t>(kN)) << name;
                EXPECT_TRUE(seen.insert(found[j].id).second)
                    << name << " returned a duplicate id";
                if (j) {
                    EXPECT_LE(found[j - 1].score, found[j].score)
                        << name << " results are not sorted best-first";
                }
            }
        }
    }
}

TEST(ThreeWay, KLargerThanTheCorpusIsClamped) {
    const Corpus& c = corpus();
    for (const auto& [name, search] : implementations()) {
        EXPECT_LE(search(c.queries.data(), kN + 500).size(), kN) << name;
    }
}

TEST(ThreeWay, ZeroKReturnsNothing) {
    const Corpus& c = corpus();
    for (const auto& [name, search] : implementations()) {
        EXPECT_TRUE(search(c.queries.data(), 0).empty()) << name;
    }
}

TEST(ThreeWay, EveryImplementationClearsTheRecallFloor) {
    const Corpus& c = corpus();
    static FlatIndex exact(kDim);
    static bool built = false;
    if (!built) { exact.add_batch(c.base.data(), kN); built = true; }

    std::vector<std::vector<std::int64_t>> truth;
    for (std::size_t i = 0; i < kQueries; ++i) {
        truth.push_back(ids_of(exact.search(c.queries.data() + i * kDim, 10)));
    }

    for (const auto& [name, search] : implementations()) {
        std::vector<std::vector<std::int64_t>> got;
        for (std::size_t i = 0; i < kQueries; ++i) {
            got.push_back(ids_of(search(c.queries.data() + i * kDim, 10)));
        }
        EXPECT_GE(overlap(got, truth, 10), 0.90)
            << name << " fell below the recall floor against the exact oracle";
    }
}

// --------------------------------------------------------------------------
// Agreement: each against the oracle is not enough
// --------------------------------------------------------------------------

TEST(ThreeWay, ImplementationsAgreeWithEachOtherNotJustWithTheOracle) {
    const Corpus& c = corpus();
    const auto impls = implementations();

    std::vector<std::vector<std::vector<std::int64_t>>> results;
    for (const auto& [name, search] : impls) {
        std::vector<std::vector<std::int64_t>> got;
        for (std::size_t i = 0; i < kQueries; ++i) {
            got.push_back(ids_of(search(c.queries.data() + i * kDim, 10)));
        }
        results.push_back(std::move(got));
    }

    for (std::size_t a = 0; a < impls.size(); ++a) {
        for (std::size_t b = a + 1; b < impls.size(); ++b) {
            EXPECT_GE(overlap(results[a], results[b], 10), 0.85)
                << impls[a].first << " and " << impls[b].first
                << " each look accurate but disagree with one another, which "
                   "means they are missing different neighbours";
        }
    }
}

TEST(ThreeWay, EveryImplementationRetrievesAStoredVectorItself) {
    const Corpus& c = corpus();
    for (const auto& [name, search] : implementations()) {
        std::size_t hits = 0, probes = 0;
        for (std::size_t i = 0; i < kN; i += 137) {
            ++probes;
            const auto found = search(c.base.data() + i * kDim, 1);
            if (!found.empty() && found[0].id == static_cast<std::int64_t>(i)) ++hits;
        }
        EXPECT_GT(hits * 100, probes * 95) << name << " failed self-retrieval";
    }
}

// --------------------------------------------------------------------------
// hnswlib, the external reference
// --------------------------------------------------------------------------

#ifdef HYLIS_HAS_HNSWLIB

TEST(HnswlibBaseline, ScoresUseOurConventionNotHnswlibs) {
    // hnswlib returns squared L2; we return true distance. If the adapter did
    // not convert, every score comparison against it would silently be
    // comparing different quantities.
    const Corpus& c = corpus();
    FlatIndex exact(kDim);
    exact.add_batch(c.base.data(), kN);
    HnswlibIndex lib(kDim, Metric::L2, kN);
    lib.add_batch(c.base.data(), kN);

    const auto truth = exact.search(c.queries.data(), 1);
    const auto got = lib.search(c.queries.data(), 1, 200);
    ASSERT_FALSE(got.empty());
    ASSERT_FALSE(truth.empty());
    if (got[0].id == truth[0].id) {
        EXPECT_NEAR(got[0].score, truth[0].score, 1e-3f);
    }
}

TEST(HnswlibBaseline, OurImplementationIsCompetitive) {
    // The question this baseline exists to answer. Not "are we faster" — we
    // are a hand-written implementation against a heavily optimised one — but
    // "is our recall in the same league", because a router that beats a
    // strawman has proved nothing.
    const Corpus& c = corpus();
    FlatIndex exact(kDim);
    exact.add_batch(c.base.data(), kN);
    HnswIndex ours(kDim);
    ours.add_batch(c.base.data(), kN);
    HnswlibIndex lib(kDim, Metric::L2, kN);
    lib.add_batch(c.base.data(), kN);

    std::vector<std::vector<std::int64_t>> truth, mine, theirs;
    for (std::size_t i = 0; i < kQueries; ++i) {
        const float* q = c.queries.data() + i * kDim;
        truth.push_back(ids_of(exact.search(q, 10)));
        mine.push_back(ids_of(ours.search(q, 10, 50)));
        theirs.push_back(ids_of(lib.search(q, 10, 50)));
    }

    const double ours_recall = overlap(mine, truth, 10);
    const double lib_recall = overlap(theirs, truth, 10);
    EXPECT_GT(ours_recall, lib_recall - 0.05)
        << "ours " << ours_recall << " vs hnswlib " << lib_recall
        << " — more than five points behind the reference suggests a defect, "
           "not just a tuning difference";
}

TEST(HnswlibBaseline, HonoursFilters) {
    const Corpus& c = corpus();
    HnswlibIndex lib(kDim, Metric::L2, kN);
    lib.add_batch(c.base.data(), kN);

    std::vector<std::int64_t> allowed;
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(kN); i += 7) {
        allowed.push_back(i);
    }
    for (std::size_t i = 0; i < 10; ++i) {
        for (const Neighbor& n : lib.search_filtered(c.queries.data() + i * kDim,
                                                     10, allowed, 200)) {
            EXPECT_EQ(n.id % 7, 0);
        }
    }
    EXPECT_THROW(lib.search_filtered(c.queries.data(), 5, {999999}), std::out_of_range);
}

#endif  // HYLIS_HAS_HNSWLIB
