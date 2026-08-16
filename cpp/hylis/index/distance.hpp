// index/distance.hpp
//
// The vocabulary every vector index in hylis shares: metrics, results,
// scoring, and bounded top-k selection.
//
// Split out of flat.hpp once HNSW needed all of it. Keeping one
// implementation matters more here than for most shared code, because the
// exact index and the approximate one are compared directly against each
// other: if they scored distances even slightly differently, a recall figure
// would be measuring the discrepancy rather than the algorithm.
//
// Score direction
// ---------------
// Internally every metric is reduced to a score where *smaller is better*, so
// there is one comparison, one heap and one code path. Inner product is
// negated to fit that convention and negated back on the way out. L2
// accumulates squared distances, with the square root taken only on the
// surviving k — monotonic, so it cannot change a ranking.
//
// Ties
// ----
// Equal scores break toward the lower id. Not cosmetic: real corpora contain
// genuine ties (SIFT descriptors are quantised, so distinct vectors collide
// at identical distances), and without a total order results would be
// unstable between runs and impossible to diff against an oracle.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hylis::index {

enum class Metric {
    L2,            // Euclidean distance; smaller is nearer
    InnerProduct,  // dot product; larger is more similar
    Cosine,        // inner product on normalised vectors; larger is similar
};

// The metric's name, and back again.
//
// Needed because a vector column's metric is persisted: a sidecar of raw
// floats is uninterpretable without it, since Cosine stores the *normalised*
// form and reloading those vectors under L2 would answer a different question
// while looking entirely plausible.
inline const char* to_string(Metric metric) {
    switch (metric) {
        case Metric::L2: return "l2";
        case Metric::InnerProduct: return "inner_product";
        case Metric::Cosine: return "cosine";
    }
    return "l2";
}

inline bool try_metric_from_string(const std::string& name, Metric* out) {
    if (name == "l2") { *out = Metric::L2; return true; }
    if (name == "inner_product") { *out = Metric::InnerProduct; return true; }
    if (name == "cosine") { *out = Metric::Cosine; return true; }
    return false;
}

inline Metric metric_from_string(const std::string& name) {
    Metric out = Metric::L2;
    if (!try_metric_from_string(name, &out)) {
        throw std::invalid_argument("unknown metric '" + name +
                                    "'; one of l2, inner_product, cosine");
    }
    return out;
}

// One search result. `score` means whichever quantity the index's metric
// measures: a Euclidean distance for L2, a dot product for InnerProduct, a
// cosine similarity in [-1, 1] for Cosine. Results are always ordered best
// first regardless of which of those it is.
struct Neighbor {
    std::int64_t id;
    float score;
};

// Total order over candidates: lower score wins, ties go to lower id.
inline bool better_than(float score_a, std::int64_t id_a,
                        float score_b, std::int64_t id_b) {
    if (score_a != score_b) return score_a < score_b;
    return id_a < id_b;
}

// The internal "smaller is better" score between two vectors.
inline float score_vectors(const float* a, const float* b, std::size_t dim,
                           Metric metric) {
    if (metric == Metric::L2) {
        // Computed directly as sum((a-b)^2) rather than expanded into
        // |a|^2 - 2a.b + |b|^2. The expansion is faster because it can be
        // written as a matrix product, but it subtracts two large similar
        // numbers and loses precision exactly where it matters — among near
        // neighbours, whose distances are the small ones. The oracle in
        // python/hylis/datasets.py uses the expanded form, so the two
        // agreeing is a meaningful cross-check.
        float acc = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            const float d = a[i] - b[i];
            acc += d * d;
        }
        return acc;
    }
    float dot = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) dot += a[i] * b[i];
    return -dot;  // negated so that smaller stays better
}

// Turn an internal score into the number a caller expects for this metric.
inline float present_score(float internal, Metric metric) {
    if (metric == Metric::L2) return std::sqrt(std::max(internal, 0.0f));
    return -internal;
}

inline void normalise(float* v, std::size_t dim) {
    float sq = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) sq += v[i] * v[i];
    // A zero vector has no direction. Leaving it at zero makes it equally
    // (dis)similar to everything, which is the least surprising answer;
    // dividing would produce NaN and poison every comparison it touches.
    if (sq <= 0.0f) return;
    // Divide rather than multiply by a precomputed reciprocal. The reciprocal
    // is the usual trick and saves dim-1 divisions, but it rounds twice where
    // dividing rounds once, and the error lands exactly where it does the most
    // damage: unit-length vectors come out as 0.99999994 instead of 1.0, so
    // vectors that should tie at similarity 1 no longer do, and the tie-break
    // stops being deterministic. Normalisation happens once per vector, so the
    // divisions cost nothing measurable.
    const float norm = std::sqrt(sq);
    for (std::size_t i = 0; i < dim; ++i) v[i] /= norm;
}

// Bounded max-heap of the best k seen so far, worst at the top.
//
// The alternative — score everything, then sort — costs O(n) memory and
// O(n log n). This costs O(k), and once the heap is full the common case is a
// single comparison that rejects the candidate outright, because most of a
// corpus is nowhere near any given query.
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

    std::size_t size() const { return heap_.size(); }
    bool full() const { return heap_.size() >= k_; }
    std::size_t capacity() const { return k_; }

    // The worst entry currently kept. Only meaningful when non-empty; HNSW
    // uses it as the beam-search cutoff.
    const Neighbor& worst() const { return heap_.front(); }

    void offer(std::int64_t id, float score) {
        if (heap_.size() < k_) {
            heap_.push_back({id, score});
            std::push_heap(heap_.begin(), heap_.end(), worse_last);
            return;
        }
        if (k_ == 0) return;
        // heap_.front() is the worst kept so far; a candidate that is not
        // strictly better than it cannot enter.
        if (!better_than(score, id, heap_.front().score, heap_.front().id)) return;
        std::pop_heap(heap_.begin(), heap_.end(), worse_last);
        heap_.back() = {id, score};
        std::push_heap(heap_.begin(), heap_.end(), worse_last);
    }

    // Empties the heap into a best-first vector, still holding internal
    // scores. Callers convert with present_score().
    std::vector<Neighbor> drain() {
        std::sort_heap(heap_.begin(), heap_.end(), worse_last);
        return std::move(heap_);
    }

    // Best-first, with scores converted for the given metric.
    std::vector<Neighbor> drain_presented(Metric metric) {
        std::vector<Neighbor> out = drain();
        for (Neighbor& n : out) n.score = present_score(n.score, metric);
        return out;
    }

private:
    // Comparator for the heap: "a is better than b", which puts the worst
    // element at the top, and — via sort_heap — leaves the range sorted
    // best-first.
    static bool worse_last(const Neighbor& a, const Neighbor& b) {
        return better_than(a.score, a.id, b.score, b.id);
    }

    std::size_t k_;
    std::vector<Neighbor> heap_;
};

}  // namespace hylis::index
