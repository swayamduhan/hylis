// index/flat.hpp
//
// Flat (brute-force) vector index: exhaustive k-nearest-neighbour search.
//
// Why an exhaustive index is worth building
// -----------------------------------------
// It has three jobs, and being fast is not one of them.
//
//   1. It is the *exact* answer. HNSW is approximate by construction, so
//      "is it correct?" is not a question that can be asked of it — only
//      "what recall does it reach, and how quickly?". Recall is undefined
//      without a true answer to compare against, and this produces it.
//   2. It is the baseline every speedup is quoted against.
//   3. It is a genuine execution strategy, not a strawman. Over a few
//      thousand candidates it beats HNSW outright, because graph traversal
//      has overhead that only amortises at scale. When a selective predicate
//      leaves a small candidate set, scanning it is the *right* plan, which
//      is why search_filtered exists and why the planner's choice is
//      non-trivial.
//
// Layout
// ------
// All vectors live in one contiguous float buffer, row-major, dim floats per
// row — not a vector-of-vectors. A full sequential sweep of the corpus is the
// entire operation, so the hardware prefetcher walking one flat array is the
// only real optimisation available here; scattering rows across the heap
// would give that up for nothing.
//
// Ids are implicit row indices, 0..n-1, assigned in insertion order. Mapping
// them onto record primary keys is the caller's job — the same separation the
// B+ tree keeps between its keys and the records they point at.
//
// Metrics and score direction
// ---------------------------
// Internally every metric is reduced to a *score where smaller is better*, so
// there is one comparison, one heap and one code path. Inner product is
// negated to fit that convention and negated back on the way out.
//
// L2 accumulates squared distances and takes the square root only of the k
// survivors: sqrt is monotonic, so it cannot change the ranking, and skipping
// it saves n-k square roots per query.
//
// Ties
// ----
// Equal scores are broken by lower id. This is not cosmetic. Real corpora
// contain genuine ties — SIFT descriptors are quantised, so distinct vectors
// collide at identical distances — and without a total order the results
// would be unstable between runs and could not be compared byte-for-byte
// against an independent oracle.

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

// One search result. `score` means whichever quantity the index's metric
// measures: a Euclidean distance for L2, a dot product for InnerProduct, a
// cosine similarity in [-1, 1] for Cosine. Results are always ordered best
// first regardless of which of those it is.
struct Neighbor {
    std::int64_t id;
    float score;
};

class FlatIndex {
public:
    explicit FlatIndex(std::size_t dim, Metric metric = Metric::L2)
        : dim_(dim), metric_(metric) {
        if (dim_ == 0) throw std::invalid_argument("FlatIndex: dim must be > 0");
    }

    std::size_t dim() const { return dim_; }
    std::size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    Metric metric() const { return metric_; }

    void reserve(std::size_t n) { data_.reserve(n * dim_); }

    void clear() {
        data_.clear();
        count_ = 0;
    }

    // Append one vector. Returns the id assigned to it.
    std::int64_t add(const float* vec) {
        data_.insert(data_.end(), vec, vec + dim_);
        if (metric_ == Metric::Cosine) normalise_row(count_);
        return static_cast<std::int64_t>(count_++);
    }

    std::int64_t add(const std::vector<float>& vec) {
        require_dim(vec.size());
        return add(vec.data());
    }

    // Append n vectors laid out row-major. Cheaper than n calls to add()
    // because the buffer grows once.
    void add_batch(const float* data, std::size_t n) {
        data_.reserve(data_.size() + n * dim_);
        data_.insert(data_.end(), data, data + n * dim_);
        if (metric_ == Metric::Cosine) {
            for (std::size_t i = count_; i < count_ + n; ++i) normalise_row(i);
        }
        count_ += n;
    }

    // Read back a stored vector. Note that for Cosine this is the normalised
    // form, not the vector originally handed to add().
    const float* vector_at(std::int64_t id) const {
        require_id(id);
        return data_.data() + static_cast<std::size_t>(id) * dim_;
    }

    // Exhaustive search over every stored vector. Returns min(k, size())
    // neighbours, best first.
    std::vector<Neighbor> search(const float* query, std::size_t k) const {
        TopK top(k);
        if (k == 0 || count_ == 0) return top.drain();

        std::vector<float> scratch;
        const float* q = prepare_query(query, scratch);
        for (std::size_t i = 0; i < count_; ++i) {
            top.offer(static_cast<std::int64_t>(i), score_row(q, i));
        }
        return finish(top);
    }

    std::vector<Neighbor> search(const std::vector<float>& query,
                                 std::size_t k) const {
        require_dim(query.size());
        return search(query.data(), k);
    }

    // Search restricted to `allowed` ids — the pre-filter execution plan.
    //
    // The planner reaches for this when a structured predicate is selective
    // enough that scanning the survivors costs less than searching the whole
    // corpus and discarding most of what comes back. Cost here is O(|allowed|)
    // rather than O(n), which is precisely the property that makes the plan
    // worth choosing.
    std::vector<Neighbor> search_filtered(const float* query, std::size_t k,
                                          const std::vector<std::int64_t>& allowed) const {
        TopK top(k);
        if (k == 0 || allowed.empty()) return top.drain();

        std::vector<float> scratch;
        const float* q = prepare_query(query, scratch);
        for (std::int64_t id : allowed) {
            require_id(id);
            top.offer(id, score_row(q, static_cast<std::size_t>(id)));
        }
        return finish(top);
    }

    std::vector<Neighbor> search_filtered(const std::vector<float>& query, std::size_t k,
                                          const std::vector<std::int64_t>& allowed) const {
        require_dim(query.size());
        return search_filtered(query.data(), k, allowed);
    }

    // Search n_queries queries laid out row-major.
    std::vector<std::vector<Neighbor>> search_batch(const float* queries,
                                                    std::size_t n_queries,
                                                    std::size_t k) const {
        std::vector<std::vector<Neighbor>> out;
        out.reserve(n_queries);
        for (std::size_t i = 0; i < n_queries; ++i) {
            out.push_back(search(queries + i * dim_, k));
        }
        return out;
    }

private:
    // Bounded max-heap of the best k seen so far, worst at the top.
    //
    // The alternative — score all n, then sort — costs O(n) memory and
    // O(n log n). This costs O(k), and once the heap is full the common case
    // is a single comparison that rejects the candidate outright, because
    // most of a corpus is nowhere near any given query.
    class TopK {
    public:
        explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

        void offer(std::int64_t id, float score) {
            if (heap_.size() < k_) {
                heap_.push_back({id, score});
                std::push_heap(heap_.begin(), heap_.end(), worse_last);
                return;
            }
            // heap_.front() is the worst kept so far; a candidate that is not
            // strictly better than it cannot enter.
            if (!better(score, id, heap_.front().score, heap_.front().id)) return;
            std::pop_heap(heap_.begin(), heap_.end(), worse_last);
            heap_.back() = {id, score};
            std::push_heap(heap_.begin(), heap_.end(), worse_last);
        }

        // Empties the heap into a best-first vector.
        std::vector<Neighbor> drain() {
            std::sort_heap(heap_.begin(), heap_.end(), worse_last);
            return std::move(heap_);
        }

    private:
        // Total order over candidates: lower score wins, ties go to lower id.
        static bool better(float sa, std::int64_t ia, float sb, std::int64_t ib) {
            if (sa != sb) return sa < sb;
            return ia < ib;
        }

        // Comparator for the heap: "a is better than b", which puts the worst
        // element at the top, and — via sort_heap — leaves the range sorted
        // best-first.
        static bool worse_last(const Neighbor& a, const Neighbor& b) {
            return better(a.score, a.id, b.score, b.id);
        }

        std::size_t k_;
        std::vector<Neighbor> heap_;
    };

    // Internal scores are always "smaller is better". Undo that for the
    // metrics whose natural reading is the other way round, and finish the
    // square root L2 deferred.
    std::vector<Neighbor> finish(TopK& top) const {
        std::vector<Neighbor> out = top.drain();
        for (Neighbor& n : out) {
            if (metric_ == Metric::L2) {
                n.score = std::sqrt(std::max(n.score, 0.0f));
            } else {
                n.score = -n.score;
            }
        }
        return out;
    }

    const float* prepare_query(const float* query, std::vector<float>& scratch) const {
        if (metric_ != Metric::Cosine) return query;
        scratch.assign(query, query + dim_);
        normalise(scratch.data());
        return scratch.data();
    }

    float score_row(const float* q, std::size_t row) const {
        const float* b = data_.data() + row * dim_;
        if (metric_ == Metric::L2) {
            // Computed directly as sum((q-b)^2) rather than expanded into
            // |q|^2 - 2q.b + |b|^2. The expansion is faster because it can be
            // written as a matrix product, but it subtracts two large similar
            // numbers and loses precision exactly where it matters — among
            // near neighbours, whose distances are the small ones. The
            // oracle in python/hylis/datasets.py uses the expanded form, so
            // the two agreeing is a meaningful cross-check.
            float acc = 0.0f;
            for (std::size_t i = 0; i < dim_; ++i) {
                const float d = q[i] - b[i];
                acc += d * d;
            }
            return acc;
        }
        float dot = 0.0f;
        for (std::size_t i = 0; i < dim_; ++i) dot += q[i] * b[i];
        return -dot;  // negated so that smaller stays better
    }

    void normalise_row(std::size_t row) { normalise(data_.data() + row * dim_); }

    void normalise(float* v) const {
        float sq = 0.0f;
        for (std::size_t i = 0; i < dim_; ++i) sq += v[i] * v[i];
        // A zero vector has no direction. Leaving it at zero makes it equally
        // (dis)similar to everything, which is the least surprising answer;
        // dividing would produce NaN and poison every comparison it touches.
        if (sq <= 0.0f) return;
        // Divide rather than multiply by a precomputed reciprocal. The
        // reciprocal is the usual trick and saves dim-1 divisions, but it
        // rounds twice where dividing rounds once, and the error lands
        // exactly where it does the most damage: unit-length vectors come
        // out as 0.99999994 instead of 1.0, so vectors that should tie at
        // similarity 1 no longer do, and the tie-break stops being
        // deterministic. Normalisation happens once per vector, so the
        // divisions cost nothing measurable.
        const float norm = std::sqrt(sq);
        for (std::size_t i = 0; i < dim_; ++i) v[i] /= norm;
    }

    void require_dim(std::size_t got) const {
        if (got != dim_) {
            throw std::invalid_argument(
                "FlatIndex: expected a " + std::to_string(dim_) +
                "-dimensional vector, got " + std::to_string(got));
        }
    }

    void require_id(std::int64_t id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= count_) {
            throw std::out_of_range(
                "FlatIndex: id " + std::to_string(id) + " out of range [0, " +
                std::to_string(count_) + ")");
        }
    }

    std::size_t dim_;
    Metric metric_;
    std::vector<float> data_;
    std::size_t count_ = 0;
};

}  // namespace hylis::index
