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

// Metric, Neighbor, the scoring loop and the bounded top-k heap are shared
// with the HNSW index — see the header for why one implementation matters
// when the two are benchmarked against each other.
#include "index/distance.hpp"

namespace hylis::index {

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
    std::vector<Neighbor> finish(TopK& top) const {
        return top.drain_presented(metric_);
    }

    const float* prepare_query(const float* query, std::vector<float>& scratch) const {
        if (metric_ != Metric::Cosine) return query;
        scratch.assign(query, query + dim_);
        normalise(scratch.data(), dim_);
        return scratch.data();
    }

    float score_row(const float* q, std::size_t row) const {
        return score_vectors(q, data_.data() + row * dim_, dim_, metric_);
    }

    void normalise_row(std::size_t row) {
        normalise(data_.data() + row * dim_, dim_);
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
