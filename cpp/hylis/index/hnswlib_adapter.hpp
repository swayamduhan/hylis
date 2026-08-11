// index/hnswlib_adapter.hpp
//
// nmslib/hnswlib behind hylis' own vector-index interface.
//
// Benchmark baseline only. Nothing in the engine depends on this file, and the
// whole thing compiles out when HYLIS_HAS_HNSWLIB is off — which is the
// constraint the README states, and the reason this lives behind an adapter
// rather than being used directly anywhere.
//
// Its job is to answer one question that cannot be answered from inside the
// project: is our HNSW competitive, or a strawman? A hand-written
// implementation that the neural router beats is worth nothing if the
// implementation itself is slow. Comparing against the reference is the only
// way to know, and reporting the gap honestly is more valuable than hiding it.
//
// The adapter presents search/search_filtered with our signatures and our
// Neighbor type, so the three implementations are interchangeable in tests and
// benchmarks. Scores are converted to match our convention exactly: hnswlib
// returns squared L2 where we return true distance, and negated inner product
// where we return the dot product itself. Getting that wrong would make a
// recall comparison meaningless.

#pragma once

#ifdef HYLIS_HAS_HNSWLIB

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "index/distance.hpp"

namespace hylis::index {

class HnswlibIndex {
public:
    HnswlibIndex(std::size_t dim, Metric metric = Metric::L2,
                 std::size_t capacity = 1000, std::size_t M = 16,
                 std::size_t ef_construction = 200, std::uint64_t seed = 100)
        : dim_(dim), metric_(metric), capacity_(capacity) {
        if (dim_ == 0) throw std::invalid_argument("HnswlibIndex: dim must be > 0");
        if (M < 2) throw std::invalid_argument("HnswlibIndex: M must be >= 2");

        if (metric_ == Metric::L2) {
            space_ = std::make_unique<hnswlib::L2Space>(dim_);
        } else {
            // hnswlib has no separate cosine space: cosine is inner product on
            // normalised vectors, which is exactly how our FlatIndex and
            // HnswIndex implement it too, so vectors are normalised on the way
            // in and the same space serves both.
            space_ip_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
        }

        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            base_space(), std::max<std::size_t>(capacity_, 1), M, ef_construction,
            static_cast<std::size_t>(seed));
    }

    std::size_t dim() const { return dim_; }
    std::size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    Metric metric() const { return metric_; }

    std::int64_t add(const float* vec) {
        if (count_ >= capacity_) {
            capacity_ = std::max<std::size_t>(capacity_ * 2, count_ + 1);
            index_->resizeIndex(capacity_);
        }
        if (metric_ == Metric::Cosine) {
            scratch_.assign(vec, vec + dim_);
            normalise(scratch_.data(), dim_);
            index_->addPoint(scratch_.data(), count_);
        } else {
            index_->addPoint(vec, count_);
        }
        return static_cast<std::int64_t>(count_++);
    }

    void add_batch(const float* data, std::size_t n) {
        if (count_ + n > capacity_) {
            capacity_ = count_ + n;
            index_->resizeIndex(capacity_);
        }
        for (std::size_t i = 0; i < n; ++i) add(data + i * dim_);
    }

    std::vector<Neighbor> search(const float* query, std::size_t k,
                                 std::size_t ef = 0) const {
        std::vector<Neighbor> out;
        if (count_ == 0 || k == 0) return out;
        index_->setEf(std::max(ef ? ef : std::size_t{50}, k));

        const float* q = prepare(query);
        auto heap = index_->searchKnn(q, std::min(k, count_));

        out.reserve(heap.size());
        while (!heap.empty()) {
            const auto& [distance, label] = heap.top();
            out.push_back({static_cast<std::int64_t>(label),
                           present(distance)});
            heap.pop();
        }
        // searchKnn yields farthest-first; ours is best-first everywhere.
        std::reverse(out.begin(), out.end());
        return out;
    }

    std::vector<Neighbor> search_filtered(const float* query, std::size_t k,
                                          const std::vector<std::int64_t>& allowed,
                                          std::size_t ef = 0) const {
        std::vector<Neighbor> out;
        if (count_ == 0 || k == 0 || allowed.empty()) return out;
        for (std::int64_t id : allowed) {
            if (id < 0 || static_cast<std::size_t>(id) >= count_) {
                throw std::out_of_range("HnswlibIndex: id " + std::to_string(id) +
                                        " out of range");
            }
        }

        index_->setEf(std::max(ef ? ef : std::size_t{50}, k));
        AllowList filter(allowed, count_);
        const float* q = prepare(query);
        auto heap = index_->searchKnn(q, std::min(k, allowed.size()), &filter);

        out.reserve(heap.size());
        while (!heap.empty()) {
            const auto& [distance, label] = heap.top();
            out.push_back({static_cast<std::int64_t>(label), present(distance)});
            heap.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    std::size_t memory_bytes() const {
        // hnswlib packs vectors and links into one block per element.
        return count_ * index_->size_data_per_element_;
    }

private:
    struct AllowList : hnswlib::BaseFilterFunctor {
        explicit AllowList(const std::vector<std::int64_t>& ids, std::size_t n)
            : mask(n, false) {
            for (std::int64_t id : ids) mask[static_cast<std::size_t>(id)] = true;
        }
        bool operator()(hnswlib::labeltype label) override {
            return label < mask.size() && mask[label];
        }
        std::vector<bool> mask;
    };

    hnswlib::SpaceInterface<float>* base_space() const {
        return space_ ? static_cast<hnswlib::SpaceInterface<float>*>(space_.get())
                      : static_cast<hnswlib::SpaceInterface<float>*>(space_ip_.get());
    }

    const float* prepare(const float* query) const {
        if (metric_ != Metric::Cosine) return query;
        query_scratch_.assign(query, query + dim_);
        normalise(query_scratch_.data(), dim_);
        return query_scratch_.data();
    }

    // hnswlib's L2Space returns *squared* distance and its InnerProductSpace
    // returns 1 - dot. Ours return true distance and the dot itself, so the
    // conversion happens here — otherwise a score comparison between the two
    // would silently be comparing different quantities.
    float present(float raw) const {
        if (metric_ == Metric::L2) return std::sqrt(std::max(raw, 0.0f));
        return 1.0f - raw;
    }

    std::size_t dim_;
    Metric metric_;
    std::size_t capacity_;
    std::size_t count_ = 0;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::InnerProductSpace> space_ip_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::vector<float> scratch_;
    mutable std::vector<float> query_scratch_;
};

}  // namespace hylis::index

#endif  // HYLIS_HAS_HNSWLIB
