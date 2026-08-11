// index/rmi.hpp
//
// Recursive Model Index — a learned index over sorted keys.
// Kraska et al., "The Case for Learned Index Structures", SIGMOD 2018.
//
// The idea
// --------
// For sorted keys, the cumulative distribution function *is* the map from key
// to array position. A B+ tree navigates that map with pointers; an RMI
// approximates it with a function and then corrects itself with a small local
// search. Where a tree pays log(n) pointer-chasing cache misses, a model pays
// two multiply-adds and one short scan.
//
// Exactness
// ---------
// This is not an approximate structure. Every second-stage model records the
// worst under- and over-prediction it made over its own training keys, so a
// lookup searches only [pred - back, pred + fwd] and that interval provably
// contains the key if the key exists. A bad model costs time; it can never
// cost correctness. That distinction — exact index, approximate model — is
// the opposite of how HNSW works, and is the point most worth being able to
// defend.
//
// Two stages, and why
// -------------------
// One linear model over a curved CDF is hopeless. Stage 1 therefore only
// routes: it maps a key to one of M second-stage models, each of which fits
// its own slice. That is piecewise-linear approximation, and for a smooth
// curve the error of a linear fit across an interval of width h is bounded by
// (h^2/8)*max|f''| — so error falls roughly quadratically as M grows, while
// a model costs 24 bytes regardless of n.
//
// What that does *not* fix is a discontinuity. A CDF with a cliff in it (the
// `clustered` generator, or SOSD's `fb`) has unbounded second derivative, so
// whichever model straddles the cliff eats the full error no matter how large
// M gets. That is exactly the case where a B+ tree — which is
// distribution-free — should win, and the benchmark is set up to show it.
//
// Immutability
// ------------
// build() only: no insert, no erase. Every model and every error bound is
// derived from the whole key set at once, so a single insert can invalidate
// them. This is the real cost of the technique against a B+ tree and is not
// worth hiding. Module 7 (incremental retrain) is where it gets addressed.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/compare_op.hpp"

namespace hylis::index {

// y = slope*x + intercept, fitted by ordinary least squares.
class LinearModel {
public:
    LinearModel() = default;
    LinearModel(double slope, double intercept)
        : slope_(slope), intercept_(intercept) {}

    // Fit ys onto xs.
    //
    // Computed in mean-centred form rather than from raw sums of squares.
    // Keys here reach ~2^40 and n reaches 10^7, so sum(x^2) would be ~10^31 —
    // representable in a double but with most of its significant digits gone,
    // and the subtraction that follows would cancel what remained. Centring
    // first keeps every accumulated quantity near the scale of the data.
    static LinearModel fit(const double* xs, const double* ys, std::size_t n) {
        if (n == 0) return LinearModel(0.0, 0.0);
        if (n == 1) return LinearModel(0.0, ys[0]);

        double mean_x = 0.0, mean_y = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            mean_x += xs[i];
            mean_y += ys[i];
        }
        mean_x /= static_cast<double>(n);
        mean_y /= static_cast<double>(n);

        double sxy = 0.0, sxx = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = xs[i] - mean_x;
            sxy += dx * (ys[i] - mean_y);
            sxx += dx * dx;
        }

        // Every x identical: no line is determined, so predict the mean. The
        // error bounds measured afterwards will be wide, which is correct —
        // such a model genuinely carries no information.
        if (sxx <= 0.0) return LinearModel(0.0, mean_y);

        const double slope = sxy / sxx;
        return LinearModel(slope, mean_y - slope * mean_x);
    }

    double predict(double x) const { return slope_ * x + intercept_; }
    double slope() const { return slope_; }
    double intercept() const { return intercept_; }

private:
    double slope_ = 0.0;
    double intercept_ = 0.0;
};

template <typename Key = std::int64_t, typename Value = std::int64_t>
class RMIndex {
public:
    struct Stats {
        std::size_t size = 0;
        std::size_t models = 0;
        std::size_t empty_models = 0;
        std::size_t max_error = 0;    // worst |predicted - actual|, in records
        double mean_error = 0.0;
        std::size_t max_window = 0;   // widest search interval any key can need
        std::size_t model_bytes = 0;  // the index overhead proper
        std::size_t total_bytes = 0;  // + the key and value arrays
    };

    // `second_stage_size` is M, the number of models in the second stage —
    // the knob that buys accuracy on a curved distribution.
    //
    // `search_threshold` is where the final search switches from a linear
    // scan to a binary search. Small windows are a single cache line or two,
    // where a scan beats binary search's branch mispredictions; above the
    // threshold binary search takes over, which is what keeps a badly fitted
    // model at O(log n) instead of O(n).
    explicit RMIndex(std::size_t second_stage_size = 1024,
                     std::size_t search_threshold = 64)
        : model_count_(second_stage_size), threshold_(search_threshold) {
        if (model_count_ == 0) {
            throw std::invalid_argument("RMIndex: second_stage_size must be >= 1");
        }
    }

    std::size_t size() const { return keys_.size(); }
    bool empty() const { return keys_.empty(); }
    std::size_t model_count() const { return model_count_; }
    std::size_t search_threshold() const { return threshold_; }

    // Fit the index to a sorted, unique key set.
    void build(const std::vector<Key>& keys, const std::vector<Value>& values) {
        if (keys.size() != values.size()) {
            throw std::invalid_argument(
                "RMIndex::build: " + std::to_string(keys.size()) + " keys but " +
                std::to_string(values.size()) + " values");
        }
        for (std::size_t i = 1; i < keys.size(); ++i) {
            if (!(keys[i - 1] < keys[i])) {
                throw std::invalid_argument(
                    "RMIndex::build: keys must be strictly ascending; index " +
                    std::to_string(i - 1) + " is not less than index " +
                    std::to_string(i));
            }
        }
        if (keys.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "RMIndex::build: error bounds are stored as uint32, so at most "
                "4294967295 keys are supported");
        }

        keys_ = keys;
        values_ = values;
        stage2_.clear();
        stats_ = Stats{};

        if (keys_.empty()) {
            stats_.models = model_count_;
            stats_.empty_models = model_count_;
            account_memory();
            return;
        }

        origin_ = static_cast<double>(keys_.front());
        const std::size_t n = keys_.size();

        std::vector<double> xs(n), ys(n);
        for (std::size_t i = 0; i < n; ++i) {
            xs[i] = to_x(keys_[i]);
            ys[i] = static_cast<double>(i);
        }

        stage1_ = LinearModel::fit(xs.data(), ys.data(), n);
        fit_second_stage(xs, ys);
        measure_errors(xs);
        account_memory();
    }

    const Value* find(const Key& key) const {
        const std::size_t n = keys_.size();
        if (n == 0) return nullptr;

        std::size_t lo, hi;
        window_for(key, &lo, &hi);
        if (lo > hi) return nullptr;

        const std::size_t idx = search_window(key, lo, hi);
        return idx == n ? nullptr : &values_[idx];
    }

    bool contains(const Key& key) const { return find(key) != nullptr; }

    // Number of key comparisons a find() would perform. Exposed so the
    // graceful-degradation claim — that a useless model still costs O(log n)
    // and never O(n) — can be asserted rather than asserted-by-hand.
    std::size_t probes(const Key& key) const {
        const std::size_t n = keys_.size();
        if (n == 0) return 0;
        std::size_t lo, hi;
        window_for(key, &lo, &hi);
        if (lo > hi) return 0;
        const std::size_t width = hi - lo + 1;
        if (width <= threshold_) return width;
        std::size_t steps = 0;
        for (std::size_t w = width; w > 0; w >>= 1) ++steps;
        return steps + 1;
    }

    // Values for all keys in [lo, hi], ascending.
    std::vector<Value> range(const Key& lo, const Key& hi) const {
        std::vector<Value> out;
        if (hi < lo) return out;
        const std::size_t begin = lower_bound_pos(lo);
        const std::size_t end = upper_bound_pos(hi);
        out.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) out.push_back(values_[i]);
        return out;
    }

    // The predicate interface shared with the B+ tree, so the query planner
    // can hold either without knowing which. Ascending by key.
    std::vector<Value> range_query(CompareOp op, const Key& value) const {
        switch (op) {
            case CompareOp::Eq: {
                std::vector<Value> out;
                if (const Value* v = find(value)) out.push_back(*v);
                return out;
            }
            case CompareOp::Lt: return slice(0, lower_bound_pos(value));
            case CompareOp::Le: return slice(0, upper_bound_pos(value));
            case CompareOp::Gt: return slice(upper_bound_pos(value), keys_.size());
            case CompareOp::Ge: return slice(lower_bound_pos(value), keys_.size());
        }
        return {};
    }

    const std::vector<Key>& keys() const { return keys_; }

    std::vector<std::pair<Key, Value>> items() const {
        std::vector<std::pair<Key, Value>> out;
        out.reserve(keys_.size());
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            out.emplace_back(keys_[i], values_[i]);
        }
        return out;
    }

    void clear() {
        keys_.clear();
        values_.clear();
        stage2_.clear();
        stats_ = Stats{};
        account_memory();
    }

    Stats stats() const { return stats_; }

    // The exactness proof, for tests: replay every key through the same path
    // a lookup takes and confirm its true position lies inside the predicted
    // window. If this holds for all keys, no lookup can miss a key that is
    // present, whatever the models look like.
    void validate() const {
        if (keys_.size() != values_.size()) {
            throw std::logic_error("validate: keys/values length mismatch");
        }
        for (std::size_t i = 1; i < keys_.size(); ++i) {
            if (!(keys_[i - 1] < keys_[i])) {
                throw std::logic_error("validate: keys are not strictly ascending at " +
                                       std::to_string(i));
            }
        }
        if (stage2_.size() != (keys_.empty() ? 0u : model_count_)) {
            throw std::logic_error("validate: second stage has " +
                                   std::to_string(stage2_.size()) +
                                   " models, expected " + std::to_string(model_count_));
        }

        for (std::size_t i = 0; i < keys_.size(); ++i) {
            std::size_t lo, hi;
            window_for(keys_[i], &lo, &hi);
            if (i < lo || i > hi) {
                throw std::logic_error(
                    "validate: key at position " + std::to_string(i) +
                    " falls outside its predicted window [" + std::to_string(lo) +
                    ", " + std::to_string(hi) + "] — the error bound is wrong, "
                    "so lookups can miss keys that exist");
            }
            if (find(keys_[i]) == nullptr) {
                throw std::logic_error("validate: key at position " +
                                       std::to_string(i) + " is not findable");
            }
        }
    }

private:
    struct Model {
        LinearModel model;
        std::uint32_t back = 0;  // how far the prediction can overshoot
        std::uint32_t fwd = 0;   // how far it can undershoot
    };

    double to_x(const Key& key) const {
        // Subtracting the origin keeps x near zero, which matters because
        // int64 keys can be 19 digits wide. Done in double rather than on the
        // Key type because keys spanning the full int64 range would overflow
        // the subtraction. Keys above 2^53 lose resolution here, which costs
        // model accuracy but not correctness: the error bounds are measured
        // through this same transform.
        return static_cast<double>(key) - origin_;
    }

    // Which second-stage model owns a key. Must be identical at build time
    // and lookup time — a key that landed in model m during build has to
    // route to m again, or its recorded error bound would not apply to it.
    std::size_t model_index(double x) const {
        const double n = static_cast<double>(keys_.size());
        const double p = stage1_.predict(x);
        if (!std::isfinite(p)) return 0;
        const double scaled = p * (static_cast<double>(model_count_) / n);
        if (scaled <= 0.0) return 0;
        if (scaled >= static_cast<double>(model_count_)) return model_count_ - 1;
        return static_cast<std::size_t>(scaled);
    }

    void fit_second_stage(const std::vector<double>& xs, const std::vector<double>& ys) {
        const std::size_t n = keys_.size();
        stage2_.assign(model_count_, Model{});

        // With a positive stage-1 slope the model index is non-decreasing in
        // i, because the keys are sorted — so each model owns a contiguous
        // run and the buckets can be found in one pass with no extra memory.
        // A non-positive slope means the fit is degenerate (identical keys,
        // or a single distinct value); everything then falls into model 0,
        // which is still correct, just one wide model.
        // Initialising every boundary to n means a model never reached by the
        // sweep below owns the empty range [n, n) — which is exactly right
        // for models past the last key, and for models the sweep skipped over.
        std::vector<std::size_t> begin(model_count_ + 1, n);
        begin[0] = 0;
        if (stage1_.slope() > 0.0) {
            std::size_t current = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t m = model_index(xs[i]);
                // Skipped models collapse to [i, i): begin and end coincide.
                while (current < m) begin[++current] = i;
            }
        }
        begin[model_count_] = n;

        for (std::size_t m = 0; m < model_count_; ++m) {
            const std::size_t b = begin[m];
            const std::size_t e = begin[m + 1];
            if (b >= e) {
                // No keys routed here. It can still be reached by a lookup
                // for a key that is absent, so it must predict something in
                // range; a zero-width window at the boundary makes that
                // lookup correctly find nothing. No key that exists can land
                // here, because routing is a pure function of the key and
                // every existing key was routed during build.
                stage2_[m] = Model{LinearModel(0.0, static_cast<double>(b)), 0, 0};
                ++stats_.empty_models;
                continue;
            }
            stage2_[m].model = LinearModel::fit(xs.data() + b, ys.data() + b, e - b);
        }
    }

    void measure_errors(const std::vector<double>& xs) {
        const std::size_t n = keys_.size();
        double error_sum = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t m = model_index(xs[i]);
            Model& mod = stage2_[m];
            const std::size_t pred = clamp_position(mod.model.predict(xs[i]));
            if (pred > i) {
                mod.back = std::max(mod.back, static_cast<std::uint32_t>(pred - i));
                error_sum += static_cast<double>(pred - i);
                stats_.max_error = std::max(stats_.max_error, pred - i);
            } else {
                mod.fwd = std::max(mod.fwd, static_cast<std::uint32_t>(i - pred));
                error_sum += static_cast<double>(i - pred);
                stats_.max_error = std::max(stats_.max_error, i - pred);
            }
        }

        stats_.size = n;
        stats_.models = model_count_;
        stats_.mean_error = n ? error_sum / static_cast<double>(n) : 0.0;
        for (const Model& mod : stage2_) {
            stats_.max_window = std::max<std::size_t>(
                stats_.max_window, std::size_t{mod.back} + mod.fwd + 1);
        }
    }

    void account_memory() {
        stats_.model_bytes = stage2_.size() * sizeof(Model) + sizeof(LinearModel);
        stats_.total_bytes = stats_.model_bytes + keys_.size() * sizeof(Key) +
                             values_.size() * sizeof(Value);
    }

    std::size_t clamp_position(double p) const {
        const std::size_t n = keys_.size();
        if (n == 0) return 0;  // n - 1 below would wrap
        if (!std::isfinite(p) || p <= 0.0) return 0;
        if (p >= static_cast<double>(n - 1)) return n - 1;
        return static_cast<std::size_t>(p);
    }

    void window_for(const Key& key, std::size_t* lo, std::size_t* hi) const {
        const std::size_t n = keys_.size();
        const double x = to_x(key);
        const Model& mod = stage2_[model_index(x)];
        const double p = mod.model.predict(x);
        if (!std::isfinite(p)) {
            *lo = 0;
            *hi = n - 1;
            return;
        }
        const std::size_t pred = clamp_position(p);
        *lo = pred > mod.back ? pred - mod.back : 0;
        *hi = std::min(pred + mod.fwd, n - 1);
    }

    // Returns the index of `key` within [lo, hi], or keys_.size() if absent.
    std::size_t search_window(const Key& key, std::size_t lo, std::size_t hi) const {
        if (hi - lo + 1 <= threshold_) {
            for (std::size_t i = lo; i <= hi; ++i) {
                if (keys_[i] == key) return i;
                if (key < keys_[i]) break;  // sorted: it cannot be further right
            }
            return keys_.size();
        }
        const auto first = keys_.begin() + static_cast<std::ptrdiff_t>(lo);
        const auto last = keys_.begin() + static_cast<std::ptrdiff_t>(hi) + 1;
        const auto it = std::lower_bound(first, last, key);
        if (it != last && *it == key) {
            return static_cast<std::size_t>(it - keys_.begin());
        }
        return keys_.size();
    }

    // First index whose key is >= `key`.
    //
    // Galloping outward from the model's prediction rather than searching the
    // guaranteed window, because that window is only proven for keys present
    // at build time — and a range query's endpoint usually is not one. This
    // is correct for any key and any model, and still costs only
    // O(log(prediction error)) when the model is good.
    std::size_t lower_bound_pos(const Key& key) const {
        const std::size_t n = keys_.size();
        if (n == 0) return 0;

        const double x = to_x(key);
        const Model& mod = stage2_[model_index(x)];
        std::size_t pos = clamp_position(mod.model.predict(x));

        std::size_t lo, hi;
        if (keys_[pos] < key) {
            std::size_t step = 1;
            lo = pos;
            while (pos + step < n && keys_[pos + step] < key) {
                lo = pos + step;
                step *= 2;
            }
            hi = std::min(pos + step, n - 1) + 1;
            lo += 1;
        } else {
            std::size_t step = 1;
            hi = pos + 1;
            while (pos >= step && !(keys_[pos - step] < key)) {
                hi = pos - step + 1;
                step *= 2;
            }
            lo = pos >= step ? pos - step + 1 : 0;
        }

        const auto first = keys_.begin() + static_cast<std::ptrdiff_t>(lo);
        const auto last = keys_.begin() + static_cast<std::ptrdiff_t>(std::min(hi, n));
        return static_cast<std::size_t>(std::lower_bound(first, last, key) - keys_.begin());
    }

    std::size_t upper_bound_pos(const Key& key) const {
        const std::size_t lb = lower_bound_pos(key);
        // Keys are unique, so at most one entry equals `key`.
        return (lb < keys_.size() && keys_[lb] == key) ? lb + 1 : lb;
    }

    std::vector<Value> slice(std::size_t begin, std::size_t end) const {
        std::vector<Value> out;
        if (begin >= end) return out;
        out.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) out.push_back(values_[i]);
        return out;
    }

    std::size_t model_count_;
    std::size_t threshold_;
    double origin_ = 0.0;
    LinearModel stage1_;
    std::vector<Model> stage2_;
    std::vector<Key> keys_;
    std::vector<Value> values_;
    Stats stats_;
};

}  // namespace hylis::index
