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
// M gets.
//
// This comment used to go on to predict that such a distribution is where a
// B+ tree — being distribution-free — should win. **That prediction was
// measured and is wrong**, and it is left here corrected rather than deleted
// because the reasoning behind it is a natural mistake to make.
//
// scripts/experiment_discontinuity.py sweeps the number of cliffs from 64 to
// 250,000 over 500,000 keys. The RMI wins every row, by 3-7x, and its margin
// *grows* as cliffs are added. Two things the original reasoning missed:
//
//   * Only the models straddling a cliff are hurt, and there are at most as
//     many of those as there are cliffs. Lookup cost follows the typical
//     model, not the worst one.
//   * A model that is hurt does not degrade without bound. Its window is
//     either scanned, capped at `search_threshold`, or binary-searched above
//     that, capped at O(log n). Measured worst case on the clustered shape:
//     64 comparisons with a max_error of 3,299 — the scan, at the threshold.
//
// That second point is worth being precise about, because a cliff genuinely
// does cost more *comparisons*: 64 against 4 for a smooth CDF. What it does
// not cost is more *time*, because those 64 are a linear pass over one or two
// cache lines of a contiguous array, while a B+ tree's 4 are pointer-chased
// and miss cache every time. Counting comparisons flatters the tree. The
// model can only ever degrade into a bounded search over a flat array, and a
// tree descent is a slower version of exactly that — which is why no
// distribution hands the tree a win on lookups.
//
// The B+ tree's real claim is mutability, which is what module 4 measured and
// what dynamic_rmi.hpp exists to answer. Not any shape of data.
//
// Immutability, and the way out of it
// -----------------------------------
// build() only: no insert, no erase. Every model and every error bound is
// derived from the whole key set at once, so a single insert can invalidate
// them. This is the real cost of the technique against a B+ tree and is not
// worth hiding.
//
// dynamic_rmi.hpp wraps this class to get mutability, and drives two entry
// points here:
//
//   * enable_incremental() allocates the least-squares sufficient statistics
//     per second-stage model. Off by default, and held in a side vector, so
//     a read-only index pays neither the memory nor the bookkeeping — its
//     Stats::model_bytes is unchanged by this file's dynamic support.
//   * merge_from() installs a new key set while *reusing* the existing
//     models: routing is frozen, so every surviving key keeps the model it
//     had, and a model that saw no insert or delete needs only its intercept
//     shifted by the net change ahead of it. Its residuals — and therefore
//     its error bounds — are provably unchanged by such a shift.
//
// That is the DynaMind incremental-learning / machine-unlearning idea (Cheng
// et al., Knowledge-Based Systems 348, 2026) adapted to an index that must
// stay exact. Where the deviations matter they are commented at the point
// they occur.

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
        // Sufficient statistics and segment table, and zero unless
        // enable_incremental() was called. Included in total_bytes.
        std::size_t incremental_bytes = 0;
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
        // A rebuild refits stage 1, so the frozen routing every stored moment
        // and segment boundary was derived under no longer applies. Dropping
        // them is the only safe answer; merge_from() re-enables afterwards.
        state_.clear();
        bounds_.clear();

        if (keys_.empty()) {
            stats_.models = model_count_;
            stats_.empty_models = model_count_;
            account_memory();
            return;
        }

        origin_ = static_cast<double>(keys_.front());
        const std::size_t n = keys_.size();
        scale_ = static_cast<double>(model_count_) / static_cast<double>(n);

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
    const std::vector<Value>& values() const { return values_; }

    // Where `key` sits in the arrays, or size() if it is absent.
    //
    // find() returns the value; a caller that needs to mark a key rather than
    // read it — DynamicRMIndex tombstoning a deletion — needs the position,
    // and re-deriving it from the returned pointer would be a trick rather
    // than an interface.
    std::size_t position_of(const Key& key) const {
        if (keys_.empty()) return 0;
        std::size_t lo, hi;
        window_for(key, &lo, &hi);
        if (lo > hi) return keys_.size();
        return search_window(key, lo, hi);
    }

    // First position whose key is >= / > `key`. The endpoints of a range, in
    // the array coordinates the caller can then slice.
    std::size_t lower_bound(const Key& key) const { return lower_bound_pos(key); }
    std::size_t upper_bound(const Key& key) const { return upper_bound_pos(key); }

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
        state_.clear();
        bounds_.clear();
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

    // ---------------------------------------------------------------------
    // Incremental support, used by DynamicRMIndex. Inert unless switched on.
    // ---------------------------------------------------------------------

    struct MergeStats {
        std::size_t models_shifted = 0;   // O(1): intercept moved, bounds kept
        std::size_t models_refitted = 0;  // statistics and bounds recomputed
        std::size_t keys_rescanned = 0;   // keys visited re-measuring bounds
        std::size_t inserted = 0;
        std::size_t deleted = 0;
        bool full_rebuild = false;        // fell back to build()
    };

    bool incremental_enabled() const { return !state_.empty(); }

    // Which second-stage model owns this key. A pure function of the key,
    // because routing is frozen — so a caller can record it when a key is
    // touched and the answer is still the right one at merge time.
    std::size_t model_of_key(const Key& key) const {
        return keys_.empty() ? 0 : model_index(to_x(key));
    }

    // Allocate and populate the per-model sufficient statistics. One O(n)
    // pass, done once; after this every model can be refitted from its
    // moments without revisiting a key.
    void enable_incremental() {
        if (!state_.empty()) return;
        if (keys_.empty()) return;
        // Routing has to be monotone in the key for any of this to hold: it is
        // what guarantees each model owns a contiguous run, which is what lets
        // segment boundaries be maintained by arithmetic instead of a scan. A
        // non-positive stage-1 slope means the fit is degenerate — every key
        // identical after the conversion to double — and rather than carry a
        // second code path for a case that cannot arise from real data, the
        // index simply stays non-incremental and every merge is a full build.
        if (!(stage1_.slope() > 0.0)) return;
        state_.assign(model_count_, ModelState{});
        bounds_.assign(model_count_ + 1, 0);
        recompute_segments();
        for (std::size_t m = 0; m < model_count_; ++m) accumulate_segment(m);
        account_memory();
    }

    // How far the pending changes have pulled model `m` away from where its
    // own data now says it should be, as a Cook's distance (paper Eq. 13).
    //
    // The paper has to approximate the parameter movement from the batch's
    // residuals (Eqs. 14-15, both written "approximately equal") because it
    // does not carry the updated moments. We do carry them, so Δa and Δb here
    // are the exact difference between the model in force and the model its
    // statistics now imply — the approximation is unnecessary, not merely
    // improved on.
    double model_score(std::size_t m) const {
        if (m >= state_.size()) return 0.0;
        const ModelState& st = state_[m];
        if (st.count < 3) return 0.0;
        const LinearModel proposed = st.fit();
        const double da = proposed.slope() - st.sync_slope;
        const double db = proposed.intercept() - st.sync_intercept;
        if (da == 0.0 && db == 0.0) return 0.0;
        const double n = static_cast<double>(st.count);
        const double mean_x = st.sx / n;
        const double var = st.sxx - n * mean_x * mean_x;
        return (da * da * var + n * db * db) / (2.0 * st.mse());
    }

    // Largest model_score over the models the caller has disturbed.
    double max_score(const std::vector<std::size_t>& models) const {
        double worst = 0.0;
        for (std::size_t m : models) worst = std::max(worst, model_score(m));
        return worst;
    }

    // Withdraw one position's contribution from its model — machine
    // unlearning, in the paper's sense, applied to a single key.
    //
    // Exact and O(1) precisely because the caller tombstones rather than
    // compacts: no other key's position changes, so every remaining term in
    // the moments is still the term it was. Compaction would renumber
    // everything after `pos` and none of this would hold.
    //
    // The fitted model is deliberately NOT updated here. Its recorded error
    // bounds were measured against it, and moving it without re-measuring
    // them would make lookups able to miss keys that exist. Statistics drift
    // ahead of the model; apply_model_update() is what lets the model catch
    // up, and pays the re-measurement to stay exact.
    void unlearn_position(std::size_t pos) {
        if (state_.empty() || pos >= keys_.size()) return;
        const double x = to_x(keys_[pos]);
        ModelState& st = state_[model_index(x)];
        st.remove(x, static_cast<double>(pos));
    }

    // Adopt the model its statistics imply, and re-measure its error bounds
    // over its own segment so the exactness guarantee still holds.
    //
    // O(segment), not O(1). The paper's update is O(1) because an ALEX-style
    // exponential search is correct whatever the model predicts; ours has to
    // stand behind a proven window, so the bounds have to be re-derived. That
    // is the honest cost of being an exact index and it is charged here
    // rather than hidden.
    void apply_model_update(std::size_t m) {
        if (m >= state_.size() || state_[m].count == 0) return;
        stage2_[m].model = state_[m].fit();
        state_[m].sync();
        remeasure(m);
        refresh_aggregate_stats();
    }

    // Mean |predicted - actual| over every key, right now.
    //
    // Deliberately a method rather than a Stats field kept current: it is an
    // O(n) pass, and merge_from() exists precisely to avoid O(n) passes.
    // Stats::mean_error therefore reports the last full measurement, and this
    // is how a caller asks what the figure is today.
    double measure_mean_error() const {
        if (keys_.empty()) return 0.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            const double x = to_x(keys_[i]);
            const std::size_t pred = clamp_position(stage2_[model_index(x)].model.predict(x));
            sum += pred > i ? static_cast<double>(pred - i) : static_cast<double>(i - pred);
        }
        return sum / static_cast<double>(keys_.size());
    }

    // Install a new key set, reusing the models rather than refitting them.
    //
    // `inserted` and `deleted` are the keys added and removed relative to the
    // current contents, each sorted ascending. They decide which models were
    // disturbed: routing is frozen, so a model that received neither an insert
    // nor a delete has the same keys it always had, merely renumbered by a
    // constant — and a constant shift moves its intercept by exactly that
    // amount while leaving every residual, and therefore every error bound,
    // untouched. Those models cost O(1) each.
    MergeStats merge_from(const std::vector<Key>& keys,
                          const std::vector<Value>& values,
                          const std::vector<Key>& inserted,
                          const std::vector<Key>& deleted) {
        MergeStats out;
        out.inserted = inserted.size();
        out.deleted = deleted.size();

        // No frozen routing to reuse, or the index is (or becomes) empty:
        // there is nothing to be incremental about.
        if (!incremental_enabled() || keys_.empty() || keys.empty()) {
            build(keys, values);
            enable_incremental();
            out.full_rebuild = true;
            out.models_refitted = model_count_;
            out.keys_rescanned = keys.size();
            return out;
        }
        if (keys.size() != values.size()) {
            throw std::invalid_argument(
                "RMIndex::merge_from: " + std::to_string(keys.size()) +
                " keys but " + std::to_string(values.size()) + " values");
        }

        // Which models the change lands in, under the frozen routing. A key
        // being removed routes the same way it did when it was added, so the
        // deleted keys can be attributed without looking them up.
        std::vector<std::ptrdiff_t> net(model_count_, 0);
        std::vector<char> touched(model_count_, 0);
        for (const Key& k : inserted) {
            const std::size_t m = model_index(to_x(k));
            ++net[m];
            touched[m] = 1;
        }
        for (const Key& k : deleted) {
            const std::size_t m = model_index(to_x(k));
            --net[m];
            touched[m] = 1;
        }

        const std::vector<std::size_t> old_bounds = bounds_;
        keys_ = keys;
        values_ = values;

        // New segment boundaries, from the old ones plus the net change ahead
        // of each. O(models), no pass over the keys.
        std::ptrdiff_t running = 0;
        bounds_[0] = 0;
        for (std::size_t m = 0; m < model_count_; ++m) {
            const std::ptrdiff_t old_count =
                static_cast<std::ptrdiff_t>(old_bounds[m + 1] - old_bounds[m]);
            running += old_count + net[m];
            bounds_[m + 1] = static_cast<std::size_t>(running);
        }

        // If the arithmetic and the data disagree, the caller's `inserted` and
        // `deleted` did not describe the change it actually made. Refusing
        // here beats carrying on with segments that no longer bracket their
        // keys, which would corrupt every bound derived from them.
        if (bounds_[model_count_] != keys_.size()) {
            build(keys, values);
            state_.clear();
            enable_incremental();
            out.full_rebuild = true;
            out.models_refitted = model_count_;
            out.keys_rescanned = keys.size();
            return out;
        }

        stats_.empty_models = 0;
        for (std::size_t m = 0; m < model_count_; ++m) {
            const std::ptrdiff_t shift =
                static_cast<std::ptrdiff_t>(bounds_[m]) -
                static_cast<std::ptrdiff_t>(old_bounds[m]);
            if (bounds_[m] >= bounds_[m + 1]) ++stats_.empty_models;

            if (!touched[m]) {
                const double d = static_cast<double>(shift);
                state_[m].shift_positions(d);
                stage2_[m].model = LinearModel(stage2_[m].model.slope(),
                                               stage2_[m].model.intercept() + d);
                // back/fwd deliberately untouched: the residuals they were
                // measured from are unchanged by a uniform shift.
                ++out.models_shifted;
                continue;
            }

            // Disturbed. Its keys were renumbered by differing amounts — the
            // ones before an internal insertion moved less than the ones after
            // — so no single shift describes it and the moments are rebuilt
            // over the segment. That is O(segment), the same order as the
            // bound re-measurement this model needs regardless.
            accumulate_segment(m);
            stage2_[m].model = state_[m].count ? state_[m].fit()
                                               : LinearModel(0.0, static_cast<double>(bounds_[m]));
            remeasure(m);
            out.keys_rescanned += bounds_[m + 1] - bounds_[m];
            ++out.models_refitted;
        }

        refresh_aggregate_stats();
        return out;
    }

private:
    struct Model {
        LinearModel model;
        std::uint32_t back = 0;  // how far the prediction can overshoot
        std::uint32_t fwd = 0;   // how far it can undershoot
    };

    // Ordinary-least-squares sufficient statistics for one second-stage model,
    // enough to refit it — or to unfit a batch back out of it — without
    // revisiting the keys. DynaMind's Algorithm 2 in our coordinates.
    //
    // Accumulated over x = key - origin_, not over the raw key. The paper
    // writes S_kk = sum(k^2); our keys reach ~2^40 and n reaches 10^7, so that
    // is ~10^31, which a double holds only to a few significant figures, and
    // the subtraction n*S_kk - S_k^2 that follows then cancels away what was
    // left. That is the same reasoning behind LinearModel::fit being written
    // mean-centred. Shifting by a constant is absorbed entirely into the
    // intercept and leaves the slope untouched, so this is the paper's
    // algorithm with every accumulated quantity kept near the scale of the
    // data rather than 20 orders of magnitude above it.
    struct ModelState {
        double sx = 0.0;    // sum of x
        double sp = 0.0;    // sum of positions
        double sxx = 0.0;   // sum of x^2
        double sxp = 0.0;   // sum of x*position
        double sse = 0.0;   // residual sum of squares, for the score's MSE
        std::size_t count = 0;
        // The fit these moments implied when they were last synchronised with
        // the installed model. The score measures movement away from this,
        // not away from stage2_[m].model.
        //
        // The distinction is not pedantic. ModelState::fit and
        // LinearModel::fit compute the same line by different arithmetic, so
        // they differ in the last bits — and the score divides by the
        // segment's MSE, which for a short, well-fitted segment is very
        // close to zero. Scoring against the installed model would therefore
        // report a large score for an index nothing had yet been done to.
        double sync_slope = 0.0;
        double sync_intercept = 0.0;

        void sync() {
            const LinearModel m = fit();
            sync_slope = m.slope();
            sync_intercept = m.intercept();
        }

        // Fit y = a*x + b from the accumulated moments.
        //
        // Computed in the centred form rather than the textbook
        // (n*Sxp - Sx*Sp) / (n*Sxx - Sx^2): algebraically identical, but it
        // subtracts means before squaring instead of after, which is what
        // keeps the cancellation from eating the result.
        LinearModel fit() const {
            if (count == 0) return LinearModel(0.0, 0.0);
            const double n = static_cast<double>(count);
            const double mean_x = sx / n;
            const double mean_p = sp / n;
            const double cov = sxp - n * mean_x * mean_p;
            const double var = sxx - n * mean_x * mean_x;
            if (!(var > 0.0)) return LinearModel(0.0, mean_p);
            const double slope = cov / var;
            return LinearModel(slope, mean_p - slope * mean_x);
        }

        double mse() const {
            // n - 2 is the regression's degrees of freedom (paper, Eq. 13).
            // Below three points a line fits exactly and MSE is 0, which would
            // divide the score by zero; 1.0 makes the score report the raw
            // parameter movement instead, which is the right behaviour for a
            // segment too small for the statistic to mean anything.
            if (count < 3) return 1.0;
            const double dof = static_cast<double>(count) - 2.0;
            const double m = sse / dof;
            return m > 0.0 ? m : 1.0;
        }

        void add(double x, double p) {
            sx += x; sp += p; sxx += x * x; sxp += x * p;
            ++count;
        }

        void remove(double x, double p) {
            sx -= x; sp -= p; sxx -= x * x; sxp -= x * p;
            if (count) --count;
        }

        // Every position in this segment moved by the same delta.
        //
        // sum((p+d)) = sp + count*d and sum(x*(p+d)) = sxp + d*sx, so the
        // update is O(1). The slope is unchanged by construction and the
        // intercept moves by exactly d, which is why a shifted model's
        // residuals — and therefore its error bounds — survive untouched.
        void shift_positions(double d) {
            sxp += d * sx;
            sp += static_cast<double>(count) * d;
            // The reference line moves with the data, so a shift on its own
            // registers as no drift at all — which is the point: a uniform
            // renumbering is not a change in the distribution.
            sync_intercept += d;
        }
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
    //
    // The scale factor is a stored member rather than model_count_/keys_.size()
    // recomputed on the spot, and that is what makes merge_from() possible: it
    // freezes routing against changes in n. If the divisor moved with the key
    // count, inserting a single key would re-route every key in the index and
    // invalidate every model at once. Frozen, routing is a pure function of
    // the key, so a surviving key is guaranteed to keep its model across a
    // merge. build() is the only thing that ever sets it.
    std::size_t model_index(double x) const {
        const double p = stage1_.predict(x);
        if (!std::isfinite(p)) return 0;
        const double scaled = p * scale_;
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
        // Reported apart from model_bytes on purpose. A read-only index has
        // state_ empty and is charged nothing, so the static index's memory
        // figure means the same thing it did before this file gained dynamic
        // support — and a dynamic one is charged the difference visibly.
        stats_.incremental_bytes =
            state_.size() * sizeof(ModelState) + bounds_.size() * sizeof(std::size_t);
        stats_.total_bytes += stats_.incremental_bytes;
    }

    // Segment boundaries under the frozen routing: model m owns
    // [bounds_[m], bounds_[m + 1]).
    void recompute_segments() {
        const std::size_t n = keys_.size();
        bounds_.assign(model_count_ + 1, n);
        bounds_[0] = 0;
        std::size_t current = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t m = model_index(to_x(keys_[i]));
            while (current < m) bounds_[++current] = i;
        }
        while (current < model_count_) bounds_[++current] = n;
    }

    // Rebuild one model's moments from the keys it currently owns.
    void accumulate_segment(std::size_t m) {
        ModelState& st = state_[m];
        st = ModelState{};
        const std::size_t b = bounds_[m];
        const std::size_t e = bounds_[m + 1];
        for (std::size_t i = b; i < e; ++i) {
            st.add(to_x(keys_[i]), static_cast<double>(i));
        }
        st.sync();
        if (st.count < 3) return;

        // Residual sum of squares of the best fit to this segment, which is
        // the MSE the score divides by. Measured against st.fit() rather than
        // whatever model is currently installed, so it does not depend on the
        // order the caller does things in. It is not maintained as keys are
        // unlearned afterwards: it is a scale factor on the denominator, and
        // the paper likewise takes the MSE of the model as originally fitted.
        const LinearModel best = st.fit();
        double sse = 0.0;
        for (std::size_t i = b; i < e; ++i) {
            const double r = best.predict(to_x(keys_[i])) - static_cast<double>(i);
            sse += r * r;
        }
        st.sse = sse;
    }

    // Re-derive one model's error bounds from its own segment.
    //
    // Measured over every key physically present, including ones the caller
    // has tombstoned. Narrower bounds are available by skipping the dead, but
    // then validate() — which insists every stored key is findable — would
    // fail on an index that is in fact correct, and a weaker invariant is a
    // bad trade for a slightly tighter window.
    void remeasure(std::size_t m) {
        Model& mod = stage2_[m];
        mod.back = 0;
        mod.fwd = 0;
        for (std::size_t i = bounds_[m]; i < bounds_[m + 1]; ++i) {
            const double x = to_x(keys_[i]);
            const std::size_t pred = clamp_position(mod.model.predict(x));
            if (pred > i) {
                mod.back = std::max(mod.back, static_cast<std::uint32_t>(pred - i));
            } else {
                mod.fwd = std::max(mod.fwd, static_cast<std::uint32_t>(i - pred));
            }
        }
    }

    // Aggregates that can be refreshed without touching the keys.
    // Stats::mean_error is not among them — see measure_mean_error().
    void refresh_aggregate_stats() {
        stats_.size = keys_.size();
        stats_.models = model_count_;
        stats_.max_error = 0;
        stats_.max_window = 0;
        for (const Model& mod : stage2_) {
            stats_.max_error =
                std::max<std::size_t>(stats_.max_error, std::max(mod.back, mod.fwd));
            stats_.max_window = std::max<std::size_t>(
                stats_.max_window, std::size_t{mod.back} + mod.fwd + 1);
        }
        account_memory();
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
    // model_count_ / n, fixed at build. See model_index().
    double scale_ = 0.0;
    LinearModel stage1_;
    std::vector<Model> stage2_;
    std::vector<Key> keys_;
    std::vector<Value> values_;
    Stats stats_;

    // Incremental-merge support. Empty unless enable_incremental() was called,
    // which is why Model itself carries none of this: a read-only index would
    // otherwise pay 40 bytes per model for machinery it never uses, and
    // Stats::model_bytes — a number this project reports — would quietly grow.
    std::vector<ModelState> state_;
    // Segment boundaries, size model_count_ + 1, so segment m is
    // [bounds_[m], bounds_[m + 1]). Also only allocated when incremental.
    std::vector<std::size_t> bounds_;
};

}  // namespace hylis::index
