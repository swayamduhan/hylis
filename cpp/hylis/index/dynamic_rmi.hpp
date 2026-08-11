// index/dynamic_rmi.hpp
//
// A learned index you can write to.
//
// The problem
// -----------
// RMIndex is build-only, and for a good reason: every model and every error
// bound is derived from the whole key set at once, so one insert can
// invalidate all of them. That is the real cost of the technique against a
// B+ tree. This file pays that cost down without giving up exactness.
//
// The shape of the answer
// -----------------------
// Out-of-place updates, in the sense of PGM-index and XIndex rather than
// ALEX's in-place gapped arrays:
//
//   base_    an immutable RMIndex — fast, exact, untouched between merges
//   delta_   a B+ tree holding everything inserted since the last merge
//   dead_    one bit per base position, marking logical deletions
//
// The delta buffer is a B+ tree we already had. It is sorted, mutable, and
// answers the same predicates, so the merge below is a linear walk of two
// ordered streams rather than a sort. DynaMind uses gapped arrays; we get
// out-of-place behaviour for free from module 2 instead of writing a second
// mutable structure to get in-place behaviour.
//
// Why deletions are marked and not removed
// ----------------------------------------
// Compacting on delete would renumber every key after the hole, and every
// second-stage model's error bound is a statement about positions. One
// deletion would invalidate the lot. Tombstoning leaves positions fixed, so:
//
//   * every surviving key's bound is still exactly the bound that was
//     measured for it, and
//   * the withdrawal of the deleted key from its model's least-squares
//     moments is exact and O(1), because no other key's (x, position) pair
//     moved.
//
// That second point is machine unlearning in the DynaMind sense (Cheng et
// al., Knowledge-Based Systems 348, 2026, §4.1), and it is exact here rather
// than approximate precisely because we refused to compact.
//
// What the score function is used for, and how that differs from the paper
// -----------------------------------------------------------------------
// DynaMind's insight is that triggering a model update when a buffer fills is
// the wrong trigger, because the distribution can shift badly while the
// buffer is still half empty. It measures the damage directly instead, via a
// Cook's-distance score, and updates when the score crosses a threshold.
//
// We adopt the trigger and point it at a different decision. In DynaMind
// inserts land in the leaf's own array, so the model is wrong the moment one
// arrives and the score decides when to fix it. Here inserts land in the
// delta buffer, so base_'s models stay *exactly* right for base_'s contents
// no matter how many arrive — nothing to fix. What the score measures for us
// is how far the deletions have pulled each model from its own data, and,
// through max_score(), whether it is time to merge. It is a merge trigger,
// which is the same idea applied at the level our architecture puts the
// decision.
//
// The honest cost
// ---------------
// Every read now checks the delta buffer and a tombstone bit. On a read-only
// workload this is slower than a plain RMIndex, and the benchmark reports it
// rather than burying it. That is the same trade a B+ tree makes to be
// mutable; the difference is that here it is opt-in per column.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/btree.hpp"
#include "index/compare_op.hpp"
#include "index/rmi.hpp"

namespace hylis::index {

template <typename Key = std::int64_t, typename Value = std::int64_t>
class DynamicRMIndex {
public:
    struct Config {
        // Passed straight to the underlying RMIndex.
        std::size_t second_stage_size = 1024;
        std::size_t search_threshold = 64;
        // Order of the delta buffer's B+ tree.
        std::size_t delta_order = 32;

        // Merge once the pending changes reach this fraction of the base.
        // The size trigger, and the one DynaMind argues is insufficient on
        // its own — kept as the backstop so a workload that never moves the
        // models still eventually reclaims the tombstones.
        double merge_ratio = 0.05;

        // Merge once any disturbed model's Cook's distance passes this.
        //
        // Infinity — the score trigger off — is the measured default, not a
        // placeholder. Two findings put it there, both reproducible from
        // scripts/experiment_merge_threshold.py:
        //
        // 1. The paper's tuned value of 1.0 does not transfer. Cook's
        //    distance is not scale-free in the segment length, and DynaMind
        //    evaluates it over an ALEX leaf while this evaluates it over an
        //    RMI second-stage segment. Measured on 100k keys, 32 deletions
        //    in, the median score runs 0.58 at 1562 keys per model, 53 at
        //    391, 3.4e3 at 98, and 6.5e6 at 6. A single threshold cannot
        //    serve that range, and 1.0 is only meaningful at the top of it.
        //
        // 2. More decisively, the trigger buys nothing here whatever it is
        //    set to. Every finite value from 0.1 to 100 produced the same
        //    1370 merges and the same 1.83 mean position error as each
        //    other, at 0.15M ops/s against 3.4M with the trigger off — which
        //    reached 1.80. Twenty times the cost for no accuracy.
        //
        // The reason is architectural, and it is why this is a finding about
        // our design rather than a criticism of theirs. DynaMind needs the
        // trigger because its inserts land *in* the leaf array, so the model
        // in force starts drifting immediately. Ours land in the delta
        // buffer, and deletions are tombstoned rather than compacted, so the
        // base models stay exactly right for the base's contents until the
        // merge. There is no drift for the trigger to catch.
        //
        // The score itself is kept, and is worth keeping: it is the honest
        // measure of how far the data has moved, and staleness detection
        // elsewhere wants exactly that number.
        double score_threshold = std::numeric_limits<double>::infinity();

        // Writes to accept between evaluations of the score.
        //
        // Evaluating it costs one Cook's distance per model disturbed since
        // the last merge, so checking on every write would make writes cost
        // O(models disturbed) — unbounded when the score trigger is switched
        // off and merges are rare. Amortising over a small batch bounds that,
        // and cannot delay a merge by more than this many writes.
        std::size_t score_check_interval = 64;

        // Rebuild from scratch, refitting stage 1 as well, when a merge's
        // frozen routing has left the segments this badly unbalanced. Frozen
        // routing is what makes an incremental merge possible, and it is also
        // what slowly stops fitting as the data grows away from the shape
        // stage 1 was fitted to; this is the escape hatch.
        double rebuild_error_ratio = 4.0;
    };

    struct Stats {
        std::size_t size = 0;          // live keys: base - tombstones + delta
        std::size_t base_size = 0;
        std::size_t delta_size = 0;
        std::size_t tombstones = 0;
        std::size_t merges = 0;
        std::size_t full_rebuilds = 0;
        std::size_t models_shifted = 0;   // cumulative, O(1) each
        std::size_t models_refitted = 0;  // cumulative, O(segment) each
        std::size_t keys_rescanned = 0;   // cumulative
        double last_merge_seconds = 0.0;
        double total_merge_seconds = 0.0;
        std::size_t index_bytes = 0;
        // Mean error at the last merge. Measured there because a merge is
        // already an O(n) pass, so the figure is free at exactly the moment
        // it changes.
        double mean_error = 0.0;
        std::size_t max_error = 0;
        // Mean error immediately after the last *full* build. The yardstick
        // drift is measured against, since what matters is how far frozen
        // routing has slipped from a properly fitted stage 1, not the raw
        // number — which is distribution-dependent and not comparable across
        // columns.
        double baseline_mean_error = 0.0;
    };

    using Tree = BPlusTree<Key, Value>;
    using Base = RMIndex<Key, Value>;

    explicit DynamicRMIndex(Config config = Config{})
        : config_(config),
          base_(config.second_stage_size, config.search_threshold),
          delta_(config.delta_order) {
        if (!(config_.merge_ratio > 0.0)) {
            throw std::invalid_argument("DynamicRMIndex: merge_ratio must be > 0");
        }
    }

    const Config& config() const { return config_; }

    // ---- construction ---------------------------------------------------

    void build(const std::vector<Key>& keys, const std::vector<Value>& values) {
        base_.build(keys, values);
        base_.enable_incremental();
        delta_.clear();
        dead_.assign(keys.size(), false);
        tombstones_ = 0;
        touched_.clear();
        touched_flag_.assign(config_.second_stage_size, 0);
        writes_since_score_ = 0;
        stats_ = Stats{};
        refresh_stats();
        stats_.mean_error = base_.stats().mean_error;
        stats_.baseline_mean_error = stats_.mean_error;
    }

    // ---- reads ----------------------------------------------------------

    // Delta first: it holds the newer version of any key present in both.
    const Value* find(const Key& key) const {
        if (const Value* v = delta_.find(key)) return v;
        const std::size_t pos = base_.position_of(key);
        if (pos >= base_.size() || dead_[pos]) return nullptr;
        return &base_.values()[pos];
    }

    bool contains(const Key& key) const { return find(key) != nullptr; }

    std::vector<Value> range(const Key& lo, const Key& hi) const {
        std::vector<Value> out;
        if (hi < lo) return out;
        merge_streams(base_.lower_bound(lo), base_.upper_bound(hi),
                      delta_.range_items(lo, hi), out);
        return out;
    }

    // The predicate interface shared with the B+ tree and the static RMI, so
    // ColumnIndex can hold this without knowing it is holding it.
    std::vector<Value> range_query(CompareOp op, const Key& value) const {
        switch (op) {
            case CompareOp::Eq: {
                std::vector<Value> out;
                if (const Value* v = find(value)) out.push_back(*v);
                return out;
            }
            case CompareOp::Lt:
                return sliced(0, base_.lower_bound(value),
                              delta_.range_query_items(CompareOp::Lt, value));
            case CompareOp::Le:
                return sliced(0, base_.upper_bound(value),
                              delta_.range_query_items(CompareOp::Le, value));
            case CompareOp::Gt:
                return sliced(base_.upper_bound(value), base_.size(),
                              delta_.range_query_items(CompareOp::Gt, value));
            case CompareOp::Ge:
                return sliced(base_.lower_bound(value), base_.size(),
                              delta_.range_query_items(CompareOp::Ge, value));
        }
        return {};
    }

    std::size_t size() const {
        return base_.size() - tombstones_ + delta_.size();
    }
    bool empty() const { return size() == 0; }

    // ---- writes ---------------------------------------------------------

    // False if the key is already live, matching BPlusTree::insert.
    bool insert(const Key& key, const Value& value) {
        if (delta_.contains(key)) return false;
        const std::size_t pos = base_.position_of(key);
        const bool in_base = pos < base_.size();
        if (in_base && !dead_[pos]) return false;

        // A key that is present in base_ but tombstoned goes into the delta
        // alongside its own corpse. The tombstone stays set, so the merge
        // drops the old value and keeps this one, and until then find()
        // consults the delta first and gets the new value. Resurrecting the
        // base slot in place would be wrong: the value can differ, and base_
        // is not writable.
        delta_.insert(key, value);
        maybe_merge();
        return true;
    }

    bool erase(const Key& key) {
        if (delta_.erase(key)) {
            // Never reached base_, so there is nothing to unlearn. But it may
            // still be shadowing a tombstoned base entry, which must stay
            // tombstoned — and it does, untouched.
            maybe_merge();
            return true;
        }
        const std::size_t pos = base_.position_of(key);
        if (pos >= base_.size() || dead_[pos]) return false;

        dead_[pos] = true;
        ++tombstones_;
        // Machine unlearning: withdraw this key's contribution from its
        // model's moments. Exact and O(1) because nothing was renumbered.
        // The model itself is not moved here — its error bounds were measured
        // against it, and moving it without re-measuring them would let a
        // lookup miss a key that exists. The statistics simply run ahead of
        // the model, and the score below reports by how much.
        base_.unlearn_position(pos);
        mark_touched(base_.model_of_key(key));
        maybe_merge();
        return true;
    }

    // ---- merging --------------------------------------------------------

    // How far the deletions have pulled the disturbed models away from their
    // own data. Cook's distance; see RMIndex::model_score.
    double score() const { return base_.max_score(touched_); }

    // Whether a merge would happen if the triggers were evaluated right now.
    // Ignores score_check_interval, which is a sampling rate rather than part
    // of the condition.
    bool merge_due() const {
        const std::size_t pending = delta_.size() + tombstones_;
        if (pending == 0) return false;
        return size_trigger(pending) || score() > config_.score_threshold;
    }

    // Fold the delta buffer and the tombstones back into the base.
    //
    // Both inputs are already ordered, so this is one linear pass, and the
    // model work afterwards is proportional to how *localised* the changes
    // were rather than to n — see RMIndex::merge_from.
    void merge() {
        if (delta_.empty() && tombstones_ == 0) return;

        const auto start = std::chrono::steady_clock::now();

        const std::vector<Key>& bk = base_.keys();
        const std::vector<Value>& bv = base_.values();
        const std::vector<std::pair<Key, Value>> dv = delta_.items();

        std::vector<Key> keys;
        std::vector<Value> values;
        std::vector<Key> inserted;
        std::vector<Key> deleted;
        keys.reserve(size());
        values.reserve(size());
        inserted.reserve(dv.size());
        deleted.reserve(tombstones_);

        std::size_t i = 0, j = 0;
        while (i < bk.size() || j < dv.size()) {
            const bool take_delta =
                i >= bk.size() || (j < dv.size() && dv[j].first < bk[i]);
            if (take_delta) {
                keys.push_back(dv[j].first);
                values.push_back(dv[j].second);
                inserted.push_back(dv[j].first);
                ++j;
                continue;
            }
            if (dead_[i]) {
                deleted.push_back(bk[i]);
                // A delta entry with the same key is this key's replacement,
                // not an insert of a new one. Emit it here so it takes the
                // dead entry's place, and report the pair as a delete and an
                // insert of the same key, which nets to no position change.
                if (j < dv.size() && !(bk[i] < dv[j].first)) {
                    keys.push_back(dv[j].first);
                    values.push_back(dv[j].second);
                    inserted.push_back(dv[j].first);
                    ++j;
                }
                ++i;
                continue;
            }
            keys.push_back(bk[i]);
            values.push_back(bv[i]);
            ++i;
        }

        const typename Base::MergeStats ms =
            base_.merge_from(keys, values, inserted, deleted);

        delta_.clear();
        dead_.assign(keys.size(), false);
        tombstones_ = 0;
        touched_.clear();
        std::fill(touched_flag_.begin(), touched_flag_.end(), 0);
        writes_since_score_ = 0;

        ++stats_.merges;
        stats_.full_rebuilds += ms.full_rebuild ? 1 : 0;
        stats_.models_shifted += ms.models_shifted;
        stats_.models_refitted += ms.models_refitted;
        stats_.keys_rescanned += ms.keys_rescanned;

        // A merge is already an O(n) pass over the keys, so measuring the
        // error here costs a constant factor rather than an order, and it is
        // the only moment the figure can have changed.
        const double mean = base_.measure_mean_error();
        const bool drifted =
            stats_.baseline_mean_error > 0.0 &&
            mean > config_.rebuild_error_ratio * stats_.baseline_mean_error;
        if (drifted && !ms.full_rebuild) {
            // Frozen routing has stopped fitting. Refit stage 1 too.
            base_.build(keys, values);
            base_.enable_incremental();
            ++stats_.full_rebuilds;
            stats_.baseline_mean_error = base_.stats().mean_error;
            stats_.mean_error = stats_.baseline_mean_error;
        } else {
            if (stats_.baseline_mean_error <= 0.0 || ms.full_rebuild) {
                stats_.baseline_mean_error = mean;
            }
            stats_.mean_error = mean;
        }

        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        stats_.last_merge_seconds = seconds;
        stats_.total_merge_seconds += seconds;
        refresh_stats();
    }

    // ---- introspection --------------------------------------------------

    Stats stats() const {
        Stats out = stats_;
        out.size = size();
        out.base_size = base_.size();
        out.delta_size = delta_.size();
        out.tombstones = tombstones_;
        out.max_error = base_.stats().max_error;
        out.index_bytes = base_.stats().total_bytes + delta_.memory_bytes() +
                          dead_.capacity() / 8;
        return out;
    }

    const Base& base() const { return base_; }
    const Tree& delta() const { return delta_; }

    // Every invariant this structure claims, checked.
    //
    // base_.validate() is the strong one: it replays every stored key through
    // the lookup path and insists the predicted window contains it. That has
    // to keep holding after an incremental merge, or the merge has silently
    // broken the exactness guarantee — which is the single most important
    // thing that could go wrong in this file.
    void validate() const {
        base_.validate();
        delta_.validate();
        if (dead_.size() != base_.size()) {
            throw std::logic_error("DynamicRMIndex::validate: tombstone bitmap is " +
                                   std::to_string(dead_.size()) + " long for " +
                                   std::to_string(base_.size()) + " base keys");
        }
        std::size_t counted = 0;
        for (std::size_t i = 0; i < dead_.size(); ++i) counted += dead_[i] ? 1 : 0;
        if (counted != tombstones_) {
            throw std::logic_error("DynamicRMIndex::validate: " +
                                   std::to_string(counted) + " tombstones set but " +
                                   std::to_string(tombstones_) + " counted");
        }
    }

private:
    bool size_trigger(std::size_t pending) const {
        const std::size_t base_n = base_.size() ? base_.size() : 1;
        return static_cast<double>(pending) >=
               config_.merge_ratio * static_cast<double>(base_n);
    }

    void mark_touched(std::size_t m) {
        if (m >= touched_flag_.size() || touched_flag_[m]) return;
        touched_flag_[m] = 1;
        touched_.push_back(m);
    }

    // The size trigger is cheap enough to evaluate on every write; the score
    // is not, so it is sampled. See Config::score_check_interval.
    void maybe_merge() {
        const std::size_t pending = delta_.size() + tombstones_;
        if (pending == 0) return;
        if (size_trigger(pending)) {
            merge();
            return;
        }
        if (++writes_since_score_ < config_.score_check_interval) return;
        writes_since_score_ = 0;
        if (score() > config_.score_threshold) merge();
    }

    void refresh_stats() {
        stats_.size = size();
        stats_.base_size = base_.size();
        stats_.delta_size = delta_.size();
        stats_.tombstones = tombstones_;
    }

    // Interleave the live part of a base slice with a delta slice.
    //
    // Both are ascending, so this is the merge step of a merge sort. A key in
    // both is the delta's — same rule find() follows, which is what keeps a
    // point lookup and a range that spans it from disagreeing.
    void merge_streams(std::size_t begin, std::size_t end,
                       const std::vector<std::pair<Key, Value>>& delta,
                       std::vector<Value>& out) const {
        const std::vector<Key>& bk = base_.keys();
        const std::vector<Value>& bv = base_.values();
        std::size_t i = begin, j = 0;
        while (i < end || j < delta.size()) {
            if (i < end && dead_[i]) { ++i; continue; }
            if (i >= end) { out.push_back(delta[j].second); ++j; continue; }
            if (j >= delta.size()) { out.push_back(bv[i]); ++i; continue; }
            if (delta[j].first < bk[i]) {
                out.push_back(delta[j].second);
                ++j;
            } else if (bk[i] < delta[j].first) {
                out.push_back(bv[i]);
                ++i;
            } else {
                out.push_back(delta[j].second);  // delta wins
                ++i;
                ++j;
            }
        }
    }

    std::vector<Value> sliced(std::size_t begin, std::size_t end,
                              const std::vector<std::pair<Key, Value>>& delta) const {
        std::vector<Value> out;
        merge_streams(begin, end, delta, out);
        return out;
    }

    Config config_;
    Base base_;
    Tree delta_;
    // One bit per base position. std::vector<bool> is the right container
    // here for once: the bitmap is read on the hot path of every lookup, and
    // eight times fewer cache lines matters more than the proxy references.
    std::vector<bool> dead_;
    std::size_t tombstones_ = 0;
    // Models disturbed since the last merge, for score(). Duplicates are
    // harmless — max_score takes a maximum — and de-duplicating would cost
    // more than the repeated evaluations do.
    std::vector<std::size_t> touched_;
    // Membership side of touched_, so marking is O(1) and the list holds each
    // model once. Without it a hot model is appended on every deletion and
    // scoring re-evaluates it that many times.
    std::vector<char> touched_flag_;
    std::size_t writes_since_score_ = 0;
    Stats stats_;
};

}  // namespace hylis::index
