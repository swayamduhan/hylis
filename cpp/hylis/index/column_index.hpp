// index/column_index.hpp
//
// Picking an index per column, and hiding which one was picked.
//
// Why this layer exists
// ---------------------
// Different columns have different key distributions. An auto-increment id is
// near-linear and a learned index eats it; a column of clustered ids has
// cliffs in its CDF that no piecewise-linear model can absorb, and a B+ tree
// — which is distribution-free — wins outright. There is no single right
// answer, so the choice has to be made per column.
//
// How the choice is made: by measuring
// ------------------------------------
// choose_index() builds every candidate and *times real lookups on each*,
// then keeps the winner. The obvious alternative — an analytic cost model
// over assumed cache-miss and branch-misprediction constants — would be
// unfalsifiable and would need retuning for every machine it ran on. Building
// an RMI costs ~20ms per million keys, so trying half a dozen configurations
// is seconds of one-time work in exchange for a number that is actually true
// on the hardware in front of it. This is the approach CDFShop takes
// (Marcus et al., SIGMOD 2020).
//
// Why ColumnIndex hides the result
// --------------------------------
// It exposes the same find/range_query pair whichever structure is inside.
// That is what the CompareOp contract is for: the query planner asks a column
// for the rows matching a predicate and never learns what answered. Swapping
// a tree for a model is then a performance decision, not an interface change.
//
// Keys and values are fixed at std::int64_t here rather than templated. That
// is what the planner and the bindings actually use, and it keeps IndexPlan a
// plain serialisable struct — which matters, because plans get persisted.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/btree.hpp"
#include "index/compare_op.hpp"
#include "index/rmi.hpp"

namespace hylis::index {

using ColumnKey = std::int64_t;
using ColumnValue = std::int64_t;

enum class IndexKind { BPlusTree, RMI };

inline const char* to_string(IndexKind kind) {
    return kind == IndexKind::BPlusTree ? "btree" : "rmi";
}

inline IndexKind index_kind_from_string(const std::string& text) {
    if (text == "btree") return IndexKind::BPlusTree;
    if (text == "rmi") return IndexKind::RMI;
    throw std::invalid_argument("unknown index kind: " + text);
}

// The decision, plus the evidence behind it.
//
// The measured fields are why this is worth persisting: reproducing them
// means rebuilding and re-timing every candidate, which is the expensive part
// of choosing. The fingerprint fields are what make a stored plan safe to
// reuse — see matches().
struct IndexPlan {
    IndexKind kind = IndexKind::BPlusTree;

    // Structure parameters. Only the one belonging to `kind` is meaningful.
    std::size_t rmi_models = 1024;
    std::size_t search_threshold = 64;
    std::size_t btree_order = 32;

    // Measured at selection time on this machine, not assumed.
    double ns_per_lookup = 0.0;
    std::size_t max_error = 0;    // RMI only; 0 for a tree
    std::size_t index_bytes = 0;

    // Fingerprint of the data this was chosen for.
    std::size_t n_keys = 0;
    ColumnKey key_min = 0;
    ColumnKey key_max = 0;

    // Whether this plan is still *evidence* about the given keys.
    //
    // Cheap on purpose: length plus both endpoints, not a content hash. The
    // consequence of a false positive is only that a column keeps a
    // now-suboptimal structure, because correctness never depends on the plan
    // — every model and every error bound is recomputed at build time
    // regardless of which plan was followed. Paying O(n) to hash the column
    // on every startup would cost more than the mistake does.
    bool matches(const std::vector<ColumnKey>& keys) const {
        if (keys.size() != n_keys) return false;
        if (keys.empty()) return true;
        return keys.front() == key_min && keys.back() == key_max;
    }
};

// Defined below, once ColumnIndex exists — it has to build candidates to
// measure them. Declared here so ColumnIndex::build can call it.
inline IndexPlan choose_index(
    const std::vector<ColumnKey>& keys,
    const std::vector<ColumnValue>& values,
    std::size_t size_budget = std::numeric_limits<std::size_t>::max());

class ColumnIndex {
public:
    using Tree = BPlusTree<ColumnKey, ColumnValue>;
    using Learned = RMIndex<ColumnKey, ColumnValue>;

    // Auto-tune: build every candidate, time it, keep the best.
    static ColumnIndex build(const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values,
                             std::size_t size_budget = std::numeric_limits<std::size_t>::max()) {
        return build_with(keys, values, choose_index(keys, values, size_budget));
    }

    // Replay a decision made earlier, skipping the measurement.
    //
    // The plan's structural fields are obeyed; its measured and fingerprint
    // fields are recomputed for the data actually supplied, so the resulting
    // plan() always describes this index rather than a remembered one.
    static ColumnIndex build_with(const std::vector<ColumnKey>& keys,
                                  const std::vector<ColumnValue>& values,
                                  const IndexPlan& plan) {
        if (keys.size() != values.size()) {
            throw std::invalid_argument(
                "ColumnIndex: " + std::to_string(keys.size()) + " keys but " +
                std::to_string(values.size()) + " values");
        }

        ColumnIndex out;
        out.plan_ = plan;
        out.plan_.n_keys = keys.size();
        out.plan_.key_min = keys.empty() ? 0 : keys.front();
        out.plan_.key_max = keys.empty() ? 0 : keys.back();

        if (plan.kind == IndexKind::RMI) {
            out.rmi_ = std::make_unique<Learned>(std::max<std::size_t>(plan.rmi_models, 1),
                                                 plan.search_threshold);
            out.rmi_->build(keys, values);
            const auto stats = out.rmi_->stats();
            out.plan_.max_error = stats.max_error;
            out.plan_.index_bytes = stats.total_bytes;
        } else {
            out.tree_ = std::make_unique<Tree>(std::max<std::size_t>(plan.btree_order, 3));
            for (std::size_t i = 0; i < keys.size(); ++i) {
                out.tree_->insert(keys[i], values[i]);
            }
            out.plan_.max_error = 0;
            out.plan_.index_bytes = out.tree_->memory_bytes();
        }
        return out;
    }

    const ColumnValue* find(const ColumnKey& key) const {
        return rmi_ ? rmi_->find(key) : tree_->find(key);
    }

    bool contains(const ColumnKey& key) const { return find(key) != nullptr; }

    std::vector<ColumnValue> range(const ColumnKey& lo, const ColumnKey& hi) const {
        return rmi_ ? rmi_->range(lo, hi) : tree_->range(lo, hi);
    }

    std::vector<ColumnValue> range_query(CompareOp op, const ColumnKey& value) const {
        return rmi_ ? rmi_->range_query(op, value) : tree_->range_query(op, value);
    }

    std::size_t size() const { return rmi_ ? rmi_->size() : tree_->size(); }
    bool empty() const { return size() == 0; }
    IndexKind kind() const { return plan_.kind; }
    const IndexPlan& plan() const { return plan_; }

    void validate() const {
        if (rmi_) {
            rmi_->validate();
        } else {
            tree_->validate();
        }
    }

private:
    ColumnIndex() = default;

    IndexPlan plan_;
    std::unique_ptr<Tree> tree_;
    std::unique_ptr<Learned> rmi_;
};

namespace detail {

// A spread of keys to probe with: mostly present, some absent, so a candidate
// cannot look good by being fast only on hits.
inline std::vector<ColumnKey> make_probes(const std::vector<ColumnKey>& keys,
                                          std::size_t count) {
    std::vector<ColumnKey> probes;
    if (keys.empty()) return probes;
    count = std::min(count, std::max<std::size_t>(keys.size(), 1) * 2);
    probes.reserve(count);

    std::mt19937_64 rng(12345);  // fixed: every candidate sees the same probes
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
    for (std::size_t i = 0; i < count; ++i) {
        const ColumnKey k = keys[pick(rng)];
        probes.push_back((i % 4 == 3) ? k + 1 : k);
    }
    return probes;
}

inline double time_lookups(const ColumnIndex& index,
                           const std::vector<ColumnKey>& probes) {
    if (probes.empty()) return 0.0;

    // One untimed pass so every candidate is measured warm; otherwise the
    // first candidate would be charged for the cold caches of all of them.
    volatile std::int64_t sink = 0;
    for (ColumnKey k : probes) {
        const ColumnValue* v = index.find(k);
        if (v) sink = sink + *v;
    }

    const int reps = probes.size() < 1000 ? 20 : 3;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (ColumnKey k : probes) {
            const ColumnValue* v = index.find(k);
            // Consumed through a volatile so the compiler cannot delete the
            // lookups it can see are otherwise unused.
            if (v) sink = sink + *v;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    (void)sink;

    const double ns = std::chrono::duration<double, std::nano>(stop - start).count();
    return ns / (static_cast<double>(reps) * static_cast<double>(probes.size()));
}

}  // namespace detail

// Build one candidate and time it, returning the plan with its measured
// fields filled in.
//
// Exposed rather than kept private because it is the only honest way to time
// a lookup from outside C++: measuring through a language binding would
// charge the index for the bridge, which costs more than the lookup does.
inline IndexPlan measure_plan(const std::vector<ColumnKey>& keys,
                              const std::vector<ColumnValue>& values,
                              const IndexPlan& candidate) {
    const ColumnIndex built = ColumnIndex::build_with(keys, values, candidate);
    IndexPlan measured = built.plan();
    measured.ns_per_lookup = detail::time_lookups(built, detail::make_probes(keys, 2000));
    return measured;
}

// Build each candidate, time it, and return the plan for the fastest one that
// fits the budget. Ties break toward the smaller index.
inline IndexPlan choose_index(const std::vector<ColumnKey>& keys,
                              const std::vector<ColumnValue>& values,
                              std::size_t size_budget) {
    std::vector<IndexPlan> candidates;

    IndexPlan tree;
    tree.kind = IndexKind::BPlusTree;
    tree.btree_order = 32;
    candidates.push_back(tree);

    // Skip model counts that exceed the key count: past that point almost
    // every model is routed nothing, so it is pure overhead.
    for (std::size_t models : {std::size_t{64}, std::size_t{1024},
                               std::size_t{16384}, std::size_t{262144}}) {
        if (!keys.empty() && models > keys.size()) break;
        IndexPlan learned;
        learned.kind = IndexKind::RMI;
        learned.rmi_models = models;
        learned.search_threshold = 64;
        candidates.push_back(learned);
    }

    const std::vector<ColumnKey> probes = detail::make_probes(keys, 2000);

    bool have_best = false;
    IndexPlan best;
    for (const IndexPlan& candidate : candidates) {
        const ColumnIndex built = ColumnIndex::build_with(keys, values, candidate);
        IndexPlan measured = built.plan();
        measured.ns_per_lookup = detail::time_lookups(built, probes);

        if (measured.index_bytes > size_budget) continue;

        const bool better =
            !have_best ||
            measured.ns_per_lookup < best.ns_per_lookup ||
            (measured.ns_per_lookup == best.ns_per_lookup &&
             measured.index_bytes < best.index_bytes);
        if (better) {
            best = measured;
            have_best = true;
        }
    }

    // Nothing fit the budget. A B+ tree is the honest fallback: it is the
    // structure with no tunable overhead to trade away, and returning
    // something correct beats refusing to index the column.
    if (!have_best) {
        best = tree;
        best.n_keys = keys.size();
        best.key_min = keys.empty() ? 0 : keys.front();
        best.key_max = keys.empty() ? 0 : keys.back();
    }
    return best;
}

}  // namespace hylis::index
