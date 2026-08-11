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
#include "index/dynamic_rmi.hpp"
#include "index/rmi.hpp"

namespace hylis::index {

using ColumnKey = std::int64_t;
using ColumnValue = std::int64_t;

enum class IndexKind { BPlusTree, RMI, DynamicRMI };

inline const char* to_string(IndexKind kind) {
    switch (kind) {
        case IndexKind::BPlusTree: return "btree";
        case IndexKind::RMI: return "rmi";
        case IndexKind::DynamicRMI: return "dynamic_rmi";
    }
    return "btree";
}

inline IndexKind index_kind_from_string(const std::string& text) {
    if (text == "btree") return IndexKind::BPlusTree;
    if (text == "rmi") return IndexKind::RMI;
    if (text == "dynamic_rmi") return IndexKind::DynamicRMI;
    throw std::invalid_argument("unknown index kind: " + text);
}

// Whether a kind can accept writes after it is built. The static RMI cannot,
// and that is not a limitation to route around — it is the whole reason the
// choice below has to know the write rate.
inline bool is_mutable(IndexKind kind) { return kind != IndexKind::RMI; }

// What the column is going to be used for.
//
// choose_index() times lookups, and on lookups alone the static RMI wins
// nearly everything — so without this it would be picked for every column,
// including ones that are about to be written to and cannot take it. The
// write rate is the missing input, not a refinement.
struct Workload {
    // Fraction of operations that modify the column. Zero means read-only,
    // and only then is an immutable structure a legal answer.
    double write_fraction = 0.0;

    bool read_only() const { return write_fraction <= 0.0; }
};

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

    // DynamicRMI only.
    double merge_ratio = 0.05;

    // Measured at selection time on this machine, not assumed.
    double ns_per_lookup = 0.0;
    double ns_per_write = 0.0;    // 0 when the column was chosen read-only
    std::size_t max_error = 0;    // RMI only; 0 for a tree
    std::size_t index_bytes = 0;

    // The write rate this plan was chosen for. A plan picked for a read-only
    // column is not evidence about a column that has started taking writes,
    // even if the keys are untouched — so this is part of what makes a plan
    // reusable, alongside the data fingerprint below.
    double write_fraction = 0.0;

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

    // Whether this plan is a legal answer for the given workload at all.
    //
    // Distinct from matches(): that asks whether the plan is still evidence,
    // this asks whether it is still *permissible*. An immutable structure
    // chosen for a read-only column becomes illegal the moment the column
    // starts taking writes, however well it fits the keys.
    bool serves(const Workload& workload) const {
        return workload.read_only() || is_mutable(kind);
    }
};

// Defined below, once ColumnIndex exists — it has to build candidates to
// measure them. Declared here so ColumnIndex::build can call it.
inline IndexPlan choose_index(
    const std::vector<ColumnKey>& keys,
    const std::vector<ColumnValue>& values,
    std::size_t size_budget = std::numeric_limits<std::size_t>::max(),
    Workload workload = Workload{});

class ColumnIndex {
public:
    using Tree = BPlusTree<ColumnKey, ColumnValue>;
    using Learned = RMIndex<ColumnKey, ColumnValue>;
    using Writable = DynamicRMIndex<ColumnKey, ColumnValue>;

    // Auto-tune: build every candidate, time it, keep the best.
    static ColumnIndex build(const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values,
                             std::size_t size_budget = std::numeric_limits<std::size_t>::max(),
                             Workload workload = Workload{}) {
        return build_with(keys, values,
                          choose_index(keys, values, size_budget, workload));
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

        if (plan.kind == IndexKind::DynamicRMI) {
            typename Writable::Config cfg;
            cfg.second_stage_size = std::max<std::size_t>(plan.rmi_models, 1);
            cfg.search_threshold = plan.search_threshold;
            cfg.merge_ratio = plan.merge_ratio > 0.0 ? plan.merge_ratio : 0.05;
            out.dynamic_ = std::make_unique<Writable>(cfg);
            out.dynamic_->build(keys, values);
            const auto stats = out.dynamic_->stats();
            out.plan_.max_error = stats.max_error;
            out.plan_.index_bytes = stats.index_bytes;
        } else if (plan.kind == IndexKind::RMI) {
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
        if (dynamic_) return dynamic_->find(key);
        return rmi_ ? rmi_->find(key) : tree_->find(key);
    }

    bool contains(const ColumnKey& key) const { return find(key) != nullptr; }

    std::vector<ColumnValue> range(const ColumnKey& lo, const ColumnKey& hi) const {
        if (dynamic_) return dynamic_->range(lo, hi);
        return rmi_ ? rmi_->range(lo, hi) : tree_->range(lo, hi);
    }

    std::vector<ColumnValue> range_query(CompareOp op, const ColumnKey& value) const {
        if (dynamic_) return dynamic_->range_query(op, value);
        return rmi_ ? rmi_->range_query(op, value) : tree_->range_query(op, value);
    }

    // Writes. Available on every kind except the static RMI, which cannot take
    // them at all — and rather than silently returning false, which would look
    // like "that key was already there", it says so.
    bool is_mutable() const { return !rmi_; }

    bool insert(const ColumnKey& key, const ColumnValue& value) {
        if (dynamic_) return dynamic_->insert(key, value);
        if (tree_) return tree_->insert(key, value);
        throw std::logic_error(
            "ColumnIndex::insert: this column was given a static RMI, which is "
            "build-only. It was chosen under a read-only workload; re-tune with "
            "a non-zero Workload::write_fraction to get a writable structure.");
    }

    bool erase(const ColumnKey& key) {
        if (dynamic_) return dynamic_->erase(key);
        if (tree_) return tree_->erase(key);
        throw std::logic_error(
            "ColumnIndex::erase: this column was given a static RMI, which is "
            "build-only. See insert().");
    }

    std::size_t size() const {
        if (dynamic_) return dynamic_->size();
        return rmi_ ? rmi_->size() : tree_->size();
    }
    bool empty() const { return size() == 0; }
    IndexKind kind() const { return plan_.kind; }
    const IndexPlan& plan() const { return plan_; }

    void validate() const {
        if (dynamic_) {
            dynamic_->validate();
        } else if (rmi_) {
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
    std::unique_ptr<Writable> dynamic_;
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

// Time a mixed write stream: alternating inserts of absent keys and erases of
// present ones, which is the 50/50 split the DynaMind evaluation uses.
//
// Timed on a fresh copy per measurement, because a write stream changes the
// structure it is measured on — a B+ tree splits nodes, a dynamic RMI merges
// — so repeating on one instance would measure a different index each time.
inline double time_writes(const std::vector<ColumnKey>& keys,
                          const std::vector<ColumnValue>& values,
                          const IndexPlan& candidate,
                          std::size_t count) {
    if (keys.empty() || count == 0 || !is_mutable(candidate.kind)) return 0.0;

    ColumnIndex index = ColumnIndex::build_with(keys, values, candidate);
    std::mt19937_64 rng(54321);  // fixed: every candidate sees the same stream
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);

    std::vector<ColumnKey> inserts;
    std::vector<ColumnKey> erases;
    inserts.reserve(count);
    erases.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        inserts.push_back(keys[pick(rng)] + 1);
        erases.push_back(keys[pick(rng)]);
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < count; ++i) {
        index.insert(inserts[i], static_cast<ColumnValue>(i));
        index.erase(erases[i]);
    }
    const auto stop = std::chrono::steady_clock::now();

    const double ns = std::chrono::duration<double, std::nano>(stop - start).count();
    return ns / (2.0 * static_cast<double>(count));
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
                              const IndexPlan& candidate,
                              Workload workload = Workload{}) {
    const ColumnIndex built = ColumnIndex::build_with(keys, values, candidate);
    IndexPlan measured = built.plan();
    measured.ns_per_lookup = detail::time_lookups(built, detail::make_probes(keys, 2000));
    measured.write_fraction = workload.write_fraction;
    if (!workload.read_only()) {
        measured.ns_per_write = detail::time_writes(
            keys, values, candidate, std::min<std::size_t>(keys.size(), 2000));
    }
    return measured;
}

// What one operation costs on average under this workload. The single number
// the candidates are ranked by, and the reason the write rate has to be an
// input: at 0% writes it is the lookup time, at 100% it is the write time,
// and the crossover in between is exactly the decision being made.
inline double blended_cost(const IndexPlan& plan, const Workload& workload) {
    const double w = workload.write_fraction;
    return (1.0 - w) * plan.ns_per_lookup + w * plan.ns_per_write;
}

// Build each candidate, time it, and return the plan for the fastest one that
// fits the budget. Ties break toward the smaller index.
inline IndexPlan choose_index(const std::vector<ColumnKey>& keys,
                              const std::vector<ColumnValue>& values,
                              std::size_t size_budget,
                              Workload workload) {
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

        // The static RMI is only a legal answer for a read-only column. It is
        // build-only, so offering it to a column that takes writes would be
        // offering something that cannot do the job, however fast it looks on
        // a lookup benchmark.
        if (workload.read_only()) {
            IndexPlan learned;
            learned.kind = IndexKind::RMI;
            learned.rmi_models = models;
            learned.search_threshold = 64;
            candidates.push_back(learned);
        } else {
            for (double ratio : {0.02, 0.05, 0.20}) {
                IndexPlan writable;
                writable.kind = IndexKind::DynamicRMI;
                writable.rmi_models = models;
                writable.search_threshold = 64;
                writable.merge_ratio = ratio;
                candidates.push_back(writable);
            }
        }
    }

    const std::vector<ColumnKey> probes = detail::make_probes(keys, 2000);
    const std::size_t write_samples = std::min<std::size_t>(keys.size(), 2000);

    bool have_best = false;
    IndexPlan best;
    double best_cost = 0.0;
    for (const IndexPlan& candidate : candidates) {
        const ColumnIndex built = ColumnIndex::build_with(keys, values, candidate);
        IndexPlan measured = built.plan();
        measured.ns_per_lookup = detail::time_lookups(built, probes);
        measured.write_fraction = workload.write_fraction;
        if (!workload.read_only()) {
            measured.ns_per_write =
                detail::time_writes(keys, values, candidate, write_samples);
        }

        if (measured.index_bytes > size_budget) continue;

        const double cost = blended_cost(measured, workload);
        const bool better =
            !have_best || cost < best_cost ||
            (cost == best_cost && measured.index_bytes < best.index_bytes);
        if (better) {
            best = measured;
            best_cost = cost;
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
        best.write_fraction = workload.write_fraction;
    }
    return best;
}

}  // namespace hylis::index
