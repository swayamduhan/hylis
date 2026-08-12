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
// How the choice is made: type eliminates, measurement decides
// -----------------------------------------------------------
// Two stages, and only the second is a cost comparison.
//
// The *logical type* of a column decides which families are candidates at all.
// That is a hard filter, not a preference: RMIndex fits models to
// static_cast<double>(key), so a learned index over strings would be fitting a
// model to an ordering the model itself imposed. BPlusTree, by contrast,
// touches keys only through lower_bound and ==, so it serves every ordered
// type natively — which is why a string column here is indexed as a string and
// never encoded to an integer to reach a structure that never wanted it.
//
// Among the survivors, choose_index() builds every candidate and *times real
// lookups on each*, then keeps the winner. The obvious alternative — an
// analytic cost model over assumed cache-miss and branch-misprediction
// constants — would be unfalsifiable and would need retuning for every machine
// it ran on. Building an RMI costs ~20ms per million keys, so trying half a
// dozen configurations is seconds of one-time work in exchange for a number
// that is actually true on the hardware in front of it. This is the approach
// CDFShop takes (Marcus et al., SIGMOD 2020).
//
// Why ColumnIndex hides the result
// --------------------------------
// It exposes the same find/range_query pair whichever structure is inside.
// That is what the CompareOp contract is for: the query planner asks a column
// for the rows matching a predicate and never learns what answered. Swapping
// a tree for a model is then a performance decision, not an interface change.
//
// What type erasure costs, and where it does not
// ----------------------------------------------
// Holding four key types behind one class means one virtual call — on
// range_query, once per predicate. It is deliberately *not* on key comparison,
// where an indirect call per probe would be ruinous. A predicate returns
// thousands of row ids, so the dispatch is unmeasurable against the work it
// dispatches. Templating ColumnIndex instead would push the key type into the
// planner's column map, into the catalog, and into every binding, which is a
// much larger cost paid in a much worse place.
//
// The payload is always a row id (int64), for every key type and every
// encoding. Only the *key* varies, which halves the instantiation matrix and
// keeps IndexPlan a plain serialisable struct — and plans get persisted.

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
#include <type_traits>
#include <utility>
#include <vector>

#include "index/btree.hpp"
#include "index/compare_op.hpp"
#include "index/dynamic_rmi.hpp"
#include "index/logical_type.hpp"
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

// What one pass over a sorted column tells us, and what the candidate filter
// needs. Collected during extraction, which has to sort anyway, so it is free.
struct ColumnShape {
    std::size_t rows = 0;
    std::size_t distinct = 0;
    bool unique = false;
    // Whether the column arrives already in order: row ids ascending when the
    // keys are. This is what makes an append-only write path legal, and it is
    // why Timestamp is a distinct logical type rather than an alias for Int64.
    bool monotone = false;

    double duplicate_fraction() const {
        return rows ? 1.0 - static_cast<double>(distinct) / static_cast<double>(rows)
                    : 0.0;
    }
};

// The decision, plus the evidence behind it.
//
// The measured fields are why this is worth persisting: reproducing them
// means rebuilding and re-timing every candidate, which is the expensive part
// of choosing. The fingerprint fields are what make a stored plan safe to
// reuse — see matches().
struct IndexPlan {
    IndexKind kind = IndexKind::BPlusTree;

    // What the column *is*, and how its values become index keys. Together
    // these decide which structures were even candidates — see
    // candidates_for(). Defaults keep every existing int64 caller unchanged.
    LogicalType type = LogicalType::Int64;
    KeyEncoding encoding = KeyEncoding::Native;

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

    // Shape, recorded because it is what the candidate filter consumed. A plan
    // that cannot say why a structure was eligible is not reproducible.
    std::size_t distinct = 0;
    bool monotone = false;

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

namespace detail {

// ---------------------------------------------------------------------------
// The erased scalar index
// ---------------------------------------------------------------------------
//
// Every method here is per-*predicate*, never per-key — see the note at the
// top of this file on where the virtual call is and is not.
class ScalarIndex {
public:
    virtual ~ScalarIndex() = default;

    virtual std::vector<ColumnValue> range_query(CompareOp op,
                                                 const Datum& value) const = 0;
    virtual std::vector<ColumnValue> range(const Datum& lo,
                                           const Datum& hi) const = 0;
    // Every row whose value equals this one. The general form of find(), and
    // the only correct one for a column that repeats values.
    virtual std::vector<ColumnValue> lookup(const Datum& value) const = 0;
    virtual std::vector<ColumnValue> prefix(const std::string& p) const = 0;

    // A borrowed pointer to the single row holding this key, or nullptr — both
    // when the key is absent and when the encoding cannot promise a single
    // row. Exists so the int64 fast path keeps its original zero-allocation
    // shape; lookup() is the general answer.
    virtual const ColumnValue* find_native(const Datum& value) const = 0;

    // The same, without a Datum.
    //
    // Not premature: std::variant<..., std::string, ...> has a non-trivial
    // destructor, so wrapping every probe would add a switch-and-destroy to a
    // path that costs ~32 ns in total. That is a few percent, which is enough
    // to move every lookup figure this project has already published — and a
    // measurement that changes because of a refactor is worse than no
    // measurement. Defaults return nullptr, which is the right answer for
    // every index whose keys are of another type or whose encoding names no
    // single row.
    virtual const ColumnValue* find_i64(ColumnKey) const { return nullptr; }
    virtual const ColumnValue* find_f64(double) const { return nullptr; }

    virtual bool insert(const Datum& value, const ColumnValue& row) = 0;
    virtual bool erase(const Datum& value, const ColumnValue& row) = 0;

    virtual bool is_mutable() const = 0;
    virtual bool is_native() const = 0;
    virtual std::size_t size() const = 0;
    virtual std::size_t bytes() const = 0;
    virtual std::size_t max_error() const = 0;
    virtual void validate() const = 0;
};

// Which key types a learned index can serve. Not a policy — a consequence of
// RMIndex casting keys to double to fit them.
template <typename T>
inline constexpr bool kRmiCapable = std::is_arithmetic_v<T>;

template <typename T>
const T& datum_as(const Datum& d) {
    if (const T* p = std::get_if<T>(&d)) return *p;
    throw std::invalid_argument(
        "ColumnIndex: the predicate's value is not of this column's type");
}

inline std::vector<ColumnValue> one_or_none(const ColumnValue* v) {
    return v ? std::vector<ColumnValue>{*v} : std::vector<ColumnValue>{};
}

// --- B+ tree, keyed on the value itself -------------------------------------
//
// Legal only when the column's values are unique, because every ordered
// structure here maps one key to one row.
template <typename T>
class NativeTree final : public ScalarIndex {
public:
    explicit NativeTree(std::size_t order) : tree_(order) {}

    void load(const std::vector<T>& keys, const std::vector<ColumnValue>& values) {
        for (std::size_t i = 0; i < keys.size(); ++i) tree_.insert(keys[i], values[i]);
    }

    std::vector<ColumnValue> range_query(CompareOp op, const Datum& v) const override {
        return tree_.range_query(op, datum_as<T>(v));
    }
    std::vector<ColumnValue> range(const Datum& lo, const Datum& hi) const override {
        return tree_.range(datum_as<T>(lo), datum_as<T>(hi));
    }
    std::vector<ColumnValue> lookup(const Datum& v) const override {
        return one_or_none(tree_.find(datum_as<T>(v)));
    }
    std::vector<ColumnValue> prefix(const std::string& p) const override {
        if constexpr (std::is_same_v<T, std::string>) {
            std::string upper;
            if (!prefix_upper_bound(p, &upper)) {
                return tree_.range_query(CompareOp::Ge, p);
            }
            return tree_.range_half_open(p, upper);
        } else {
            (void)p;
            throw std::invalid_argument(
                "ColumnIndex::prefix: only string columns have prefixes");
        }
    }
    const ColumnValue* find_native(const Datum& v) const override {
        return tree_.find(datum_as<T>(v));
    }
    const ColumnValue* find_i64(ColumnKey k) const override {
        if constexpr (std::is_same_v<T, ColumnKey>) return tree_.find(k);
        else { (void)k; return nullptr; }
    }
    const ColumnValue* find_f64(double k) const override {
        if constexpr (std::is_same_v<T, double>) return tree_.find(k);
        else { (void)k; return nullptr; }
    }
    bool insert(const Datum& v, const ColumnValue& row) override {
        return tree_.insert(datum_as<T>(v), row);
    }
    bool erase(const Datum& v, const ColumnValue& row) override {
        (void)row;  // a native key already identifies the row
        return tree_.erase(datum_as<T>(v));
    }
    bool is_mutable() const override { return true; }
    bool is_native() const override { return true; }
    std::size_t size() const override { return tree_.size(); }
    std::size_t bytes() const override { return tree_.memory_bytes(); }
    std::size_t max_error() const override { return 0; }
    void validate() const override { tree_.validate(); }

private:
    BPlusTree<T, ColumnValue> tree_;
};

// --- B+ tree, keyed on (value, row id) --------------------------------------
//
// The textbook secondary-index representation, and what makes a duplicated
// column indexable at all: BPlusTree::insert overwrites on an equal key, so
// `category = shoes` for a million rows cannot map value -> row directly.
// Pairing with the row makes every key unique, and std::pair already compares
// lexicographically-then-tiebreak, so btree.hpp needed no change whatsoever.
//
// Every predicate below is exact without reserving any row id as a sentinel.
// The probes use the extreme row values only as *bounds*, and the inclusive /
// exclusive sense is chosen so a real row sitting exactly on one lands on the
// correct side: (v, r) <= (v, MAX_ROW) for every r, and (v, r) >= (v, MIN_ROW)
// for every r.
template <typename T>
class CompositeTree final : public ScalarIndex {
public:
    using Row = ColumnValue;
    using CompositeKey = std::pair<T, Row>;

    explicit CompositeTree(std::size_t order) : tree_(order) {}

    void load(const std::vector<T>& keys, const std::vector<ColumnValue>& values) {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            tree_.insert(CompositeKey{keys[i], values[i]}, values[i]);
        }
    }

    std::vector<ColumnValue> range_query(CompareOp op, const Datum& d) const override {
        const T& v = datum_as<T>(d);
        switch (op) {
            case CompareOp::Eq: return equal_range(v);
            case CompareOp::Lt: return tree_.range_query(CompareOp::Lt, low(v));
            case CompareOp::Le: return tree_.range_query(CompareOp::Le, high(v));
            case CompareOp::Gt: return tree_.range_query(CompareOp::Gt, high(v));
            case CompareOp::Ge: return tree_.range_query(CompareOp::Ge, low(v));
        }
        return {};
    }
    std::vector<ColumnValue> range(const Datum& lo, const Datum& hi) const override {
        return tree_.range(low(datum_as<T>(lo)), high(datum_as<T>(hi)));
    }
    std::vector<ColumnValue> lookup(const Datum& d) const override {
        return equal_range(datum_as<T>(d));
    }
    std::vector<ColumnValue> prefix(const std::string& p) const override {
        if constexpr (std::is_same_v<T, std::string>) {
            std::string upper;
            if (!prefix_upper_bound(p, &upper)) {
                return tree_.range_query(CompareOp::Ge, low(p));
            }
            return tree_.range_half_open(low(p), low(upper));
        } else {
            (void)p;
            throw std::invalid_argument(
                "ColumnIndex::prefix: only string columns have prefixes");
        }
    }
    // A composite column repeats values by construction, so there is no single
    // row to point at. lookup() is the honest answer and callers are steered
    // to it by this returning nullptr rather than by a guess.
    const ColumnValue* find_native(const Datum&) const override { return nullptr; }

    bool insert(const Datum& d, const ColumnValue& row) override {
        return tree_.insert(CompositeKey{datum_as<T>(d), row}, row);
    }
    bool erase(const Datum& d, const ColumnValue& row) override {
        return tree_.erase(CompositeKey{datum_as<T>(d), row});
    }
    bool is_mutable() const override { return true; }
    bool is_native() const override { return false; }
    std::size_t size() const override { return tree_.size(); }
    std::size_t bytes() const override { return tree_.memory_bytes(); }
    std::size_t max_error() const override { return 0; }
    void validate() const override { tree_.validate(); }

private:
    static constexpr Row kMinRow = std::numeric_limits<Row>::min();
    static constexpr Row kMaxRow = std::numeric_limits<Row>::max();

    static CompositeKey low(const T& v) { return CompositeKey{v, kMinRow}; }
    static CompositeKey high(const T& v) { return CompositeKey{v, kMaxRow}; }

    std::vector<ColumnValue> equal_range(const T& v) const {
        return tree_.range(low(v), high(v));
    }

    BPlusTree<CompositeKey, ColumnValue> tree_;
};

// --- Learned index, keyed on the value itself -------------------------------
//
// Build-only, and unique-keys-only. Both are properties of the structure
// rather than of this wrapper: RMIndex::build throws on a repeated key, and
// there is no incremental path at all — that is what DynamicRMIndex exists
// for.
template <typename T>
class NativeRmi final : public ScalarIndex {
public:
    NativeRmi(std::size_t models, std::size_t threshold) : rmi_(models, threshold) {}

    void load(const std::vector<T>& keys, const std::vector<ColumnValue>& values) {
        rmi_.build(keys, values);
    }

    std::vector<ColumnValue> range_query(CompareOp op, const Datum& v) const override {
        return rmi_.range_query(op, datum_as<T>(v));
    }
    std::vector<ColumnValue> range(const Datum& lo, const Datum& hi) const override {
        return rmi_.range(datum_as<T>(lo), datum_as<T>(hi));
    }
    std::vector<ColumnValue> lookup(const Datum& v) const override {
        return one_or_none(rmi_.find(datum_as<T>(v)));
    }
    std::vector<ColumnValue> prefix(const std::string&) const override {
        throw std::invalid_argument(
            "ColumnIndex::prefix: only string columns have prefixes, and a "
            "string column never gets a learned index");
    }
    const ColumnValue* find_native(const Datum& v) const override {
        return rmi_.find(datum_as<T>(v));
    }
    const ColumnValue* find_i64(ColumnKey k) const override {
        if constexpr (std::is_same_v<T, ColumnKey>) return rmi_.find(k);
        else { (void)k; return nullptr; }
    }
    const ColumnValue* find_f64(double k) const override {
        if constexpr (std::is_same_v<T, double>) return rmi_.find(k);
        else { (void)k; return nullptr; }
    }
    bool insert(const Datum&, const ColumnValue&) override {
        throw std::logic_error(
            "ColumnIndex::insert: this column was given a static RMI, which is "
            "build-only. It was chosen under a read-only workload; re-tune with "
            "a non-zero Workload::write_fraction to get a writable structure.");
    }
    bool erase(const Datum&, const ColumnValue&) override {
        throw std::logic_error(
            "ColumnIndex::erase: this column was given a static RMI, which is "
            "build-only. See insert().");
    }
    bool is_mutable() const override { return false; }
    bool is_native() const override { return true; }
    std::size_t size() const override { return rmi_.size(); }
    std::size_t bytes() const override { return rmi_.stats().total_bytes; }
    std::size_t max_error() const override { return rmi_.stats().max_error; }
    void validate() const override { rmi_.validate(); }

private:
    RMIndex<T, ColumnValue> rmi_;
};

// --- Learned index with a delta buffer (module 7) ---------------------------
template <typename T>
class NativeDynamicRmi final : public ScalarIndex {
public:
    using Index = DynamicRMIndex<T, ColumnValue>;

    explicit NativeDynamicRmi(const typename Index::Config& config) : dyn_(config) {}

    void load(const std::vector<T>& keys, const std::vector<ColumnValue>& values) {
        dyn_.build(keys, values);
    }

    std::vector<ColumnValue> range_query(CompareOp op, const Datum& v) const override {
        return dyn_.range_query(op, datum_as<T>(v));
    }
    std::vector<ColumnValue> range(const Datum& lo, const Datum& hi) const override {
        return dyn_.range(datum_as<T>(lo), datum_as<T>(hi));
    }
    std::vector<ColumnValue> lookup(const Datum& v) const override {
        return one_or_none(dyn_.find(datum_as<T>(v)));
    }
    std::vector<ColumnValue> prefix(const std::string&) const override {
        throw std::invalid_argument(
            "ColumnIndex::prefix: only string columns have prefixes, and a "
            "string column never gets a learned index");
    }
    const ColumnValue* find_native(const Datum& v) const override {
        return dyn_.find(datum_as<T>(v));
    }
    const ColumnValue* find_i64(ColumnKey k) const override {
        if constexpr (std::is_same_v<T, ColumnKey>) return dyn_.find(k);
        else { (void)k; return nullptr; }
    }
    const ColumnValue* find_f64(double k) const override {
        if constexpr (std::is_same_v<T, double>) return dyn_.find(k);
        else { (void)k; return nullptr; }
    }
    bool insert(const Datum& v, const ColumnValue& row) override {
        return dyn_.insert(datum_as<T>(v), row);
    }
    bool erase(const Datum& v, const ColumnValue& row) override {
        (void)row;
        return dyn_.erase(datum_as<T>(v));
    }
    bool is_mutable() const override { return true; }
    bool is_native() const override { return true; }
    std::size_t size() const override { return dyn_.size(); }
    std::size_t bytes() const override { return dyn_.stats().index_bytes; }
    std::size_t max_error() const override { return dyn_.stats().max_error; }
    void validate() const override { dyn_.validate(); }

private:
    Index dyn_;
};

// Build the structure a plan describes. The `if constexpr` guards are what
// stop the compiler instantiating a learned index over std::string — the one
// combination that has no meaning.
template <typename T>
std::unique_ptr<ScalarIndex> make_index(const std::vector<T>& keys,
                                        const std::vector<ColumnValue>& values,
                                        const IndexPlan& plan) {
    if (plan.kind == IndexKind::BPlusTree) {
        const std::size_t order = std::max<std::size_t>(plan.btree_order, 3);
        if (plan.encoding == KeyEncoding::Composite) {
            auto out = std::make_unique<CompositeTree<T>>(order);
            out->load(keys, values);
            return out;
        }
        auto out = std::make_unique<NativeTree<T>>(order);
        out->load(keys, values);
        return out;
    }

    if constexpr (kRmiCapable<T>) {
        if (plan.kind == IndexKind::DynamicRMI) {
            typename DynamicRMIndex<T, ColumnValue>::Config config;
            config.second_stage_size = std::max<std::size_t>(plan.rmi_models, 1);
            config.search_threshold = plan.search_threshold;
            config.merge_ratio = plan.merge_ratio > 0.0 ? plan.merge_ratio : 0.05;
            auto out = std::make_unique<NativeDynamicRmi<T>>(config);
            out->load(keys, values);
            return out;
        }
        auto out = std::make_unique<NativeRmi<T>>(
            std::max<std::size_t>(plan.rmi_models, 1), plan.search_threshold);
        out->load(keys, values);
        return out;
    } else {
        throw std::invalid_argument(
            "ColumnIndex: a learned index was asked for over a non-numeric "
            "column. RMIndex fits models to static_cast<double>(key), so this "
            "combination has no meaning; choose_index never produces it.");
    }
}

}  // namespace detail

// One pass over a column already sorted by key, ascending. Free, because
// extraction has to sort anyway.
template <typename T>
inline ColumnShape measure_shape(const std::vector<T>& keys,
                                 const std::vector<ColumnValue>& values) {
    ColumnShape shape;
    shape.rows = keys.size();
    if (keys.empty()) {
        shape.unique = true;
        shape.monotone = true;
        return shape;
    }

    shape.distinct = 1;
    for (std::size_t i = 1; i < keys.size(); ++i) {
        if (keys[i - 1] < keys[i]) ++shape.distinct;
    }
    shape.unique = shape.distinct == keys.size();

    // Ascending row ids in key order means the column arrived in order, which
    // is what makes an append-only write path legal.
    shape.monotone = true;
    for (std::size_t i = 1; i < values.size() && shape.monotone; ++i) {
        if (values[i] < values[i - 1]) shape.monotone = false;
    }
    return shape;
}

// Defined below, once ColumnIndex exists — it has to build candidates to
// measure them. Declared here so ColumnIndex::build can call it.
template <typename T>
inline IndexPlan choose_index_for(LogicalType type, const std::vector<T>& keys,
                                  const std::vector<ColumnValue>& values,
                                  std::size_t size_budget, Workload workload);

inline IndexPlan choose_index(
    const std::vector<ColumnKey>& keys,
    const std::vector<ColumnValue>& values,
    std::size_t size_budget = std::numeric_limits<std::size_t>::max(),
    Workload workload = Workload{});

class ColumnIndex {
public:
    // Auto-tune: build every candidate legal for the type, time it, keep the
    // best.
    template <typename T>
    static ColumnIndex build_typed(
        LogicalType type, const std::vector<T>& keys,
        const std::vector<ColumnValue>& values,
        std::size_t size_budget = std::numeric_limits<std::size_t>::max(),
        Workload workload = Workload{}) {
        return build_typed_with(
            type, keys, values,
            choose_index_for(type, keys, values, size_budget, workload));
    }

    // The int64 entry point every caller before the typed layer used. Kept
    // exactly as it was: an Int64 column, natively keyed.
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
    template <typename T>
    static ColumnIndex build_typed_with(LogicalType type,
                                        const std::vector<T>& keys,
                                        const std::vector<ColumnValue>& values,
                                        const IndexPlan& plan) {
        if (keys.size() != values.size()) {
            throw std::invalid_argument(
                "ColumnIndex: " + std::to_string(keys.size()) + " keys but " +
                std::to_string(values.size()) + " values");
        }

        ColumnIndex out;
        out.plan_ = plan;
        out.plan_.type = type;
        out.plan_.n_keys = keys.size();

        const ColumnShape shape = measure_shape(keys, values);
        out.plan_.distinct = shape.distinct;
        out.plan_.monotone = shape.monotone;

        // The fingerprint is int64 endpoints, so a non-integer column
        // fingerprints on its length and shape alone. That weakens staleness
        // detection for those columns and nothing else: correctness never
        // depends on a plan, because every model and bound is recomputed at
        // build time whichever plan was followed.
        if constexpr (std::is_same_v<T, ColumnKey>) {
            out.plan_.key_min = keys.empty() ? 0 : keys.front();
            out.plan_.key_max = keys.empty() ? 0 : keys.back();
        } else {
            out.plan_.key_min = 0;
            out.plan_.key_max = static_cast<ColumnKey>(shape.distinct);
        }

        out.index_ = detail::make_index(keys, values, out.plan_);
        out.plan_.max_error = out.index_->max_error();
        out.plan_.index_bytes = out.index_->bytes();
        return out;
    }

    static ColumnIndex build_with(const std::vector<ColumnKey>& keys,
                                  const std::vector<ColumnValue>& values,
                                  const IndexPlan& plan) {
        return build_typed_with(plan.type, keys, values, plan);
    }

    // --- typed queries ------------------------------------------------------

    std::vector<ColumnValue> query(CompareOp op, const Datum& value) const {
        return index_->range_query(op, value);
    }
    std::vector<ColumnValue> query_range(const Datum& lo, const Datum& hi) const {
        return index_->range(lo, hi);
    }
    // Every row whose value equals this one. The correct form for a column
    // that repeats values, where find() has no single answer to give.
    std::vector<ColumnValue> lookup(const Datum& value) const {
        return index_->lookup(value);
    }
    // Rows whose value begins with `p`. Exact, byte-order, and a single
    // leaf-chain walk — the capability that would be impossible had the column
    // been encoded to integers to reach a structure it did not need.
    std::vector<ColumnValue> query_prefix(const std::string& p) const {
        return index_->prefix(p);
    }

    // --- int64 conveniences, unchanged from before the typed layer ----------

    const ColumnValue* find(const ColumnKey& key) const {
        return index_->find_i64(key);
    }
    bool contains(const ColumnKey& key) const { return find(key) != nullptr; }

    // The allocation-free point lookups, for the two arithmetic key types.
    // See ScalarIndex::find_i64 for why these exist rather than everything
    // going through a Datum.
    const ColumnValue* find_i64(ColumnKey key) const { return index_->find_i64(key); }
    const ColumnValue* find_f64(double key) const { return index_->find_f64(key); }
    const ColumnValue* find_native(const Datum& key) const {
        return index_->find_native(key);
    }

    std::vector<ColumnValue> range(const ColumnKey& lo, const ColumnKey& hi) const {
        return index_->range(Datum{lo}, Datum{hi});
    }
    std::vector<ColumnValue> range_query(CompareOp op, const ColumnKey& value) const {
        return index_->range_query(op, Datum{value});
    }

    // Writes. Available on every kind except the static RMI, which cannot take
    // them at all — and rather than silently returning false, which would look
    // like "that key was already there", it says so.
    bool is_mutable() const { return index_->is_mutable(); }

    // Whether one key names at most one row. False for a composite encoding,
    // where find() has nothing to point at and lookup() is the answer.
    bool is_native() const { return index_->is_native(); }

    bool insert(const ColumnKey& key, const ColumnValue& value) {
        return index_->insert(Datum{key}, value);
    }
    bool erase(const ColumnKey& key) {
        return erase_row(Datum{key}, 0);
    }

    bool insert_row(const Datum& value, const ColumnValue& row) {
        return index_->insert(value, row);
    }
    bool erase_row(const Datum& value, const ColumnValue& row) {
        if (!index_->is_native() && row == 0) {
            // A composite key is (value, row): without the row there is no key
            // to erase, and picking one arbitrarily would delete a row the
            // caller did not name.
            throw std::invalid_argument(
                "ColumnIndex::erase: this column repeats values, so a value "
                "alone does not identify a row. Pass the row id.");
        }
        return index_->erase(value, row);
    }

    std::size_t size() const { return index_->size(); }
    bool empty() const { return size() == 0; }
    IndexKind kind() const { return plan_.kind; }
    LogicalType type() const { return plan_.type; }
    KeyEncoding encoding() const { return plan_.encoding; }
    const IndexPlan& plan() const { return plan_; }

    void validate() const { index_->validate(); }

private:
    ColumnIndex() = default;

    IndexPlan plan_;
    std::unique_ptr<detail::ScalarIndex> index_;
};

namespace detail {

// A spread of keys to probe with: mostly present, some absent, so a candidate
// cannot look good by being fast only on hits.
template <typename T>
inline std::vector<T> make_probes_typed(const std::vector<T>& keys,
                                        std::size_t count) {
    std::vector<T> probes;
    if (keys.empty()) return probes;
    count = std::min(count, std::max<std::size_t>(keys.size(), 1) * 2);
    probes.reserve(count);

    std::mt19937_64 rng(12345);  // fixed: every candidate sees the same probes
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);
    for (std::size_t i = 0; i < count; ++i) {
        const T& k = keys[pick(rng)];
        if constexpr (std::is_arithmetic_v<T>) {
            probes.push_back((i % 4 == 3) ? static_cast<T>(k + 1) : k);
        } else {
            // A miss for a string is a key with a character appended: still
            // near the real key, so the walk down the tree is the same length.
            probes.push_back((i % 4 == 3) ? k + "\x01" : k);
        }
    }
    return probes;
}

inline std::vector<ColumnKey> make_probes(const std::vector<ColumnKey>& keys,
                                          std::size_t count) {
    return make_probes_typed(keys, count);
}

template <typename T>
inline double time_lookups_typed(const ColumnIndex& index,
                                 const std::vector<T>& probes) {
    if (probes.empty()) return 0.0;

    // A native column can answer through find(), which allocates nothing. A
    // composite one has no single row to point at, so it must go through
    // lookup(). Charging the native path for an allocation it never makes
    // would change every figure this project has already published.
    const bool native = index.is_native();

    volatile std::int64_t sink = 0;
    auto probe_once = [&](const T& k) {
        if (native) {
            const ColumnValue* v = nullptr;
            if constexpr (std::is_same_v<T, ColumnKey>) {
                v = index.find_i64(k);
            } else if constexpr (std::is_same_v<T, double>) {
                v = index.find_f64(k);
            } else {
                v = index.find_native(Datum{k});
            }
            if (v) sink = sink + *v;
            return;
        }
        const std::vector<ColumnValue> rows = index.lookup(Datum{k});
        if (!rows.empty()) sink = sink + rows.front();
    };

    // One untimed pass so every candidate is measured warm; otherwise the
    // first candidate would be charged for the cold caches of all of them.
    for (const T& k : probes) probe_once(k);

    const int reps = probes.size() < 1000 ? 20 : 3;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (const T& k : probes) probe_once(k);
    }
    const auto stop = std::chrono::steady_clock::now();
    (void)sink;

    const double ns = std::chrono::duration<double, std::nano>(stop - start).count();
    return ns / (static_cast<double>(reps) * static_cast<double>(probes.size()));
}

inline double time_lookups(const ColumnIndex& index,
                           const std::vector<ColumnKey>& probes) {
    return time_lookups_typed(index, probes);
}

// Time a mixed write stream: alternating inserts of absent keys and erases of
// present ones, which is the 50/50 split the DynaMind evaluation uses.
//
// Timed on a fresh copy per measurement, because a write stream changes the
// structure it is measured on — a B+ tree splits nodes, a dynamic RMI merges
// — so repeating on one instance would measure a different index each time.
template <typename T>
inline double time_writes_typed(LogicalType type, const std::vector<T>& keys,
                                const std::vector<ColumnValue>& values,
                                const IndexPlan& candidate, std::size_t count) {
    if (keys.empty() || count == 0 || !is_mutable(candidate.kind)) return 0.0;

    ColumnIndex index = ColumnIndex::build_typed_with(type, keys, values, candidate);
    std::mt19937_64 rng(54321);  // fixed: every candidate sees the same stream
    std::uniform_int_distribution<std::size_t> pick(0, keys.size() - 1);

    std::vector<T> inserts;
    std::vector<T> erases;
    std::vector<ColumnValue> erase_rows;
    inserts.reserve(count);
    erases.reserve(count);
    erase_rows.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t at = pick(rng);
        if constexpr (std::is_arithmetic_v<T>) {
            inserts.push_back(static_cast<T>(keys[at] + 1));
        } else {
            inserts.push_back(keys[at] + "\x01");
        }
        erases.push_back(keys[at]);
        erase_rows.push_back(values[at]);
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < count; ++i) {
        index.insert_row(Datum{inserts[i]}, static_cast<ColumnValue>(i));
        index.erase_row(Datum{erases[i]}, erase_rows[i]);
    }
    const auto stop = std::chrono::steady_clock::now();

    const double ns = std::chrono::duration<double, std::nano>(stop - start).count();
    return ns / (2.0 * static_cast<double>(count));
}

inline double time_writes(const std::vector<ColumnKey>& keys,
                          const std::vector<ColumnValue>& values,
                          const IndexPlan& candidate,
                          std::size_t count) {
    return time_writes_typed(LogicalType::Int64, keys, values, candidate, count);
}

}  // namespace detail

// Build one candidate and time it, returning the plan with its measured
// fields filled in.
//
// Exposed rather than kept private because it is the only honest way to time
// a lookup from outside C++: measuring through a language binding would
// charge the index for the bridge, which costs more than the lookup does.
template <typename T>
inline IndexPlan measure_plan_for(LogicalType type, const std::vector<T>& keys,
                                  const std::vector<ColumnValue>& values,
                                  const IndexPlan& candidate,
                                  Workload workload = Workload{}) {
    const ColumnIndex built =
        ColumnIndex::build_typed_with(type, keys, values, candidate);
    IndexPlan measured = built.plan();
    measured.ns_per_lookup =
        detail::time_lookups_typed(built, detail::make_probes_typed(keys, 2000));
    measured.write_fraction = workload.write_fraction;
    if (!workload.read_only()) {
        measured.ns_per_write = detail::time_writes_typed(
            type, keys, values, candidate, std::min<std::size_t>(keys.size(), 2000));
    }
    return measured;
}

inline IndexPlan measure_plan(const std::vector<ColumnKey>& keys,
                              const std::vector<ColumnValue>& values,
                              const IndexPlan& candidate,
                              Workload workload = Workload{}) {
    return measure_plan_for(candidate.type, keys, values, candidate, workload);
}

// What one operation costs on average under this workload. The single number
// the candidates are ranked by, and the reason the write rate has to be an
// input: at 0% writes it is the lookup time, at 100% it is the write time,
// and the crossover in between is exactly the decision being made.
inline double blended_cost(const IndexPlan& plan, const Workload& workload) {
    const double w = workload.write_fraction;
    return (1.0 - w) * plan.ns_per_lookup + w * plan.ns_per_write;
}

// Stage one of the choice: which structures can serve this column *at all*.
//
// A hard filter, applied before anything is built or timed. Without it the
// selector would spend the expensive path — building and timing every
// candidate — on structures that could not have answered a query, and would
// have to reject a learned index over strings by failing rather than by never
// offering it.
inline std::vector<IndexPlan> candidates_for(LogicalType type,
                                             const ColumnShape& shape,
                                             const Workload& workload) {
    std::vector<IndexPlan> out;

    // The tree serves every ordered type. Native keying when the values are
    // unique, composite when they repeat — the row id in the key is what makes
    // a duplicated column indexable at all.
    IndexPlan tree;
    tree.kind = IndexKind::BPlusTree;
    tree.type = type;
    tree.encoding = shape.unique ? KeyEncoding::Native : KeyEncoding::Composite;
    tree.btree_order = 32;
    out.push_back(tree);

    // A learned index needs a metric on the key space (it fits models to
    // static_cast<double>(key)) and needs one key per row. Neither is a
    // preference; both are structural.
    if (!type_supports_rmi(type) || !shape.unique) return out;

    for (std::size_t models : {std::size_t{64}, std::size_t{1024},
                               std::size_t{16384}, std::size_t{262144}}) {
        // Skip model counts that exceed the key count: past that point almost
        // every model is routed nothing, so it is pure overhead.
        if (shape.rows != 0 && models > shape.rows) break;

        // The static RMI is only a legal answer for a read-only column. It is
        // build-only, so offering it to a column that takes writes would be
        // offering something that cannot do the job, however fast it looks on
        // a lookup benchmark.
        if (workload.read_only()) {
            IndexPlan learned;
            learned.kind = IndexKind::RMI;
            learned.type = type;
            learned.encoding = KeyEncoding::Native;
            learned.rmi_models = models;
            learned.search_threshold = 64;
            out.push_back(learned);
        } else {
            for (double ratio : {0.02, 0.05, 0.20}) {
                IndexPlan writable;
                writable.kind = IndexKind::DynamicRMI;
                writable.type = type;
                writable.encoding = KeyEncoding::Native;
                writable.rmi_models = models;
                writable.search_threshold = 64;
                writable.merge_ratio = ratio;
                out.push_back(writable);
            }
        }
    }
    return out;
}

// Build each candidate, time it, and return the plan for the fastest one that
// fits the budget. Ties break toward the smaller index.
template <typename T>
inline IndexPlan choose_index_for(LogicalType type, const std::vector<T>& keys,
                                  const std::vector<ColumnValue>& values,
                                  std::size_t size_budget, Workload workload) {
    const ColumnShape shape = measure_shape(keys, values);
    const std::vector<IndexPlan> candidates = candidates_for(type, shape, workload);

    const std::vector<T> probes = detail::make_probes_typed(keys, 2000);
    const std::size_t write_samples = std::min<std::size_t>(keys.size(), 2000);

    bool have_best = false;
    IndexPlan best;
    double best_cost = 0.0;
    for (const IndexPlan& candidate : candidates) {
        const ColumnIndex built =
            ColumnIndex::build_typed_with(type, keys, values, candidate);
        IndexPlan measured = built.plan();
        measured.ns_per_lookup = detail::time_lookups_typed(built, probes);
        measured.write_fraction = workload.write_fraction;
        if (!workload.read_only()) {
            measured.ns_per_write = detail::time_writes_typed(
                type, keys, values, candidate, write_samples);
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
        best = candidates.front();
        best.n_keys = keys.size();
        best.distinct = shape.distinct;
        best.monotone = shape.monotone;
        if constexpr (std::is_same_v<T, ColumnKey>) {
            best.key_min = keys.empty() ? 0 : keys.front();
            best.key_max = keys.empty() ? 0 : keys.back();
        }
        best.write_fraction = workload.write_fraction;
    }
    return best;
}

inline IndexPlan choose_index(const std::vector<ColumnKey>& keys,
                              const std::vector<ColumnValue>& values,
                              std::size_t size_budget,
                              Workload workload) {
    return choose_index_for(LogicalType::Int64, keys, values, size_budget, workload);
}

}  // namespace hylis::index
