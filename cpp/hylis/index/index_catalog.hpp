// index/index_catalog.hpp
//
// Which index each column got, and why — persisted.
//
// What is stored, and what is not
// -------------------------------
// Only the *plan* is written to disk, never the fitted models. That split is
// deliberate:
//
//   * The models are cheap to reproduce — an RMI rebuilds in ~20ms per
//     million keys — but would need a serialisation format, a version number,
//     and a corruption story. They are recomputed on load instead.
//   * The plan is expensive to reproduce, because producing it means building
//     and timing every candidate structure (see choose_index). That is the
//     part worth caching, and it is a few hundred bytes.
//
// So this file answers "what did we decide, and on what evidence", and the
// answer is replayed rather than re-derived on every startup.
//
// Staleness
// ---------
// Each plan carries a fingerprint of the data it was chosen for. If the
// column has changed, the plan is no longer evidence about it and is
// reported stale so the caller can re-tune. A stale plan is a performance
// bug, never a correctness one: models and error bounds are always recomputed
// at build time, whatever plan was followed.
//
// Durability reuses storage/detail.hpp's atomic_write — the same temp-file,
// fsync, atomic-rename sequence the record store's checkpoint uses. A crash
// mid-save leaves the previous catalog intact.

#pragma once

#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/column_index.hpp"
#include "storage/detail.hpp"
#include "storage/json_detail.hpp"

namespace hylis::index {

namespace catalog_detail {

using namespace hylis::storage::json_detail;

inline std::string to_json(const std::string& column, const IndexPlan& plan) {
    std::string out = "{";
    out += "\"column\":\"" + escape_string(column) + "\",";
    out += "\"kind\":\"" + std::string(to_string(plan.kind)) + "\",";
    // What the column is, and how its values became keys. These are what
    // decided which structures were candidates at all, so a plan without them
    // records the answer but not the question.
    out += "\"type\":\"" + std::string(to_string(plan.type)) + "\",";
    out += "\"encoding\":\"" + std::string(to_string(plan.encoding)) + "\",";
    out += "\"distinct\":" + std::to_string(plan.distinct) + ",";
    out += "\"monotone\":" + std::string(plan.monotone ? "1" : "0") + ",";
    out += "\"rmi_models\":" + std::to_string(plan.rmi_models) + ",";
    out += "\"search_threshold\":" + std::to_string(plan.search_threshold) + ",";
    out += "\"btree_order\":" + std::to_string(plan.btree_order) + ",";
    // Nanoseconds are written as integer picoseconds so the file has no
    // floating-point text to round-trip; sub-picosecond precision is far
    // beyond what the measurement itself is worth.
    out += "\"ps_per_lookup\":" +
           std::to_string(static_cast<std::int64_t>(plan.ns_per_lookup * 1000.0)) + ",";
    out += "\"ps_per_write\":" +
           std::to_string(static_cast<std::int64_t>(plan.ns_per_write * 1000.0)) + ",";
    // Both ratios are permilles for the same reason the timings are
    // picoseconds: the file stays free of floating-point text.
    out += "\"merge_ratio_permille\":" +
           std::to_string(static_cast<std::int64_t>(plan.merge_ratio * 1000.0)) + ",";
    out += "\"write_fraction_permille\":" +
           std::to_string(static_cast<std::int64_t>(plan.write_fraction * 1000.0)) + ",";
    out += "\"max_error\":" + std::to_string(plan.max_error) + ",";
    out += "\"index_bytes\":" + std::to_string(plan.index_bytes) + ",";
    out += "\"n_keys\":" + std::to_string(plan.n_keys) + ",";
    out += "\"key_min\":" + std::to_string(plan.key_min) + ",";
    out += "\"key_max\":" + std::to_string(plan.key_max);
    out += "}";
    return out;
}

// Parses one entry. Unknown fields are skipped rather than rejected, so a
// catalog written by a later build still loads here.
inline std::pair<std::string, IndexPlan> from_json(const char*& p) {
    std::string column;
    IndexPlan plan;

    skip_ws(p);
    expect(p, '{');
    skip_ws(p);
    if (*p == '}') { ++p; return {column, plan}; }

    while (true) {
        skip_ws(p);
        const std::string key = read_string(p);
        skip_ws(p);
        expect(p, ':');
        skip_ws(p);

        if (key == "column") {
            column = read_string(p);
        } else if (key == "kind") {
            plan.kind = index_kind_from_string(read_string(p));
        } else if (key == "type") {
            plan.type = logical_type_from_string(read_string(p));
        } else if (key == "encoding") {
            plan.encoding = key_encoding_from_string(read_string(p));
        } else if (key == "distinct") {
            plan.distinct = static_cast<std::size_t>(read_int(p));
        } else if (key == "monotone") {
            plan.monotone = read_int(p) != 0;
        } else if (key == "rmi_models") {
            plan.rmi_models = static_cast<std::size_t>(read_int(p));
        } else if (key == "search_threshold") {
            plan.search_threshold = static_cast<std::size_t>(read_int(p));
        } else if (key == "btree_order") {
            plan.btree_order = static_cast<std::size_t>(read_int(p));
        } else if (key == "ps_per_lookup") {
            plan.ns_per_lookup = static_cast<double>(read_int(p)) / 1000.0;
        } else if (key == "ps_per_write") {
            plan.ns_per_write = static_cast<double>(read_int(p)) / 1000.0;
        } else if (key == "merge_ratio_permille") {
            plan.merge_ratio = static_cast<double>(read_int(p)) / 1000.0;
        } else if (key == "write_fraction_permille") {
            plan.write_fraction = static_cast<double>(read_int(p)) / 1000.0;
        } else if (key == "max_error") {
            plan.max_error = static_cast<std::size_t>(read_int(p));
        } else if (key == "index_bytes") {
            plan.index_bytes = static_cast<std::size_t>(read_int(p));
        } else if (key == "n_keys") {
            plan.n_keys = static_cast<std::size_t>(read_int(p));
        } else if (key == "key_min") {
            plan.key_min = read_int(p);
        } else if (key == "key_max") {
            plan.key_max = read_int(p);
        } else {
            skip_value(p);
        }

        skip_ws(p);
        if (*p == ',') { ++p; continue; }
        if (*p == '}') { ++p; break; }
        throw std::runtime_error("catalog: expected , or } in entry");
    }
    return {column, plan};
}

}  // namespace catalog_detail

class IndexCatalog {
public:
    // Whether a stored plan can still be trusted as evidence about a column.
    enum class Freshness { Missing, Stale, Fresh };

    void set(const std::string& column, const IndexPlan& plan) {
        plans_[column] = plan;
    }

    std::optional<IndexPlan> get(const std::string& column) const {
        const auto it = plans_.find(column);
        if (it == plans_.end()) return std::nullopt;
        return it->second;
    }

    bool erase(const std::string& column) { return plans_.erase(column) > 0; }
    std::size_t size() const { return plans_.size(); }
    bool empty() const { return plans_.empty(); }
    void clear() { plans_.clear(); }

    std::vector<std::string> columns() const {
        std::vector<std::string> out;
        out.reserve(plans_.size());
        for (const auto& [column, plan] : plans_) out.push_back(column);
        return out;
    }

    Freshness freshness(const std::string& column,
                        const std::vector<ColumnKey>& keys) const {
        const auto it = plans_.find(column);
        if (it == plans_.end()) return Freshness::Missing;
        return it->second.matches(keys) ? Freshness::Fresh : Freshness::Stale;
    }

    // How far a stale plan's performance may degrade before it is worth
    // paying to re-choose.
    //
    // Re-choosing means building and timing every candidate structure, which
    // is the expensive half of this whole subsystem. Any change to the column
    // makes a plan stale, and most changes do not make it *wrong* — so
    // triggering a full re-tune on staleness alone spends the expensive path
    // on columns that were going to reach the same answer anyway.
    //
    // Instead: replay the plan, re-time it, and compare against the figure
    // recorded when it was chosen. Re-tune only if it has actually got worse
    // by this factor. Same measure-don't-model stance choose_index() already
    // takes, applied one level up to the decision to re-decide.
    void set_retune_threshold(double factor) { retune_factor_ = factor; }
    double retune_threshold() const { return retune_factor_; }

    // Why a column got the structure it got, for tests and for reporting.
    enum class Action { Replayed, Retuned, Chosen };

    struct Decision {
        Action action = Action::Chosen;
        Freshness freshness = Freshness::Missing;
        // Only meaningful when the plan was replayed and re-timed.
        double recorded_ns = 0.0;
        double measured_ns = 0.0;
    };

    // Build a column, replaying the stored plan when it still applies and
    // re-tuning when it does not. The catalog is updated with whatever was
    // actually used, so the next run replays the current decision.
    ColumnIndex build_column(const std::string& column,
                             const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values,
                             Workload workload = Workload{}) {
        Decision ignored;
        return build_column(column, keys, values, workload, &ignored);
    }

    ColumnIndex build_column(const std::string& column,
                             const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values,
                             Workload workload,
                             Decision* decision) {
        Decision out;
        out.freshness = freshness(column, keys);
        const std::optional<IndexPlan> stored = get(column);

        // A plan that cannot serve this workload is not a candidate at all,
        // however well it fits the keys — a static RMI cannot take writes.
        const bool usable = stored.has_value() && stored->serves(workload) &&
                            stored->write_fraction == workload.write_fraction;

        if (usable && out.freshness == Freshness::Fresh) {
            out.action = Action::Replayed;
            ColumnIndex index = ColumnIndex::build_with(keys, values, *stored);
            set(column, index.plan());
            if (decision) *decision = out;
            return index;
        }

        if (usable && out.freshness == Freshness::Stale) {
            // Stale, but maybe only technically. Measure before spending.
            const IndexPlan replayed = measure_plan(keys, values, *stored, workload);
            out.recorded_ns = stored->ns_per_lookup;
            out.measured_ns = replayed.ns_per_lookup;

            const bool still_good =
                stored->ns_per_lookup > 0.0 &&
                replayed.ns_per_lookup <= stored->ns_per_lookup * retune_factor_;
            if (still_good) {
                out.action = Action::Replayed;
                ColumnIndex index = ColumnIndex::build_with(keys, values, *stored);
                set(column, index.plan());
                if (decision) *decision = out;
                return index;
            }
            out.action = Action::Retuned;
        }

        ColumnIndex index = ColumnIndex::build(
            keys, values, std::numeric_limits<std::size_t>::max(), workload);
        set(column, index.plan());
        if (decision) *decision = out;
        return index;
    }

    // Freshness for a column whose keys are not int64.
    //
    // The int64 path fingerprints on length plus both endpoints. A string or
    // double column has no endpoints this struct can hold, so it fingerprints
    // on length plus the distinct count instead — weaker, and weaker in a way
    // that costs nothing: a false positive only means a column keeps a
    // now-suboptimal structure, because every model and every error bound is
    // recomputed at build time whichever plan was followed.
    Freshness freshness_typed(const std::string& column, std::size_t n_keys,
                              std::size_t distinct) const {
        const auto it = plans_.find(column);
        if (it == plans_.end()) return Freshness::Missing;
        const bool same = it->second.n_keys == n_keys &&
                          it->second.distinct == distinct;
        return same ? Freshness::Fresh : Freshness::Stale;
    }

    // build_column for a typed column. Same three-way decision — replay,
    // re-time then decide, or re-choose — over a key type the int64 version
    // cannot express.
    template <typename T>
    ColumnIndex build_column_typed(const std::string& column, LogicalType type,
                                   const std::vector<T>& keys,
                                   const std::vector<ColumnValue>& values,
                                   Workload workload = Workload{},
                                   Decision* decision = nullptr,
                                   const std::vector<ColumnValue>* row_space = nullptr) {
        Decision out;
        const ColumnShape shape = measure_shape(keys, values);
        out.freshness = freshness_typed(column, keys.size(), shape.distinct);
        const std::optional<IndexPlan> stored = get(column);

        // A plan for a different type is not a plan for this column at all —
        // it describes a structure that cannot hold these keys.
        //
        // Nor is a plan that assumed uniqueness the data no longer has. Both
        // a learned index and a natively-keyed tree map one key to one row:
        // RMIndex::build throws outright on a repeated key, and BPlusTree
        // silently overwrites, which is worse. Only the composite tree can
        // hold a duplicated column, so when uniqueness is gone every other
        // stored plan stops being a legal answer however well it once fitted.
        //
        // Found by a test that made a unique column non-unique with one
        // insert and then watched the catalog replay the old plan onto it.
        const bool shape_fits =
            shape.unique || (stored.has_value() &&
                             stored->kind == IndexKind::BPlusTree &&
                             stored->encoding == KeyEncoding::Composite);
        const bool usable = stored.has_value() && stored->type == type &&
                            stored->serves(workload) && shape_fits &&
                            stored->write_fraction == workload.write_fraction;

        if (usable && out.freshness == Freshness::Fresh) {
            out.action = Action::Replayed;
            ColumnIndex index = ColumnIndex::build_typed_with(
                type, keys, values, *stored, row_space);
            set(column, index.plan());
            if (decision) *decision = out;
            return index;
        }

        if (usable && out.freshness == Freshness::Stale) {
            // Stale, but maybe only technically. Measure before spending.
            const IndexPlan replayed =
                measure_plan_for(type, keys, values, *stored, workload);
            out.recorded_ns = stored->ns_per_lookup;
            out.measured_ns = replayed.ns_per_lookup;

            const bool still_good =
                stored->ns_per_lookup > 0.0 &&
                replayed.ns_per_lookup <= stored->ns_per_lookup * retune_factor_;
            if (still_good) {
                out.action = Action::Replayed;
                ColumnIndex index = ColumnIndex::build_typed_with(
                    type, keys, values, *stored, row_space);
                set(column, index.plan());
                if (decision) *decision = out;
                return index;
            }
            out.action = Action::Retuned;
        }

        ColumnIndex index = ColumnIndex::build_typed(
            type, keys, values, std::numeric_limits<std::size_t>::max(), workload,
            row_space);
        set(column, index.plan());
        if (decision) *decision = out;
        return index;
    }

    std::string serialize() const {
        std::string out = "{\"version\":1,\"plans\":[";
        bool first = true;
        for (const auto& [column, plan] : plans_) {
            if (!first) out += ",";
            first = false;
            out += catalog_detail::to_json(column, plan);
        }
        out += "]}";
        return out;
    }

    static IndexCatalog parse(const std::string& blob) {
        using namespace hylis::storage::json_detail;
        IndexCatalog out;
        const char* p = blob.c_str();

        skip_ws(p);
        expect(p, '{');
        while (true) {
            skip_ws(p);
            if (*p == '}') { ++p; break; }
            const std::string key = read_string(p);
            skip_ws(p);
            expect(p, ':');
            skip_ws(p);

            if (key == "plans") {
                expect(p, '[');
                skip_ws(p);
                if (*p == ']') { ++p; }
                else {
                    while (true) {
                        auto [column, plan] = catalog_detail::from_json(p);
                        if (!column.empty()) out.plans_[column] = plan;
                        skip_ws(p);
                        if (*p == ',') { ++p; continue; }
                        if (*p == ']') { ++p; break; }
                        throw std::runtime_error("catalog: expected , or ] in plans");
                    }
                }
            } else {
                skip_value(p);
            }

            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("catalog: expected , or } at top level");
        }
        return out;
    }

    void save(const std::string& path) const {
        hylis::storage::detail::atomic_write(path, path + ".tmp", serialize());
    }

    // Returns an empty catalog when the file is absent — a first run is not
    // an error. A file that exists but does not parse *is* an error: silently
    // discarding it would turn a corrupted catalog into a mysterious
    // slowdown, and the caller can always choose to delete and re-tune.
    static IndexCatalog load(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return IndexCatalog{};

        std::string blob;
        char buffer[4096];
        std::size_t got = 0;
        while ((got = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
            blob.append(buffer, got);
        }
        std::fclose(f);

        try {
            return parse(blob);
        } catch (const std::exception& e) {
            throw std::runtime_error("IndexCatalog::load: " + path +
                                     " is corrupt: " + e.what());
        }
    }

private:
    // See set_retune_threshold(). 1.5 is a starting point, not a measured
    // value: it says "half again as slow is worth re-deciding for", and the
    // thing it protects against is paying to rebuild every candidate because
    // a column grew by one row.
    double retune_factor_ = 1.5;
    std::map<std::string, IndexPlan> plans_;
};

}  // namespace hylis::index
