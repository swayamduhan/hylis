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
    out += "\"rmi_models\":" + std::to_string(plan.rmi_models) + ",";
    out += "\"search_threshold\":" + std::to_string(plan.search_threshold) + ",";
    out += "\"btree_order\":" + std::to_string(plan.btree_order) + ",";
    // Nanoseconds are written as integer picoseconds so the file has no
    // floating-point text to round-trip; sub-picosecond precision is far
    // beyond what the measurement itself is worth.
    out += "\"ps_per_lookup\":" +
           std::to_string(static_cast<std::int64_t>(plan.ns_per_lookup * 1000.0)) + ",";
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
        } else if (key == "rmi_models") {
            plan.rmi_models = static_cast<std::size_t>(read_int(p));
        } else if (key == "search_threshold") {
            plan.search_threshold = static_cast<std::size_t>(read_int(p));
        } else if (key == "btree_order") {
            plan.btree_order = static_cast<std::size_t>(read_int(p));
        } else if (key == "ps_per_lookup") {
            plan.ns_per_lookup = static_cast<double>(read_int(p)) / 1000.0;
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

    // Build a column, replaying the stored plan when it still applies and
    // re-tuning when it does not. The catalog is updated with whatever was
    // actually used, so the next run replays the current decision.
    ColumnIndex build_column(const std::string& column,
                             const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values) {
        ColumnIndex index =
            freshness(column, keys) == Freshness::Fresh
                ? ColumnIndex::build_with(keys, values, *get(column))
                : ColumnIndex::build(keys, values);
        set(column, index.plan());
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
    std::map<std::string, IndexPlan> plans_;
};

}  // namespace hylis::index
