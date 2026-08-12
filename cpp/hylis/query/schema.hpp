// query/schema.hpp
//
// What a table's columns are, and the enforcement that makes that a schema
// rather than a convention.
//
// Why this sits above both storage/ and index/
// --------------------------------------------
// A schema needs the record store (it validates records) and the index layer
// (it names logical types). index/ already depends on storage/ — see
// index_catalog.hpp — so query/ is the only layer above both. That is also
// where the planner lives, which is right: a schema is a query-layer concept.
// The record store stays a thin, untyped key/value layer and never learns what
// a column means.
//
// What enforcement buys
// ---------------------
// RecordStore holds map<string, string>. Without a schema, `pirce = "40"` is a
// new column and `price = "abc"` is a value; both load, and the mistake shows
// up much later as an index that is missing rows nobody can account for. With
// one, both are rejected at put() with the column named.
//
// So two rules, and they are asymmetric on purpose:
//
//   * An **unknown column is an error.** Catching the typo is the whole point,
//     and a column you want to store but not index costs one line to declare.
//   * A **missing column is not.** A row simply is not in that column's index
//     and matches no predicate on it — which is close enough to SQL's
//     `NULL < 5 -> unknown`, and is stated rather than assumed.

#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/logical_type.hpp"
#include "storage/json_detail.hpp"
#include "storage/record.hpp"

namespace hylis::query {

using hylis::index::Datum;
using hylis::index::LogicalType;

struct ColumnDef {
    std::string name;
    LogicalType type = LogicalType::Int64;

    // Vector columns only. Zero everywhere else.
    std::size_t dim = 0;

    ColumnDef() = default;
    ColumnDef(std::string n, LogicalType t, std::size_t d = 0)
        : name(std::move(n)), type(t), dim(d) {}
};

class Schema {
public:
    Schema() = default;

    explicit Schema(std::vector<ColumnDef> columns) {
        for (auto& c : columns) add(std::move(c));
    }

    // Adding is the only schema evolution supported. Dropping or retyping a
    // column would invalidate every index and every stored plan over it, and
    // the migration story that needs is out of proportion to what this project
    // measures.
    void add(ColumnDef column) {
        if (column.name.empty()) {
            throw std::invalid_argument("Schema: a column must have a name");
        }
        if (has(column.name)) {
            throw std::invalid_argument("Schema: column '" + column.name +
                                        "' is already declared");
        }
        if (column.type == LogicalType::Vector && column.dim == 0) {
            throw std::invalid_argument("Schema: vector column '" + column.name +
                                        "' needs a dimension");
        }
        if (column.type != LogicalType::Vector && column.dim != 0) {
            throw std::invalid_argument("Schema: only a vector column has a "
                                        "dimension; '" + column.name + "' is " +
                                        index::to_string(column.type));
        }
        index_[column.name] = columns_.size();
        columns_.push_back(std::move(column));
    }

    bool has(const std::string& name) const { return index_.count(name) != 0; }
    std::size_t size() const { return columns_.size(); }
    bool empty() const { return columns_.empty(); }
    const std::vector<ColumnDef>& columns() const { return columns_; }

    const ColumnDef& column(const std::string& name) const {
        const auto it = index_.find(name);
        if (it == index_.end()) {
            throw std::invalid_argument("Schema: no column named '" + name +
                                        "'. Declared: " + joined());
        }
        return columns_[it->second];
    }

    LogicalType type_of(const std::string& name) const {
        return column(name).type;
    }

    // Columns an ordered or bitmap index can be built over. Vector columns are
    // served by the graph indexes and never appear as a scalar predicate.
    std::vector<std::string> scalar_columns() const {
        std::vector<std::string> out;
        for (const ColumnDef& c : columns_) {
            if (c.type != LogicalType::Vector) out.push_back(c.name);
        }
        return out;
    }

    std::vector<std::string> vector_columns() const {
        std::vector<std::string> out;
        for (const ColumnDef& c : columns_) {
            if (c.type == LogicalType::Vector) out.push_back(c.name);
        }
        return out;
    }

    // --- values -------------------------------------------------------------

    Datum parse(const std::string& name, const std::string& text) const {
        const ColumnDef& def = column(name);
        Datum out;
        if (!index::try_parse_datum(def.type, text, &out)) {
            throw std::invalid_argument(
                "column '" + name + "' is " + index::to_string(def.type) +
                ", but the value '" + text + "' does not parse as one");
        }
        return out;
    }

    std::string format(const std::string& name, const Datum& value) const {
        return index::format_datum(column(name).type, value);
    }

    // --- enforcement --------------------------------------------------------

    // Check a record against the schema before anything is written. Throws on
    // the first problem, naming the column — a record that half-loads is worse
    // than one that is refused.
    void validate(const storage::Record& record) const {
        for (const auto& [name, text] : record.columns) {
            const auto it = index_.find(name);
            if (it == index_.end()) {
                throw std::invalid_argument(
                    "record " + std::to_string(record.key) + ": column '" +
                    name + "' is not in the schema. Declared: " + joined());
            }
            const ColumnDef& def = columns_[it->second];
            if (def.type == LogicalType::Vector) {
                throw std::invalid_argument(
                    "record " + std::to_string(record.key) + ": column '" +
                    name + "' is a vector. Embeddings are not stored in the "
                    "record — a 128-float vector is ~700 bytes of base64 per "
                    "row and would make the write-ahead log the dominant cost "
                    "of the system. Use put_vector() instead.");
            }
            Datum ignored;
            if (!index::try_parse_datum(def.type, text, &ignored)) {
                throw std::invalid_argument(
                    "record " + std::to_string(record.key) + ": column '" +
                    name + "' is " + index::to_string(def.type) +
                    ", but the value '" + text + "' does not parse as one");
            }
        }
    }

    // Whether a record would be accepted, without the message. For callers
    // counting rejects across a bulk load rather than aborting on the first.
    bool accepts(const storage::Record& record) const {
        try {
            validate(record);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
    }

    // --- persistence --------------------------------------------------------
    //
    // The schema is written beside the catalog, because a stored plan is
    // uninterpretable without it: "encoding: composite" says nothing unless
    // the key type is known.

    std::string serialize() const {
        using hylis::storage::json_detail::escape_string;
        std::string out = "{\"version\":1,\"columns\":[";
        bool first = true;
        for (const ColumnDef& c : columns_) {
            if (!first) out += ",";
            first = false;
            out += "{\"name\":\"" + escape_string(c.name) + "\",";
            out += "\"type\":\"" + std::string(index::to_string(c.type)) + "\",";
            out += "\"dim\":" + std::to_string(c.dim) + "}";
        }
        out += "]}";
        return out;
    }

    static Schema parse_json(const std::string& blob) {
        using namespace hylis::storage::json_detail;
        Schema out;
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

            if (key == "columns") {
                expect(p, '[');
                skip_ws(p);
                if (*p == ']') { ++p; }
                else {
                    while (true) {
                        out.add(parse_column(p));
                        skip_ws(p);
                        if (*p == ',') { ++p; continue; }
                        if (*p == ']') { ++p; break; }
                        throw std::runtime_error("schema: expected , or ] in columns");
                    }
                }
            } else {
                skip_value(p);
            }

            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("schema: expected , or } at top level");
        }
        return out;
    }

private:
    static ColumnDef parse_column(const char*& p) {
        using namespace hylis::storage::json_detail;
        ColumnDef def;
        skip_ws(p);
        expect(p, '{');
        skip_ws(p);
        if (*p == '}') { ++p; return def; }

        while (true) {
            skip_ws(p);
            const std::string key = read_string(p);
            skip_ws(p);
            expect(p, ':');
            skip_ws(p);
            if (key == "name") {
                def.name = read_string(p);
            } else if (key == "type") {
                def.type = index::logical_type_from_string(read_string(p));
            } else if (key == "dim") {
                def.dim = static_cast<std::size_t>(read_int(p));
            } else {
                skip_value(p);
            }
            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("schema: expected , or } in a column");
        }
        return def;
    }

    std::string joined() const {
        std::string out;
        for (const ColumnDef& c : columns_) {
            if (!out.empty()) out += ", ";
            out += c.name;
            out += ":";
            out += index::to_string(c.type);
        }
        return out.empty() ? "(none)" : out;
    }

    std::vector<ColumnDef> columns_;
    std::map<std::string, std::size_t> index_;
};

}  // namespace hylis::query
