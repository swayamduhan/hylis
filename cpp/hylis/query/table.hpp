// query/table.hpp
//
// The join between the record store and the indexes.
//
// What was missing
// ----------------
// Module 1 was complete, durable and tested, and no index had ever read from
// it. Every index in this project was built by handing ColumnIndex::build two
// vectors that came from a test fixture or a benchmark script. So "a hybrid
// database indexing system" was true of the indexes and not of the database:
// there was no path from a stored record to an index over it, and no
// `SELECT ... WHERE` that returned rows.
//
// This is that path. A Table owns nothing except the connection: the records
// belong to the RecordStore, the structures belong to ColumnIndex, the
// decisions belong to IndexCatalog, and the types belong to Schema. What it
// adds is extraction, write-path maintenance, and the reopen story.
//
// Row identity
// ------------
// The record's primary key *is* the row id. Secondary indexes map
// (column value) -> record key, so a predicate's answer is directly a set of
// keys the store can fetch. No separate row numbering, and therefore nothing
// to renumber. (Vector columns do need dense row ids, and that is phase E.)
//
// The write path
// --------------
// Store first, then indexes, and indexes are **never journalled** — they are
// derived state, and a crash between the two is repaired by rebuilding from
// the store, which is what opening a Table already does. That is the same
// split IndexCatalog makes: the *plan* is expensive to reproduce and is
// persisted, the fitted models are cheap and are not.
//
// Maintenance is exact and incremental for every mutable structure. A
// composite key is (value, record key), so changing a row's value is an erase
// of the old pair and an insert of the new one — O(log n), anywhere in the
// key space, with nothing renumbered.
//
// That is a consequence of experiment E3. The design this file was planned
// against used sorted-rank keys, where a mid-range insert renumbers every row
// after the insertion point and the only repair is a full column rebuild. E3
// measured that encoding against composite keys and it lost, so the rebuild
// path it needed is gone with it. What remains of the lazy-rebuild machinery
// serves exactly two cases, both of which are a structure being asked for
// something it cannot do:
//
//   * a column holding a **static RMI**, which is build-only, being written to
//   * a column keyed **natively** (chosen because its values were unique)
//     receiving a write that makes them non-unique
//
// Both mark the column dirty and rebuild it on the next read, and both are
// counted in WriteResult so the cost is reported rather than hidden.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/column_index.hpp"
#include "index/index_catalog.hpp"
#include "index/logical_type.hpp"
#include "query/predicate.hpp"
#include "query/schema.hpp"
#include "storage/detail.hpp"
#include "storage/record.hpp"
#include "storage/store.hpp"

namespace hylis::query {

using hylis::index::ColumnIndex;
using hylis::index::ColumnShape;
using hylis::index::ColumnValue;
using hylis::index::IndexCatalog;
using hylis::index::IndexKind;
using hylis::index::IndexPlan;
using hylis::index::KeyEncoding;
using hylis::index::LogicalType;
using hylis::index::Workload;

// A column, its index, and how the index came to exist.
struct ColumnInfo {
    std::string name;
    LogicalType type = LogicalType::Int64;

    bool indexed = false;
    // Rows carrying a value for this column, and rows that do not. A row
    // without the column is absent from the index and matches no predicate on
    // it, so `skipped` is the size of that set rather than an error count.
    std::size_t rows = 0;
    std::size_t skipped = 0;
    std::size_t distinct = 0;
    bool unique = false;
    bool monotone = false;
    // Whether a write has left the index needing a rebuild before it can be
    // read. Never a correctness hazard: every read path rebuilds first.
    bool dirty = false;

    IndexKind kind = IndexKind::BPlusTree;
    KeyEncoding encoding = KeyEncoding::Native;
    std::size_t index_bytes = 0;
    double ns_per_lookup = 0.0;
};

// What a write cost, so the expensive paths are visible rather than silent.
struct WriteResult {
    // True when the record was new rather than an update. For a batch, see
    // rows_created.
    bool created = false;
    std::size_t rows_written = 0;
    std::size_t rows_created = 0;
    std::size_t indexes_touched = 0;
    // Columns a write forced into a full rebuild. Zero on every workload that
    // does not write to an immutable or natively-keyed column.
    std::size_t rebuilds_triggered = 0;
};

// Why a query executed the way it did. `scanned` is the number that matters:
// it is non-zero exactly when no index could serve the predicate.
struct QueryTrace {
    std::string reason;
    bool used_index = false;
    std::size_t scanned = 0;
    std::size_t matched = 0;
};

class Table {
public:
    static constexpr const char* CATALOG_NAME = "catalog.json";
    static constexpr const char* SCHEMA_NAME = "schema.json";

    // The store is borrowed and must outlive the table. It is the system of
    // record; everything here is derived from it and can be rebuilt from it.
    Table(storage::RecordStore& store, Schema schema)
        : store_(&store), schema_(std::move(schema)) {
        const std::string stored_schema = read_file(path_of(SCHEMA_NAME));
        if (!stored_schema.empty()) {
            reconcile(Schema::parse_json(stored_schema));
        }
        catalog_ = IndexCatalog::load(path_of(CATALOG_NAME));
    }

    // Move-only, and said explicitly rather than left implicit.
    //
    // The columns hold unique_ptr, so a copy could never have compiled — but
    // std::is_copy_constructible<std::map<K, V>> is true whatever V is, so a
    // trait-based caller (pybind11 registering a copy constructor, for one)
    // sees a copyable type and only finds out otherwise deep inside a template
    // instantiation. Deleting it here turns that into an obvious answer.
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    Table(Table&&) noexcept = default;
    Table& operator=(Table&&) noexcept = default;

    // Reopen using the schema on disk. The reason a schema is persisted at
    // all: a stored plan is uninterpretable without it, because
    // "encoding: composite" says nothing unless the key type is known.
    static Table open(storage::RecordStore& store) {
        const std::string blob =
            read_file((store.directory() / SCHEMA_NAME).string());
        if (blob.empty()) {
            throw std::runtime_error(
                "Table::open: no schema at " +
                (store.directory() / SCHEMA_NAME).string() +
                ". A table's columns cannot be inferred from the records, "
                "because a record is a map of strings; pass a Schema instead.");
        }
        return Table(store, Schema::parse_json(blob));
    }

    const Schema& schema() const { return schema_; }
    const storage::RecordStore& store() const { return *store_; }
    const IndexCatalog& catalog() const { return catalog_; }
    std::size_t size() const { return store_->size(); }

    // --- schema and index DDL ----------------------------------------------

    void add_column(ColumnDef column) { schema_.add(std::move(column)); }

    // Build (or re-tune) an index on a column.
    //
    // The workload is an input, not a refinement: on lookups alone the static
    // RMI wins nearly everything, so without a write rate it would be chosen
    // for columns that are about to be written to and cannot take it.
    ColumnInfo create_index(const std::string& name, Workload workload = Workload{}) {
        const ColumnDef& def = schema_.column(name);
        reject_unindexable(def);

        Column& column = columns_[name];
        column.workload = workload;
        build(name, def, &column);
        return info_of(name, def, column);
    }

    bool drop_index(const std::string& name) {
        // The column stays queryable — by scanning, which the trace reports.
        catalog_.erase(name);
        return columns_.erase(name) > 0;
    }

    bool has_index(const std::string& name) const {
        const auto it = columns_.find(name);
        return it != columns_.end() && it->second.index != nullptr;
    }

    std::vector<ColumnInfo> describe() const {
        std::vector<ColumnInfo> out;
        for (const ColumnDef& def : schema_.columns()) {
            const auto it = columns_.find(def.name);
            static const Column empty;
            out.push_back(info_of(def.name, def,
                                  it == columns_.end() ? empty : it->second));
        }
        return out;
    }

    ColumnInfo info(const std::string& name) const {
        const ColumnDef& def = schema_.column(name);
        const auto it = columns_.find(name);
        static const Column empty;
        return info_of(name, def, it == columns_.end() ? empty : it->second);
    }

    // The plan and the evidence behind it: ns/lookup, bytes, error bound.
    IndexPlan explain_column(const std::string& name) const {
        const auto it = columns_.find(name);
        if (it == columns_.end() || !it->second.index) {
            throw std::invalid_argument("Table: column '" + name +
                                        "' has no index to explain");
        }
        return it->second.index->plan();
    }

    // --- DML ----------------------------------------------------------------

    // Insert or update one record. Type-checked against the schema *before*
    // anything is written, so a record that would half-load is refused.
    WriteResult put(const storage::Record& record) {
        schema_.validate(record);

        WriteResult result;
        write_one(record, &result);
        if (!deferring_) result.rebuilds_triggered += flush_dirty();
        return result;
    }

    // Load many records with one index-maintenance pass.
    //
    // Not an optimisation — the only usable bulk-load path. A column that has
    // to fall back to a rebuild would pay for one per record here; batched it
    // pays one at the end, whatever the batch size.
    WriteResult put_batch(const std::vector<storage::Record>& records) {
        for (const storage::Record& r : records) schema_.validate(r);

        WriteResult result;
        // Suspend rebuilds for the duration: a column that goes dirty mid-batch
        // would otherwise be rebuilt, dirtied and rebuilt again per record.
        const bool was_deferring = deferring_;
        deferring_ = true;
        for (const storage::Record& r : records) write_one(r, &result);
        deferring_ = was_deferring;

        if (!deferring_) result.rebuilds_triggered += flush_dirty();
        result.created = result.rows_created > 0;
        return result;
    }

    // Change some columns of an existing record, leaving the rest alone.
    WriteResult update(std::int64_t key, const std::map<std::string, Datum>& changes) {
        const storage::Record* existing = store_->get(key);
        if (existing == nullptr) {
            throw std::invalid_argument("Table::update: no record with key " +
                                        std::to_string(key));
        }
        storage::Record next = *existing;
        for (const auto& [name, value] : changes) {
            next.columns[name] = schema_.format(name, value);
        }
        return put(next);
    }

    WriteResult erase(std::int64_t key) {
        WriteResult result;
        const storage::Record* existing = store_->get(key);
        if (existing == nullptr) return result;

        const storage::Record before = *existing;
        store_->del(key);
        for (auto& [name, column] : columns_) {
            if (!column.index) continue;
            const auto it = before.columns.find(name);
            if (it == before.columns.end()) continue;
            remove_entry(name, column, it->second, key, &result);
        }
        return result;
    }

    std::optional<storage::Record> get(std::int64_t key) const {
        const storage::Record* r = store_->get(key);
        return r ? std::optional<storage::Record>(*r) : std::nullopt;
    }

    // --- queries ------------------------------------------------------------

    // Record keys matching the predicate, **ascending**.
    //
    // Sorted deliberately. An index returns rows in the column's key order,
    // which is not row order, while a scan returns them in the store's hash
    // order — so without this the ordering would depend on which structure
    // happened to answer, and a caller could not tell the two apart. The
    // planner's own matching_rows() stays unsorted because it feeds a filter
    // that does not care; here a person reads the output.
    std::vector<std::int64_t> select_keys(const Predicate& predicate,
                                          QueryTrace* trace = nullptr) const {
        QueryTrace local;
        std::vector<std::int64_t> keys = execute(predicate, &local);
        std::sort(keys.begin(), keys.end());
        local.matched = keys.size();
        if (trace) *trace = local;
        return keys;
    }

    std::vector<storage::Record> select(const Predicate& predicate,
                                        QueryTrace* trace = nullptr) const {
        const std::vector<std::int64_t> keys = select_keys(predicate, trace);
        std::vector<storage::Record> out;
        out.reserve(keys.size());
        for (std::int64_t key : keys) {
            if (const storage::Record* r = store_->get(key)) out.push_back(*r);
        }
        return out;
    }

    // Matching rows, without fetching them.
    //
    // Materialises the key list today, so it saves only the record fetch. A
    // bitmap column answers this by popcount without materialising anything,
    // which is the asymmetry phase C is built on — the signature is here now
    // so that change is invisible to callers.
    std::size_t count(const Predicate& predicate) const {
        return select_keys(predicate).size();
    }

    std::vector<storage::Record> scan(std::size_t limit = 0,
                                      std::size_t offset = 0) const {
        std::vector<storage::Record> all = store_->records();
        std::sort(all.begin(), all.end(),
                  [](const storage::Record& a, const storage::Record& b) {
                      return a.key < b.key;
                  });
        if (offset >= all.size()) return {};
        const std::size_t end =
            limit == 0 ? all.size() : std::min(all.size(), offset + limit);
        return std::vector<storage::Record>(
            all.begin() + static_cast<std::ptrdiff_t>(offset),
            all.begin() + static_cast<std::ptrdiff_t>(end));
    }

    // --- persistence --------------------------------------------------------

    // Both files, atomically. The schema first: a catalog naming an encoding
    // for a column whose type is unknown cannot be interpreted, so the reverse
    // order would leave a readable catalog and an unreadable meaning.
    void save() const {
        storage::detail::atomic_write(path_of(SCHEMA_NAME),
                                      path_of(SCHEMA_NAME) + ".tmp",
                                      schema_.serialize());
        catalog_.save(path_of(CATALOG_NAME));
    }

    void checkpoint() {
        flush_dirty();
        store_->checkpoint();
        save();
    }

    IndexCatalog::Freshness freshness(const std::string& name) const {
        const auto it = columns_.find(name);
        if (it == columns_.end() || !it->second.index) {
            return IndexCatalog::Freshness::Missing;
        }
        const IndexPlan& plan = it->second.index->plan();
        return catalog_.freshness_typed(name, plan.n_keys, plan.distinct);
    }

    // How often a write forced a whole-column rebuild, cumulatively. Zero on
    // every workload that does not write to an immutable or natively-keyed
    // column; see the header note on why that is now the only case left.
    std::size_t rebuilds() const { return rebuilds_; }

    void rebuild(const std::string& name) {
        const ColumnDef& def = schema_.column(name);
        const auto it = columns_.find(name);
        if (it == columns_.end()) return;
        build(name, def, &it->second);
        ++rebuilds_;
    }

    // Every index agrees with the store, for every indexed column.
    //
    // The assertion that makes this layer testable in one call: it spans the
    // store, the indexes and the schema together, where each component's own
    // validate() can only speak for itself.
    void validate() const {
        for (const auto& [name, column] : columns_) {
            if (!column.index) continue;
            ensure_fresh(name);
            column.index->validate();

            const ColumnDef& def = schema_.column(name);
            std::size_t skipped = 0;
            const std::size_t indexed = check_against_store(name, def, column,
                                                            &skipped);
            if (indexed != column.index->size()) {
                throw std::logic_error(
                    "Table::validate: column '" + name + "' has " +
                    std::to_string(column.index->size()) + " index entries but " +
                    std::to_string(indexed) + " rows in the store carry a value");
            }
        }
    }

private:
    struct Column {
        std::unique_ptr<ColumnIndex> index;
        Workload workload;
        // Mutable so a read path can repair a column it finds stale. A dirty
        // index is never read; ensure_fresh() rebuilds first.
        mutable bool dirty = false;
    };

    // Rows carrying a value for a column is the index's own size, and rows
    // without it is the remainder. Both are exact and free, which is why
    // neither is tracked incrementally — a counter maintained across inserts,
    // erases and rebuilds is a second source of truth waiting to drift.
    static std::size_t indexed_rows(const Column& column) {
        return column.index ? column.index->size() : 0;
    }

    // ---- extraction --------------------------------------------------------

    // Every (value, record key) pair for a column, sorted. The pass that
    // sorts is the pass that measures the shape, so uniqueness and
    // monotonicity come free.
    template <typename T>
    std::vector<std::pair<T, ColumnValue>> gather(const std::string& name,
                                                  LogicalType type,
                                                  std::size_t* skipped) const {
        std::vector<std::pair<T, ColumnValue>> pairs;
        pairs.reserve(store_->size());
        *skipped = 0;

        for (const storage::Record& record : store_->records()) {
            const auto it = record.columns.find(name);
            if (it == record.columns.end()) {
                ++*skipped;  // absent, and therefore in no index and no answer
                continue;
            }
            Datum value;
            if (!index::try_parse_datum(type, it->second, &value)) {
                // Unreachable through put(), which type-checks first. A store
                // written before the schema existed can still contain one.
                throw std::invalid_argument(
                    "Table: record " + std::to_string(record.key) + " column '" +
                    name + "' holds '" + it->second + "', which is not " +
                    index::to_string(type));
            }
            pairs.emplace_back(std::get<T>(value),
                               static_cast<ColumnValue>(record.key));
        }
        std::sort(pairs.begin(), pairs.end());
        return pairs;
    }

    template <typename T>
    void build_as(const std::string& name, LogicalType type, Column* column) {
        std::size_t skipped = 0;
        const auto pairs = gather<T>(name, type, &skipped);

        std::vector<T> keys;
        std::vector<ColumnValue> rows;
        keys.reserve(pairs.size());
        rows.reserve(pairs.size());
        for (const auto& [value, row] : pairs) {
            keys.push_back(value);
            rows.push_back(row);
        }

        ColumnIndex built = catalog_.build_column_typed(name, type, keys, rows,
                                                        column->workload);
        column->index = std::make_unique<ColumnIndex>(std::move(built));
        column->dirty = false;
        (void)skipped;  // reported by info_of() from the store, not tracked
    }

    void build(const std::string& name, const ColumnDef& def, Column* column) {
        switch (def.type) {
            case LogicalType::Int64:
            case LogicalType::Timestamp:
                build_as<std::int64_t>(name, def.type, column);
                return;
            case LogicalType::Double:
                build_as<double>(name, def.type, column);
                return;
            case LogicalType::String:
                build_as<std::string>(name, def.type, column);
                return;
            default:
                break;
        }
        reject_unindexable(def);
    }

    static void reject_unindexable(const ColumnDef& def) {
        if (def.type == LogicalType::Bool) {
            throw std::invalid_argument(
                "Table: column '" + def.name + "' is a bool. Two distinct "
                "values means an ordered index over it degenerates to a sorted "
                "list of every row in the table, providing no ordering benefit "
                "at all; it is served by a bitmap instead, which is phase C.");
        }
        if (def.type == LogicalType::Vector) {
            throw std::invalid_argument(
                "Table: column '" + def.name + "' is a vector. Vector columns "
                "are served by the graph indexes, not by a scalar one, and are "
                "stored outside the record — see put_vector(), which is "
                "phase E.");
        }
    }

    // ---- write-path maintenance -------------------------------------------

    // Store first, then indexes. A crash between the two loses index state and
    // nothing else, and opening the table rebuilds it from the store.
    void write_one(const storage::Record& record, WriteResult* result) {
        const storage::Record* existing = store_->get(record.key);
        const bool is_new = existing == nullptr;
        // Copied, not borrowed: store_->put() is about to overwrite the very
        // record this points at, and the old column values are needed after.
        const std::map<std::string, std::string> before =
            is_new ? std::map<std::string, std::string>{} : existing->columns;

        store_->put(record);
        ++result->rows_written;
        result->rows_created += is_new;
        result->created = is_new;

        for (auto& [name, column] : columns_) {
            if (!column.index) continue;

            const auto old_it = before.find(name);
            const bool had = old_it != before.end();
            const auto new_it = record.columns.find(name);
            const bool has = new_it != record.columns.end();
            if (!had && !has) continue;
            // Unchanged text is an unchanged key; touching the index would be
            // an erase and an insert of the same entry.
            if (had && has && old_it->second == new_it->second) continue;

            if (had) remove_entry(name, column, old_it->second, record.key, result);
            if (has) add_entry(name, column, new_it->second, record.key, result);
        }
    }

    void add_entry(const std::string& name, Column& column,
                   const std::string& text, std::int64_t key,
                   WriteResult* result) {
        if (column.dirty) return;  // it is being rebuilt anyway
        if (!column.index->is_mutable()) {
            // A static RMI is build-only. It was a legal choice under a
            // read-only workload and stopped being one the moment this write
            // arrived; the honest repair is to rebuild, not to pretend.
            mark_dirty(column, result);
            return;
        }

        const Datum value = schema_.parse(name, text);
        const bool inserted = column.index->insert_row(value, key);
        ++result->indexes_touched;

        // A natively-keyed column was chosen because its values were unique.
        // An insert that collides has just made them non-unique, and the tree
        // overwrote the colliding row's entry rather than storing both — so
        // the index is wrong until it is rebuilt with a composite key.
        if (!inserted && column.index->is_native()) {
            mark_dirty(column, result);
        }
    }

    void remove_entry(const std::string& name, Column& column,
                      const std::string& text, std::int64_t key,
                      WriteResult* result) {
        if (column.dirty) return;
        if (!column.index->is_mutable()) {
            mark_dirty(column, result);
            return;
        }
        const Datum value = schema_.parse(name, text);
        column.index->erase_row(value, key);
        ++result->indexes_touched;
    }

    // Marking is not counting. The rebuild is counted where it actually
    // happens, in flush_dirty() — counting here as well made a batch of twenty
    // colliding inserts report two rebuilds for the one it performed.
    void mark_dirty(Column& column, WriteResult* result) {
        (void)result;
        column.dirty = true;
    }

    std::size_t flush_dirty() {
        std::size_t rebuilt = 0;
        for (auto& [name, column] : columns_) {
            if (!column.dirty) continue;
            build(name, schema_.column(name), &column);
            ++rebuilds_;
            ++rebuilt;
        }
        return rebuilt;
    }

    // A dirty column is repaired before it is read, so a stale index is never
    // an answer — only a cost.
    void ensure_fresh(const std::string& name) const {
        const auto it = columns_.find(name);
        if (it == columns_.end() || !it->second.dirty || deferring_) return;
        // Lazy repair, the one place this class is logically non-const while
        // syntactically const: a read must never see a stale index, and making
        // every query non-const to say so would be worse.
        Table* self = const_cast<Table*>(this);
        self->build(name, schema_.column(name), &self->columns_[name]);
        ++self->rebuilds_;
    }

    // ---- query execution ---------------------------------------------------

    std::vector<std::int64_t> execute(const Predicate& predicate,
                                      QueryTrace* trace) const {
        const ColumnDef& def = schema_.column(predicate.column);
        if (op_is_string_only(predicate.op) && def.type != LogicalType::String) {
            throw std::invalid_argument(
                std::string("Table: '") + to_string(predicate.op) +
                "' applies to string columns, and '" + predicate.column +
                "' is " + index::to_string(def.type));
        }

        // Two operators no ordered index can serve, and they are served
        // anyway. An infix match has no ordering to exploit; an absent value
        // is absent from the index by construction.
        if (!op_is_indexable(predicate.op)) {
            trace->reason = std::string("'") + to_string(predicate.op) +
                            "' cannot be pushed to an ordered index; scanning " +
                            std::to_string(store_->size()) + " rows";
            trace->scanned = store_->size();
            return scan_for(predicate, def);
        }

        const auto it = columns_.find(predicate.column);
        if (it == columns_.end() || !it->second.index) {
            trace->reason = "column '" + predicate.column +
                            "' has no index; scanning " +
                            std::to_string(store_->size()) + " rows";
            trace->scanned = store_->size();
            return scan_for(predicate, def);
        }

        ensure_fresh(predicate.column);
        const ColumnIndex& index = *it->second.index;
        trace->used_index = true;
        trace->reason = std::string("index on '") + predicate.column + "' (" +
                        index::to_string(index.kind()) + ", " +
                        index::to_string(index.encoding()) + ") answered it";

        switch (predicate.op) {
            case PredOp::Between:
                return index.query_range(predicate.value, predicate.value2);
            case PredOp::Prefix:
                return index.query_prefix(std::get<std::string>(predicate.value));
            default:
                return index.query(compare_op_of(predicate.op), predicate.value);
        }
    }

    // The fallback, and the honest one: correct for every operator, and the
    // trace says how many rows it read.
    std::vector<std::int64_t> scan_for(const Predicate& predicate,
                                       const ColumnDef& def) const {
        std::vector<std::int64_t> out;
        for (const storage::Record& record : store_->records()) {
            const auto it = record.columns.find(predicate.column);
            if (it == record.columns.end()) {
                if (predicate.op == PredOp::IsNull) out.push_back(record.key);
                continue;
            }
            if (predicate.op == PredOp::IsNull) continue;

            if (predicate.op == PredOp::Contains) {
                if (it->second.find(std::get<std::string>(predicate.value)) !=
                    std::string::npos) {
                    out.push_back(record.key);
                }
                continue;
            }

            Datum value;
            if (!index::try_parse_datum(def.type, it->second, &value)) continue;
            if (matches(value, predicate)) out.push_back(record.key);
        }
        return out;
    }

    static bool matches(const Datum& value, const Predicate& p) {
        using index::datum_equal;
        using index::datum_less;
        switch (p.op) {
            case PredOp::Eq: return datum_equal(value, p.value);
            case PredOp::Lt: return datum_less(value, p.value);
            case PredOp::Le: return !datum_less(p.value, value);
            case PredOp::Gt: return datum_less(p.value, value);
            case PredOp::Ge: return !datum_less(value, p.value);
            case PredOp::Between:
                return !datum_less(value, p.value) && !datum_less(p.value2, value);
            case PredOp::Prefix: {
                const std::string& s = std::get<std::string>(value);
                const std::string& prefix = std::get<std::string>(p.value);
                return s.rfind(prefix, 0) == 0;
            }
            default:
                break;
        }
        return false;
    }

    // ---- validation helpers ------------------------------------------------

    std::size_t check_against_store(const std::string& name, const ColumnDef& def,
                                    const Column& column,
                                    std::size_t* skipped) const {
        std::size_t indexed = 0;
        for (const storage::Record& record : store_->records()) {
            const auto it = record.columns.find(name);
            if (it == record.columns.end()) {
                ++*skipped;
                continue;
            }
            ++indexed;
            const Datum value = index::parse_datum(def.type, it->second);
            const std::vector<ColumnValue> rows = column.index->lookup(value);
            if (std::find(rows.begin(), rows.end(),
                          static_cast<ColumnValue>(record.key)) == rows.end()) {
                throw std::logic_error(
                    "Table::validate: record " + std::to_string(record.key) +
                    " has column '" + name + "' = '" + it->second +
                    "' but the index does not return it for that value");
            }
        }
        return indexed;
    }

    ColumnInfo info_of(const std::string& name, const ColumnDef& def,
                       const Column& column) const {
        ColumnInfo out;
        out.name = name;
        out.type = def.type;
        out.dirty = column.dirty;
        out.rows = indexed_rows(column);
        out.skipped = store_->size() > out.rows ? store_->size() - out.rows : 0;
        if (!column.index) return out;

        const IndexPlan& plan = column.index->plan();
        out.indexed = true;
        out.distinct = plan.distinct;
        out.unique = plan.distinct == plan.n_keys;
        out.monotone = plan.monotone;
        out.kind = plan.kind;
        out.encoding = plan.encoding;
        out.index_bytes = plan.index_bytes;
        out.ns_per_lookup = plan.ns_per_lookup;
        return out;
    }

    // ---- reopen ------------------------------------------------------------

    // A stored schema must still describe this table. Adding a column is fine
    // (that is the one evolution supported); changing a column's type is not,
    // because every index and every stored plan over it would be describing a
    // structure that cannot hold the new keys.
    void reconcile(const Schema& stored) const {
        for (const ColumnDef& was : stored.columns()) {
            if (!schema_.has(was.name)) {
                throw std::runtime_error(
                    "Table: the stored schema has column '" + was.name +
                    "', which the schema passed in does not declare. Dropping "
                    "a column is not supported.");
            }
            const ColumnDef& now = schema_.column(was.name);
            if (now.type != was.type) {
                throw std::runtime_error(
                    "Table: column '" + was.name + "' was stored as " +
                    index::to_string(was.type) + " and is now declared " +
                    index::to_string(now.type) + ". Retyping a column is not "
                    "supported: every index and stored plan over it describes "
                    "a structure that cannot hold the new keys.");
            }
        }
    }

    std::string path_of(const char* name) const {
        return (store_->directory() / name).string();
    }

    static std::string read_file(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return {};
        std::string blob;
        char buffer[4096];
        std::size_t got = 0;
        while ((got = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
            blob.append(buffer, got);
        }
        std::fclose(f);
        return blob;
    }

    storage::RecordStore* store_;
    Schema schema_;
    IndexCatalog catalog_;
    std::map<std::string, Column> columns_;
    std::size_t rebuilds_ = 0;
    bool deferring_ = false;
};

}  // namespace hylis::query
