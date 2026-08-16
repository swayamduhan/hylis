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
// The record's primary key *is* the row id, for every scalar index. They map
// (column value) -> record key, so a predicate's answer is directly a set of
// keys the store can fetch. No separate row numbering, and therefore nothing
// to renumber.
//
// Vector columns cannot work that way: their ids are dense positions into a
// float buffer, because the buffer is what makes a sequential sweep fast. So a
// vector column carries its own key <-> row translation, and a hybrid query is
// the place the two spaces meet — see query/vector_column.hpp, which owns that
// join and states what it costs.
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
// serves three cases, each a structure being asked for something it cannot do:
//
//   * a column holding a **static RMI**, which is build-only, being written to
//   * a column keyed **natively** (chosen because its values were unique)
//     receiving a write that makes them non-unique
//   * a **bitmap** column receiving a row whose key lands mid-table, which
//     would shift every position after it and invalidate every bitmap at once
//
// All three mark the column dirty and rebuild it on the next read, and all
// three are counted in WriteResult so the cost is reported rather than hidden.
// None of them is detected here: ScalarIndex::insert returns false when it
// cannot represent the result, and that one answer covers all three.

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
#include "query/planner.hpp"
#include "query/predicate.hpp"
#include "query/schema.hpp"
#include "query/vector_column.hpp"
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
using hylis::index::Metric;
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

// A vector column, and what it currently holds.
struct VectorInfo {
    std::string name;
    std::size_t dim = 0;
    bool indexed = false;

    // Rows with a live embedding, and rows deletion could not reclaim. The
    // second is the visible price of HNSW having no delete: it is space, and
    // it is every search running through the masked path.
    std::size_t rows = 0;
    std::size_t orphans = 0;

    VectorStructure structure = VectorStructure::Exact;
    Metric metric = Metric::L2;
    bool has_graph = false;
    // Whether row id i is record key i, which is what makes a table bitmap
    // usable as a vector mask.
    bool rows_are_keys = false;
    std::size_t memory_bytes = 0;
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

// What a hybrid query did, beyond the plan itself.
struct HybridTrace {
    QueryTrace structured;
    QueryPlan plan;
    // Rows the predicate matched that carry no embedding. Dropped, because a
    // row with no vector cannot be a nearest neighbour — and counted, because
    // returning fewer rows than the predicate matched is exactly the kind of
    // thing that must not be silent.
    std::size_t without_vector = 0;
    // Whether the structured half was answered by a popcount, so no row id was
    // materialised before the vector search began.
    bool mask_used = false;
};

class Table {
public:
    static constexpr const char* CATALOG_NAME = "catalog.json";
    static constexpr const char* SCHEMA_NAME = "schema.json";
    static constexpr const char* VECTORS_NAME = "vectors.json";

    // The store is borrowed and must outlive the table. It is the system of
    // record; everything here is derived from it and can be rebuilt from it.
    Table(storage::RecordStore& store, Schema schema)
        : store_(&store), schema_(std::move(schema)) {
        const std::string stored_schema = read_file(path_of(SCHEMA_NAME));
        if (!stored_schema.empty()) {
            reconcile(Schema::parse_json(stored_schema));
        }
        catalog_ = IndexCatalog::load(path_of(CATALOG_NAME));
        load_vectors();
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

    // Build a named structure, skipping the measurement that would have
    // chosen one.
    //
    // Exists because choose_index times lookups and writes and nothing else,
    // so a column whose workload is dominated by count() or by conjunctions is
    // judged on the one thing the bitmap is worst at — it decodes a whole
    // matching set where a tree walks a leaf run. An experiment comparing the
    // families therefore cannot get a bitmap by asking nicely, and neither can
    // a user who knows their query mix better than a lookup benchmark does.
    //
    // The plan is still measured after the fact, so info() reports what this
    // structure actually costs rather than what it was hoped to.
    ColumnInfo create_index_as(const std::string& name, IndexKind kind,
                               Workload workload = Workload{}) {
        const ColumnDef& def = schema_.column(name);
        reject_unindexable(def);

        Column& column = columns_[name];
        column.workload = workload;
        column.forced = kind;
        column.force = true;
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

    // --- vector DDL ---------------------------------------------------------

    // Build (or re-tune) the structures behind a vector column.
    //
    // Not chosen by measurement, unlike every scalar column, and the asymmetry
    // is deliberate: choose_index times lookups, while the question for a
    // vector index is recall at a target — which needs a query workload and a
    // ground truth a table does not have. Naming the structure is the honest
    // interface; the recall measurement stays in scripts/bench_vector.py,
    // against the exact index this column keeps for exactly that purpose.
    VectorInfo create_vector_index(const std::string& name,
                                   VectorPlan plan = VectorPlan{}) {
        const ColumnDef& def = schema_.column(name);
        if (def.type != LogicalType::Vector) {
            throw std::invalid_argument(
                "Table: column '" + name + "' is " +
                index::to_string(def.type) +
                ", not a vector. Scalar columns are served by create_index().");
        }
        const auto it = vectors_.find(name);
        if (it == vectors_.end()) {
            vectors_.emplace(name, VectorColumn(name, def.dim, plan));
        } else {
            // Keeps the embeddings: re-tuning M or ef_construction is a
            // legitimate operation, and losing the corpus to perform it would
            // make it unusable on any table that already holds one.
            it->second.retune(plan);
        }
        return vector_info_of(name, def, &vectors_.at(name));
    }

    bool drop_vector_index(const std::string& name) {
        return vectors_.erase(name) > 0;
    }

    bool has_vector_index(const std::string& name) const {
        return vectors_.find(name) != vectors_.end();
    }

    const VectorColumn& vector_column(const std::string& name) const {
        return *require_vector(name);
    }

    std::vector<VectorInfo> describe_vectors() const {
        std::vector<VectorInfo> out;
        for (const ColumnDef& def : schema_.columns()) {
            if (def.type != LogicalType::Vector) continue;
            const auto it = vectors_.find(def.name);
            out.push_back(vector_info_of(
                def.name, def, it == vectors_.end() ? nullptr : &it->second));
        }
        return out;
    }

    VectorInfo vector_info(const std::string& name) const {
        const ColumnDef& def = schema_.column(name);
        const auto it = vectors_.find(name);
        return vector_info_of(name, def,
                              it == vectors_.end() ? nullptr : &it->second);
    }

    // --- DML ----------------------------------------------------------------

    // Attach an embedding to a record. Returns the row it occupies.
    //
    // The record must already exist. A vector whose record does not is a row
    // no query can ever return — knn() answers in record keys, and the caller's
    // next move is to fetch them — so accepting one would only defer the
    // failure to a point where it is much harder to attribute.
    //
    // **Not write-ahead logged.** The floats go to the vector index and to
    // nothing else until save_vectors(); a crash before that loses every
    // embedding attached since the last one, while the records themselves
    // survive. That is the cost of keeping a 128-float vector out of a JSON
    // WAL, it is stated here rather than discovered, and save_vectors() is the
    // durability point a caller can take without a full checkpoint.
    std::int64_t put_vector(std::int64_t key, const std::string& column,
                            const std::vector<float>& vector) {
        VectorColumn* vec = require_vector(column);
        if (store_->get(key) == nullptr) {
            throw std::invalid_argument(
                "Table::put_vector: no record with key " + std::to_string(key) +
                ". An embedding belongs to a row; put() the record first.");
        }
        return vec->put(key, vector);
    }

    // Attach many at once. `data` is row-major, dim floats per key.
    std::size_t put_vectors(const std::string& column,
                            const std::vector<std::int64_t>& keys,
                            const std::vector<float>& data) {
        VectorColumn* vec = require_vector(column);
        if (keys.size() * vec->dim() != data.size()) {
            throw std::invalid_argument(
                "Table::put_vectors: " + std::to_string(keys.size()) +
                " keys against " + std::to_string(data.size()) +
                " floats at " + std::to_string(vec->dim()) + " dimensions");
        }
        for (std::int64_t key : keys) {
            if (store_->get(key) == nullptr) {
                throw std::invalid_argument(
                    "Table::put_vectors: no record with key " +
                    std::to_string(key));
            }
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            vec->put(keys[i], data.data() + i * vec->dim());
        }
        return keys.size();
    }

    bool erase_vector(std::int64_t key, const std::string& column) {
        return require_vector(column)->erase(key);
    }

    bool has_vector(std::int64_t key, const std::string& column) const {
        return require_vector(column)->contains(key);
    }

    // The stored embedding, or an empty vector when the row has none. For
    // Cosine this is the normalised form, which is what the index compares
    // against and therefore what a caller reasoning about a score needs.
    std::vector<float> get_vector(std::int64_t key,
                                  const std::string& column) const {
        const VectorColumn* vec = require_vector(column);
        const float* v = vec->vector_of(key);
        if (v == nullptr) return {};
        return std::vector<float>(v, v + vec->dim());
    }

    // Every record key carrying an embedding, in row order.
    std::vector<std::int64_t> vector_keys(const std::string& column) const {
        return require_vector(column)->keys();
    }

    // --- vector queries -----------------------------------------------------

    // k nearest, best first, in record keys.
    //
    // `exact` forces the brute-force scan. Selectable rather than a fallback:
    // it is the only exact answer here, and it is what every approximate
    // result is graded against.
    std::vector<VectorMatch> knn(const std::string& column,
                                 const std::vector<float>& query,
                                 std::size_t k, std::size_t ef = 0,
                                 bool exact = false) const {
        return require_vector(column)->knn(query.data(), k, ef, exact);
    }

    // More-like-this, seeded by a row already in the table. The seed is
    // excluded: a row is always its own nearest neighbour, so returning it
    // would spend one of the k on something the caller already has.
    std::vector<VectorMatch> knn_by_key(const std::string& column,
                                        std::int64_t key, std::size_t k,
                                        std::size_t ef = 0,
                                        bool exact = false) const {
        return require_vector(column)->knn_by_key(key, k, ef, exact);
    }

    // The query the whole project is for: k nearest **among rows satisfying
    // the predicates**.
    //
    // Every plan returns the same rows; the planner chooses between costs,
    // never between answers. Two routes in:
    //
    //   * when every predicate is served by a bitmap column whose positions
    //     are this vector column's row ids, the structured half is a
    //     word-parallel AND and a popcount, and **no row id is materialised at
    //     all** before the search starts;
    //   * otherwise the predicates resolve to record keys, which are then
    //     translated to rows — the one place the two id spaces meet, and the
    //     cost the key/row split buys back in exchange for not renumbering
    //     anything on a scalar write.
    //
    // The planner is built here and thrown away, holding only the two vector
    // indexes. It deliberately gets no columns: Table already owns them, and
    // HybridPlanner::set_column copies, so attaching them would keep a second
    // copy of every index alive for the lifetime of the table.
    std::vector<VectorMatch> hybrid(const std::vector<Predicate>& predicates,
                                    const std::string& column,
                                    const std::vector<float>& query,
                                    std::size_t k, std::size_t ef = 0,
                                    HybridTrace* out = nullptr) const {
        const VectorColumn* vec = require_vector(column);
        HybridTrace trace;

        if (predicates.empty()) {
            // No structured half at all, so this degenerates to a similarity
            // search and the plan says so rather than inventing a filter.
            trace.plan.kind = PlanKind::NoPredicate;
            trace.plan.corpus_rows = vec->size();
            trace.plan.matched_rows = vec->size();
            trace.plan.selectivity = 1.0;
            trace.plan.reason = "no predicate; an unconstrained similarity search";
            trace.structured.reason = trace.plan.reason;
            if (out) *out = trace;
            return vec->knn(query.data(), k, ef);
        }

        const HybridPlanner planner = planner_for(*vec);

        Bitset mask;
        if (vector_mask(predicates, *vec, &mask, &trace.structured)) {
            trace.mask_used = true;
            std::vector<Neighbor> raw =
                planner.search_mask(mask, query.data(), k, ef, &trace.plan);
            trace.structured.matched = trace.plan.matched_rows;
            if (out) *out = trace;
            return vec->present(raw);
        }

        const std::vector<std::int64_t> keys =
            select_all(predicates, &trace.structured);
        const std::vector<ColumnValue> rows =
            vec->rows_for(keys, &trace.without_vector);
        std::vector<Neighbor> raw =
            planner.search_rows(rows, query.data(), k, ef, &trace.plan);
        if (out) *out = trace;
        return vec->present(raw);
    }

    // Decide, without doing any vector work.
    //
    // The command a demo runs to show the planner reasoning, and the thing E5
    // measured: on a bitmap column aligned to the vector rows this costs a
    // popcount, while on any other column it costs the whole structured query,
    // because executing the predicate *is* how this planner knows its
    // selectivity. That asymmetry is the point, and it is why the trace says
    // which one happened.
    HybridTrace explain_hybrid(const std::vector<Predicate>& predicates,
                               const std::string& column, std::size_t k) const {
        const VectorColumn* vec = require_vector(column);
        HybridTrace trace;
        const HybridPlanner planner = planner_for(*vec);

        if (predicates.empty()) {
            trace.plan.kind = PlanKind::NoPredicate;
            trace.plan.corpus_rows = vec->size();
            trace.plan.matched_rows = vec->size();
            trace.plan.selectivity = 1.0;
            trace.plan.reason = "no predicate; an unconstrained similarity search";
            return trace;
        }

        Bitset mask;
        if (vector_mask(predicates, *vec, &mask, &trace.structured)) {
            trace.mask_used = true;
            trace.plan = planner.explain_rows(mask.count(), k, true);
            trace.plan.selectivity_was_free = true;
            trace.structured.matched = trace.plan.matched_rows;
            return trace;
        }

        const std::vector<std::int64_t> keys =
            select_all(predicates, &trace.structured);
        const std::vector<ColumnValue> rows =
            vec->rows_for(keys, &trace.without_vector);
        trace.plan = planner.explain_rows(rows.size(), k, false);
        return trace;
    }

    // Measure this corpus's own pre-filter crossover and adopt it.
    //
    // The 50% default came from one measurement on one corpus at one ef, and
    // the crossover is not a constant of the algorithm: it moves with n, with
    // the dimensionality, with ef, and with how fast this machine's cache is.
    // Same answer choose_index() gives one level down — build both, time both,
    // keep what won.
    //
    // `queries` is row-major, dim floats each. No scalar column is involved:
    // the synthetic filters are cut from the vector column's own rows, so the
    // crossover measured is a property of the corpus and the machine rather
    // than of whichever attribute happened to be indexed.
    double calibrate(const std::string& column,
                     const std::vector<float>& queries, std::size_t k = 10,
                     std::size_t ef = 0, std::size_t samples = 12) {
        const VectorColumn* vec = require_vector(column);
        if (vec->dim() == 0 || queries.size() % vec->dim() != 0) {
            throw std::invalid_argument(
                "Table::calibrate: " + std::to_string(queries.size()) +
                " floats is not a whole number of " +
                std::to_string(vec->dim()) + "-dimensional queries");
        }
        HybridPlanner planner = planner_for(*vec);
        if (!vec->has_graph()) return prefilter_threshold_;

        std::vector<ColumnValue> rows;
        rows.reserve(vec->size());
        vec->live().for_each([&](std::size_t row) {
            rows.push_back(static_cast<ColumnValue>(row));
        });

        prefilter_threshold_ = planner.calibrate_rows(
            rows, queries.data(), queries.size() / vec->dim(), k, ef, samples);
        return prefilter_threshold_;
    }

    // Force a plan. Exists so a test can assert that every legal plan returns
    // the same rows — without it the planner could pick a fast *wrong* plan and
    // nothing would notice, which is the one failure mode a query optimiser
    // must not have.
    std::vector<VectorMatch> hybrid_with(PlanKind kind,
                                         const std::vector<Predicate>& predicates,
                                         const std::string& column,
                                         const std::vector<float>& query,
                                         std::size_t k, std::size_t ef = 0,
                                         HybridTrace* out = nullptr) const {
        const VectorColumn* vec = require_vector(column);
        HybridTrace trace;
        trace.plan.kind = kind;

        const std::vector<std::int64_t> keys =
            select_all(predicates, &trace.structured);
        const std::vector<ColumnValue> rows =
            vec->rows_for(keys, &trace.without_vector);
        trace.plan.corpus_rows = vec->rows();
        trace.plan.matched_rows = rows.size();
        trace.plan.reason = std::string("forced: ") + to_string(kind);

        Bitset mask;
        if (kind == PlanKind::BitmapFilteredGraph) {
            QueryTrace ignored;
            if (!vector_mask(predicates, *vec, &mask, &ignored)) {
                throw std::invalid_argument(
                    "Table::hybrid_with: the bitmap-filtered-graph plan needs "
                    "every predicate served by a bitmap column whose positions "
                    "are '" + column + "'s row ids; " + why_no_mask(predicates, *vec));
            }
            trace.mask_used = true;
        }

        const HybridPlanner planner = planner_for(*vec);
        std::vector<Neighbor> raw =
            planner.execute_plan(kind, rows, mask, query.data(), k, ef);
        if (out) *out = trace;
        return vec->present(raw);
    }

    // Whether a plan can run at all for these predicates over this column.
    bool plan_available(PlanKind kind, const std::vector<Predicate>& predicates,
                        const std::string& column) const {
        const VectorColumn* vec = require_vector(column);
        if (kind == PlanKind::BitmapFilteredGraph) {
            Bitset mask;
            QueryTrace ignored;
            return vector_mask(predicates, *vec, &mask, &ignored);
        }
        if (kind == PlanKind::NoPredicate) return true;
        return kind == PlanKind::PreFilter || vec->has_graph();
    }

    // The selectivity at or below which a filtered exhaustive scan is
    // preferred. A single measured constant, and that is the point: it is the
    // honest baseline a learned cost model has to beat.
    double prefilter_threshold() const { return prefilter_threshold_; }
    void set_prefilter_threshold(double t) { prefilter_threshold_ = t; }

    // Reclaim the rows deletion could not, and stop every search paying the
    // masked path for them.
    std::size_t compact_vectors(const std::string& column) {
        return require_vector(column)->compact();
    }

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
        // The embedding goes with the row. Exact and immediate as far as every
        // query is concerned; the *space* comes back at the next compaction,
        // because HNSW cannot give a node back (see vector_column.hpp).
        for (auto& [name, vec] : vectors_) {
            if (vec.erase(key)) ++result.indexes_touched;
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

    // How many rows match, without producing them.
    //
    // A bitmap column answers by popcount over n/64 words and materialises
    // nothing; the tree and the learned index have to build the row list and
    // then measure it, which for a permissive predicate is the entire cost of
    // the query paid to learn one number. Both go through the same call, which
    // is the point of hiding the structure.
    std::size_t count(const Predicate& predicate,
                      QueryTrace* trace = nullptr) const {
        const ColumnDef& def = schema_.column(predicate.column);
        if (op_is_indexable(predicate.op) && predicate.op != PredOp::Between &&
            predicate.op != PredOp::Prefix) {
            const auto it = columns_.find(predicate.column);
            if (it != columns_.end() && it->second.index) {
                ensure_fresh(predicate.column);
                if (trace) {
                    trace->used_index = true;
                    trace->reason =
                        std::string("counted through the index on '") +
                        predicate.column + "'";
                }
                const std::size_t matched = it->second.index->count(
                    compare_op_of(predicate.op), predicate.value);
                if (trace) trace->matched = matched;
                return matched;
            }
        }
        (void)def;
        return select_keys(predicate, trace).size();
    }

    // --- conjunctions -------------------------------------------------------
    //
    // Record keys matching **every** predicate, ascending.
    //
    // The planner deliberately took a single predicate, on the grounds that
    // two are set intersection over row-id lists and introduce no new
    // decision. Bitmaps made that false: when both operands are bitmap columns
    // over the same rows, the intersection is a word-parallel AND that touches
    // n/64 words *regardless of how many rows match*, where a sorted merge
    // costs O(m1 + m2). Which is cheaper is now a question, and
    // scripts/experiment_conjunction.py answers it.
    std::vector<std::int64_t> select_all(const std::vector<Predicate>& predicates,
                                         QueryTrace* trace = nullptr) const {
        QueryTrace local;
        if (predicates.empty()) {
            local.reason = "no predicates; every row matches";
            std::vector<std::int64_t> all = store_->keys();
            std::sort(all.begin(), all.end());
            local.matched = all.size();
            if (trace) *trace = local;
            return all;
        }
        if (predicates.size() == 1) {
            return select_keys(predicates.front(), trace);
        }

        std::vector<std::int64_t> keys;
        if (bitmap_conjunction(predicates, &keys, &local)) {
            local.matched = keys.size();
            if (trace) *trace = local;
            return keys;
        }

        // Sorted merge, smallest first. Executing the most selective predicate
        // first bounds every later intersection by its result, so the order is
        // not cosmetic — and knowing which is most selective is free for a
        // bitmap column and costs a full execution for any other.
        keys = merge_conjunction(predicates, &local);
        local.matched = keys.size();
        if (trace) *trace = local;
        return keys;
    }

    std::vector<storage::Record> select_all_records(
        const std::vector<Predicate>& predicates,
        QueryTrace* trace = nullptr) const {
        const std::vector<std::int64_t> keys = select_all(predicates, trace);
        std::vector<storage::Record> out;
        out.reserve(keys.size());
        for (std::int64_t key : keys) {
            if (const storage::Record* r = store_->get(key)) out.push_back(*r);
        }
        return out;
    }

    // Record keys matching **any** predicate, ascending.
    std::vector<std::int64_t> select_any(const std::vector<Predicate>& predicates,
                                         QueryTrace* trace = nullptr) const {
        QueryTrace local;
        std::vector<std::int64_t> out;
        for (const Predicate& p : predicates) {
            QueryTrace one;
            const std::vector<std::int64_t> keys = select_keys(p, &one);
            local.scanned += one.scanned;
            local.used_index = local.used_index || one.used_index;
            std::vector<std::int64_t> merged;
            merged.reserve(out.size() + keys.size());
            std::set_union(out.begin(), out.end(), keys.begin(), keys.end(),
                           std::back_inserter(merged));
            out = std::move(merged);
        }
        local.reason = "union of " + std::to_string(predicates.size()) +
                       " predicates";
        local.matched = out.size();
        if (trace) *trace = local;
        return out;
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

    // Every derived file, atomically. The schema first: a catalog naming an
    // encoding for a column whose type is unknown cannot be interpreted, and
    // neither can a sidecar of raw floats, so the reverse order would leave
    // both readable and neither meaningful.
    void save() const {
        storage::detail::atomic_write(path_of(SCHEMA_NAME),
                                      path_of(SCHEMA_NAME) + ".tmp",
                                      schema_.serialize());
        save_vectors();
        catalog_.save(path_of(CATALOG_NAME));
    }

    // The embeddings, and the metadata without which they are just floats.
    //
    // Separated from save() because it is the one durability point vectors
    // have: they never reach the write-ahead log, so a caller who wants them
    // safe without paying for a full store snapshot takes this.
    //
    // What lands on disk is the *compacted* form — live rows only, renumbered
    // — whether or not memory has caught up, so a reopen is always clean even
    // if compact_vectors() was never called.
    void save_vectors() const {
        if (vectors_.empty()) return;
        for (const auto& [name, vec] : vectors_) {
            vec.save(vector_path(name));
        }
        std::string blob = "{\"version\":1,\"columns\":[";
        bool first = true;
        for (const auto& [name, vec] : vectors_) {
            if (!first) blob += ",";
            first = false;
            blob += vec.metadata();
        }
        blob += "]}";
        storage::detail::atomic_write(path_of(VECTORS_NAME),
                                      path_of(VECTORS_NAME) + ".tmp", blob);
    }

    // Flush pending rebuilds, write everything derived, then snapshot.
    //
    // The derived files go *before* the store's checkpoint, which is the
    // reverse of the obvious order and is deliberate: that call truncates the
    // write-ahead log, and the embeddings are the one thing here with no log
    // to be replayed from. Crashing between the two costs a stale schema and
    // catalog, both of which the next open rebuilds.
    void checkpoint() {
        flush_dirty();
        save();
        store_->checkpoint();
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

        for (const auto& [name, vec] : vectors_) {
            vec.validate();
            // The half VectorColumn cannot check for itself: it knows its keys
            // are consistent with its rows, not that they are rows of *this*
            // table. An embedding whose record is gone is a row knn() would
            // return and get() could not fetch.
            for (std::int64_t key : vec.keys()) {
                if (store_->get(key) == nullptr) {
                    throw std::logic_error(
                        "Table::validate: vector column '" + name +
                        "' holds an embedding for record " +
                        std::to_string(key) + ", which is not in the store");
                }
            }
        }
    }

private:
    struct Column {
        std::unique_ptr<ColumnIndex> index;
        Workload workload;
        // Set by create_index_as: the structure to build rather than measure.
        IndexKind forced = IndexKind::BPlusTree;
        bool force = false;
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

        // Bitmap columns index every row the table holds, not only the rows
        // carrying a value, so that two of them can be AND-ed: position i must
        // mean the same row in both. Rows with no value occupy a position and
        // appear in no bitmap, which is exactly "absent matches nothing".
        std::vector<ColumnValue> row_space = store_->keys();
        std::sort(row_space.begin(), row_space.end());

        ColumnIndex built =
            column->force
                ? ColumnIndex::build_typed_with(
                      type, keys, rows, forced_plan(*column, type, keys, rows),
                      &row_space)
                : catalog_.build_column_typed(name, type, keys, rows,
                                              column->workload, nullptr,
                                              &row_space);
        // A forced structure is recorded too. It skipped the measurement, not
        // the decision -- and a decision the catalog does not hold is one that
        // silently reverts on the next reopen, which is worse than never having
        // made it.
        if (column->force) catalog_.set(name, built.plan());
        column->index = std::make_unique<ColumnIndex>(std::move(built));
        column->dirty = false;
        (void)skipped;  // reported by info_of() from the store, not tracked
    }

    // The plan create_index_as implies: the named kind, with the encoding its
    // shape requires.
    template <typename T>
    static IndexPlan forced_plan(const Column& column, LogicalType type,
                                 const std::vector<T>& keys,
                                 const std::vector<ColumnValue>& rows) {
        const ColumnShape shape = index::measure_shape(keys, rows);
        IndexPlan plan;
        plan.kind = column.forced;
        plan.type = type;
        plan.btree_order = 32;
        plan.rmi_models = 1024;
        plan.search_threshold = 64;
        if (column.forced == IndexKind::Bitmap) {
            plan.encoding = KeyEncoding::Dictionary;
        } else if (column.forced == IndexKind::BPlusTree) {
            plan.encoding =
                shape.unique ? KeyEncoding::Native : KeyEncoding::Composite;
        } else {
            plan.encoding = KeyEncoding::Native;
        }
        return plan;
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
            case LogicalType::Bool:
                build_as<bool>(name, def.type, column);
                return;
            default:
                break;
        }
        reject_unindexable(def);
    }

    static void reject_unindexable(const ColumnDef& def) {
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
        const bool representable = column.index->insert_row(value, key);
        ++result->indexes_touched;

        // False means the structure cannot represent the result, whatever the
        // structure is: a natively-keyed column has just been made non-unique,
        // or a bitmap has been handed a row that would shift every position
        // after it. Either way the repair is the same and the call site does
        // not need to know which happened.
        if (!representable) mark_dirty(column, result);
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

    // ---- conjunction strategies --------------------------------------------

    // Word-parallel AND, when every predicate is served by a bitmap column and
    // those columns index the same rows in the same order.
    //
    // The row-space check is not defensive padding: two bitmaps whose
    // positions mean different rows would produce an answer that looks
    // plausible and matches neither predicate. Bitmap columns are built over a
    // table's entire row set so that they *do* line up, and this confirms it
    // rather than assuming it.
    bool bitmap_conjunction(const std::vector<Predicate>& predicates,
                            std::vector<std::int64_t>* out,
                            QueryTrace* trace) const {
        Bitset mask;
        if (!bitmap_mask(predicates, &mask, trace)) return false;
        *out = columns_.at(predicates.front().column).index->decode(mask);
        return true;
    }

    // The conjunction as a bit set, without decoding it.
    //
    // Split out from the above because the decode is exactly what a vector
    // search does not want: it takes the bit set directly, so materialising
    // the row ids first would hand back the entire saving at the door.
    bool bitmap_mask(const std::vector<Predicate>& predicates, Bitset* out,
                     QueryTrace* trace) const {
        if (predicates.empty()) return false;
        std::size_t space = 0;
        bool first = true;
        for (const Predicate& p : predicates) {
            if (!op_is_indexable(p.op) || p.op == PredOp::Between ||
                p.op == PredOp::Prefix) {
                return false;
            }
            const auto it = columns_.find(p.column);
            if (it == columns_.end() || !it->second.index) return false;
            ensure_fresh(p.column);
            const ColumnIndex& index = *it->second.index;
            if (!index.has_bitmap()) return false;
            if (first) {
                space = index.row_space();
                first = false;
            } else if (index.row_space() != space) {
                return false;
            }
        }

        Bitset mask;
        bool started = false;
        for (const Predicate& p : predicates) {
            const ColumnIndex& index = *columns_.at(p.column).index;
            Bitset one = index.bitmap_for(compare_op_of(p.op), p.value);
            if (!started) {
                mask = std::move(one);
                started = true;
            } else {
                mask &= one;
            }
        }
        *out = std::move(mask);
        trace->used_index = true;
        trace->reason = "bitmap AND over " + std::to_string(predicates.size()) +
                        " columns, " + std::to_string((space + 63) / 64) +
                        " words";
        return true;
    }

    // Execute the most selective predicate first, then intersect.
    //
    // Selectivity is known for free on a bitmap column (a popcount) and costs
    // a full execution on any other, so the ordering pass asks each column for
    // a count and only the cheap answers are actually cheap. That asymmetry is
    // itself a reason the bitmap earns its place.
    std::vector<std::int64_t> merge_conjunction(
        const std::vector<Predicate>& predicates, QueryTrace* trace) const {
        std::vector<const Predicate*> order;
        order.reserve(predicates.size());
        for (const Predicate& p : predicates) order.push_back(&p);

        std::stable_sort(order.begin(), order.end(),
                         [this](const Predicate* a, const Predicate* b) {
                             return estimate(*a) < estimate(*b);
                         });

        std::vector<std::int64_t> out;
        bool started = false;
        for (const Predicate* p : order) {
            QueryTrace one;
            const std::vector<std::int64_t> keys = select_keys(*p, &one);
            trace->scanned += one.scanned;
            trace->used_index = trace->used_index || one.used_index;
            if (!started) {
                out = keys;
                started = true;
            } else {
                std::vector<std::int64_t> merged;
                merged.reserve(std::min(out.size(), keys.size()));
                std::set_intersection(out.begin(), out.end(), keys.begin(),
                                      keys.end(), std::back_inserter(merged));
                out = std::move(merged);
            }
            if (out.empty()) break;  // nothing can rejoin an empty set
        }
        trace->reason = "sorted merge over " + std::to_string(predicates.size()) +
                        " predicates, most selective first";
        return out;
    }

    // How many rows a predicate is expected to match, for ordering only.
    //
    // Exact and nearly free on a bitmap column; on any other this is the whole
    // query, so the estimate is the row count instead — deliberately useless,
    // because paying to order the work would cost more than the ordering saves.
    std::size_t estimate(const Predicate& p) const {
        if (!op_is_indexable(p.op) || p.op == PredOp::Between ||
            p.op == PredOp::Prefix) {
            return store_->size();
        }
        const auto it = columns_.find(p.column);
        if (it == columns_.end() || !it->second.index ||
            !it->second.index->has_bitmap()) {
            return store_->size();
        }
        ensure_fresh(p.column);
        return columns_.at(p.column).index->count(compare_op_of(p.op), p.value);
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

    // ---- vector helpers ----------------------------------------------------

    VectorColumn* require_vector(const std::string& name) {
        return const_cast<VectorColumn*>(
            static_cast<const Table*>(this)->require_vector(name));
    }

    const VectorColumn* require_vector(const std::string& name) const {
        const auto it = vectors_.find(name);
        if (it != vectors_.end()) return &it->second;
        // Two distinct mistakes, and telling them apart is the difference
        // between "you forgot a call" and "you meant a different column".
        const ColumnDef& def = schema_.column(name);
        if (def.type != LogicalType::Vector) {
            throw std::invalid_argument(
                "Table: column '" + name + "' is " + index::to_string(def.type) +
                ", not a vector");
        }
        throw std::invalid_argument(
            "Table: vector column '" + name +
            "' has no index yet; call create_vector_index('" + name + "')");
    }

    // A planner holding this column's two structures and no columns at all.
    //
    // Cheap to build — two pointers and a threshold — which is what makes it
    // right to build per query rather than to keep. HybridPlanner::set_column
    // *copies* the index, so a planner held as a member would keep a second
    // copy of every scalar index alive for the table's lifetime, and Table
    // already owns the originals and resolves predicates through them.
    HybridPlanner planner_for(const VectorColumn& vec) const {
        HybridPlanner planner(prefilter_threshold_);
        planner.set_exact(&vec.exact());
        planner.set_graph(vec.graph());
        return planner;
    }

    // The predicates as a bit set the vector search can use directly.
    //
    // Three things all have to hold, and each of them is a way of being wrong
    // that would look plausible: every predicate must be served by a bitmap
    // column, those columns must agree on what a position means, and a
    // position must mean the *same row* in the vector index. The last is the
    // one that only exists here — the table's positions are ranks in its key
    // space and the vector index's are positions in a float buffer.
    bool vector_mask(const std::vector<Predicate>& predicates,
                     const VectorColumn& vec, Bitset* out,
                     QueryTrace* trace) const {
        if (!vec.rows_are_keys()) return false;
        if (!bitmap_mask(predicates, out, trace)) return false;
        if (out->size() != vec.rows()) {
            *out = Bitset();
            return false;
        }
        return true;
    }

    // Why the mask was unavailable, for the one caller that has to explain a
    // refusal rather than quietly take another route.
    std::string why_no_mask(const std::vector<Predicate>& predicates,
                            const VectorColumn& vec) const {
        if (!vec.rows_are_keys()) {
            return "this column's row ids are not its record keys (" +
                   std::to_string(vec.orphans()) +
                   " orphaned rows), so a bit set over the table's rows would "
                   "select different vectors than it names";
        }
        for (const Predicate& p : predicates) {
            const auto it = columns_.find(p.column);
            if (it == columns_.end() || !it->second.index) {
                return "'" + p.column + "' has no index";
            }
            if (!it->second.index->has_bitmap()) {
                return "'" + p.column + "' is a " +
                       index::to_string(it->second.index->kind()) +
                       ", not a bitmap";
            }
        }
        return "the predicates and the bitmaps do not cover the same rows";
    }

    VectorInfo vector_info_of(const std::string& name, const ColumnDef& def,
                              const VectorColumn* vec) const {
        VectorInfo out;
        out.name = name;
        out.dim = def.dim;
        if (vec == nullptr) return out;
        out.indexed = true;
        out.rows = vec->size();
        out.orphans = vec->orphans();
        out.structure = vec->plan().structure;
        out.metric = vec->metric();
        out.has_graph = vec->has_graph();
        out.rows_are_keys = vec->rows_are_keys();
        out.memory_bytes = vec->memory_bytes();
        return out;
    }

    // Rebuild every vector column from vectors.json and its sidecars.
    //
    // Only the *vectors* are stored, never the graph. Replaying the same
    // insertions in the same order with the seed fixed reproduces it exactly,
    // so a reopen returns the same neighbours rather than merely similar ones
    // — which is what makes that an assertion in the test suite instead of a
    // hope. The cost is real and measured in
    // scripts/experiment_vector_sidecar.py.
    void load_vectors() {
        const std::string blob = read_file(path_of(VECTORS_NAME));
        if (blob.empty()) return;
        using namespace hylis::storage::json_detail;
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
                        load_one_vector_column(p);
                        skip_ws(p);
                        if (*p == ',') { ++p; continue; }
                        if (*p == ']') { ++p; break; }
                        throw std::runtime_error("vectors: expected , or ]");
                    }
                }
            } else {
                skip_value(p);
            }
            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("vectors: expected , or } at top level");
        }
    }

    void load_one_vector_column(const char*& p) {
        using namespace hylis::storage::json_detail;
        std::string name;
        std::size_t dim = 0;
        VectorPlan plan;
        std::vector<std::int64_t> keys;

        skip_ws(p);
        expect(p, '{');
        skip_ws(p);
        if (*p == '}') { ++p; return; }
        while (true) {
            skip_ws(p);
            const std::string field = read_string(p);
            skip_ws(p);
            expect(p, ':');
            skip_ws(p);
            if (field == "name") {
                name = read_string(p);
            } else if (field == "dim") {
                dim = static_cast<std::size_t>(read_int(p));
            } else if (field == "metric") {
                plan.metric = index::metric_from_string(read_string(p));
            } else if (field == "structure") {
                const std::string s = read_string(p);
                if (!try_vector_structure_from_string(s, &plan.structure)) {
                    throw std::runtime_error("vectors: unknown structure '" + s + "'");
                }
            } else if (field == "m") {
                plan.M = static_cast<std::size_t>(read_int(p));
            } else if (field == "ef_construction") {
                plan.ef_construction = static_cast<std::size_t>(read_int(p));
            } else if (field == "ef_search") {
                plan.ef_search = static_cast<std::size_t>(read_int(p));
            } else if (field == "seed") {
                plan.seed = static_cast<std::uint64_t>(read_int(p));
            } else if (field == "keys") {
                keys = read_int_array(p);
            } else {
                skip_value(p);
            }
            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("vectors: expected , or } in a column");
        }

        // A stored sidecar for a column the schema no longer declares is a
        // schema change the reconcile() rules already forbid; a *dimension*
        // change is the same mistake one level down, and re-adding the floats
        // under the new dimension would reinterpret every one of them.
        const ColumnDef& def = schema_.column(name);
        if (def.type != LogicalType::Vector || def.dim != dim) {
            throw std::runtime_error(
                "Table: vectors.json describes '" + name + "' as a " +
                std::to_string(dim) + "-dimensional vector, and the schema "
                "declares it as " + index::to_string(def.type) +
                (def.type == LogicalType::Vector
                     ? " of " + std::to_string(def.dim) + " dimensions"
                     : ""));
        }

        VectorColumn column(name, dim, plan);
        column.load(vector_path(name), keys);
        vectors_.insert_or_assign(name, std::move(column));
    }

    fs::path vector_path(const std::string& name) const {
        return store_->directory() / (name + ".fvecs");
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
    std::map<std::string, VectorColumn> vectors_;
    std::size_t rebuilds_ = 0;
    bool deferring_ = false;
    // The measured pre-filter crossover, handed to every planner this table
    // builds. HybridPlanner::calibrate() finds a corpus's own; until it is
    // called this is the ~50% one module 5 measured, and both are constants
    // rather than a model on purpose — a learned cost model has to beat a
    // stated baseline to have beaten anything.
    double prefilter_threshold_ = 0.5;
};

}  // namespace hylis::query
