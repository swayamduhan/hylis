// bindings/table.cpp
//
// pybind11 binding for the store/index join, importable as `hylis._table`.
//
// The store is *borrowed*, not owned, so the binding uses keep_alive to stop
// Python collecting a RecordStore that a live Table still points at. Without
// it, `Table(RecordStore(dir), schema)` would leave a dangling pointer the
// moment the temporary store was collected -- which is a use-after-free that
// usually looks like a mysteriously empty table rather than a crash.
//
// LogicalType and the plan types come from `hylis._rmi`, which this module
// imports at load so registration order is guaranteed.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "query/table.hpp"

namespace py = pybind11;

using hylis::index::Datum;
using hylis::index::LogicalType;
using hylis::index::Metric;
using hylis::index::Workload;
using hylis::query::ColumnInfo;
using hylis::query::HybridTrace;
using hylis::query::PlanKind;
using hylis::query::PredOp;
using hylis::query::Predicate;
using hylis::query::QueryTrace;
using hylis::query::Schema;
using hylis::query::Table;
using hylis::query::VectorInfo;
using hylis::query::VectorMatch;
using hylis::query::VectorPlan;
using hylis::query::VectorStructure;
using hylis::query::WriteResult;
using hylis::storage::Record;
using hylis::storage::RecordStore;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// A 1-D query vector, as a std::vector so it can go straight into Table's
// interface. Copied rather than borrowed: the copy is dim floats against a
// search that touches the whole corpus, and borrowing would need a lifetime
// story for a numpy array Python may free at any point.
std::vector<float> query_vector(const FloatArray& arr) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument("a query must be a 1-D array, got " +
                                    std::to_string(arr.ndim()) + "-D");
    }
    return std::vector<float>(arr.data(), arr.data() + arr.shape(0));
}

Datum datum_from_py(py::handle value) {
    // bool before int: in Python bool is a subclass of int.
    if (py::isinstance<py::bool_>(value)) return Datum{value.cast<bool>()};
    if (py::isinstance<py::int_>(value)) return Datum{value.cast<std::int64_t>()};
    if (py::isinstance<py::float_>(value)) return Datum{value.cast<double>()};
    if (py::isinstance<py::str>(value)) return Datum{value.cast<std::string>()};
    if (value.is_none()) return Datum{};
    throw std::invalid_argument(
        "a predicate value must be an int, float, str or bool; got " +
        std::string(py::str(py::type::of(value))));
}

Predicate predicate_from_py(const std::string& column, PredOp op,
                            py::handle value, py::handle value2) {
    Predicate p;
    p.column = column;
    p.op = op;
    p.value = datum_from_py(value);
    p.value2 = datum_from_py(value2);
    return p;
}

// A list of (column, op, value[, value2]) tuples.
//
// Tuples rather than a bound Predicate class: a conjunction is written inline
// at the call site far more often than it is held, and `[("a", Eq, 1)]` reads
// better there than three constructor calls.
std::vector<Predicate> predicates_from_py(const py::list& items) {
    std::vector<Predicate> out;
    out.reserve(items.size());
    for (const py::handle item : items) {
        const py::sequence parts = py::reinterpret_borrow<py::sequence>(item);
        if (parts.size() < 3 || parts.size() > 4) {
            throw std::invalid_argument(
                "each predicate must be (column, op, value) or "
                "(column, op, value, value2)");
        }
        out.push_back(predicate_from_py(
            parts[0].cast<std::string>(), parts[1].cast<PredOp>(), parts[2],
            parts.size() == 4 ? parts[3] : py::none()));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_table, m) {
    m.doc() = "hylis table layer (C++): the join between the record store and "
              "the indexes";
    hylis::bindings::attach_build_info(m);

    py::module_::import("hylis._rmi");
    py::module_::import("hylis._schema");
    py::module_::import("hylis._storage");
    py::module_::import("hylis._flat");  // Metric
    // Predicate, PredOp, PlanKind and QueryPlan all live in _planner, and the
    // direction of that import is not a preference. pybind11 allows one owner
    // per C++ type, and table.hpp *includes* planner.hpp — so _table is
    // downstream and imports upstream. Registering Predicate here instead (as
    // phase D did) makes _planner import _table, and with this module now
    // needing QueryPlan the two form a cycle: Python hands back a
    // half-initialised module and pybind11 reports "type PlanKind is already
    // registered" at import.
    py::module_::import("hylis._planner");

    py::class_<ColumnInfo>(m, "ColumnInfo", "A column, and what indexes it.")
        .def_readonly("name", &ColumnInfo::name)
        .def_readonly("type", &ColumnInfo::type)
        .def_readonly("indexed", &ColumnInfo::indexed)
        .def_readonly("rows", &ColumnInfo::rows,
                      "Rows carrying a value for this column.")
        .def_readonly("skipped", &ColumnInfo::skipped,
                      "Rows with no value for it. Not an error count: such a\n"
                      "row is absent from the index and matches no predicate.")
        .def_readonly("distinct", &ColumnInfo::distinct)
        .def_readonly("unique", &ColumnInfo::unique)
        .def_readonly("monotone", &ColumnInfo::monotone)
        .def_readonly("dirty", &ColumnInfo::dirty,
                      "Whether a write left the index needing a rebuild before\n"
                      "it can be read. Never a correctness hazard: every read\n"
                      "path rebuilds first.")
        .def_readonly("kind", &ColumnInfo::kind)
        .def_readonly("encoding", &ColumnInfo::encoding)
        .def_readonly("index_bytes", &ColumnInfo::index_bytes)
        .def_readonly("ns_per_lookup", &ColumnInfo::ns_per_lookup)
        .def("__repr__", [](const ColumnInfo& c) {
            std::string out = "ColumnInfo('" + c.name + "', " +
                              hylis::index::to_string(c.type);
            if (c.indexed) {
                out += ", " + std::string(hylis::index::to_string(c.kind)) + "/" +
                       hylis::index::to_string(c.encoding);
            } else {
                out += ", no index";
            }
            return out + ", rows=" + std::to_string(c.rows) + ")";
        });

    py::class_<WriteResult>(m, "WriteResult", "What a write cost.")
        .def_readonly("created", &WriteResult::created)
        .def_readonly("rows_written", &WriteResult::rows_written)
        .def_readonly("rows_created", &WriteResult::rows_created)
        .def_readonly("indexes_touched", &WriteResult::indexes_touched)
        .def_readonly("rebuilds_triggered", &WriteResult::rebuilds_triggered,
                      "Columns this write forced into a full rebuild. Zero on\n"
                      "any workload that does not write to an immutable or\n"
                      "natively-keyed column.")
        .def("__repr__", [](const WriteResult& w) {
            return "WriteResult(rows=" + std::to_string(w.rows_written) +
                   ", created=" + std::to_string(w.rows_created) +
                   ", indexes=" + std::to_string(w.indexes_touched) +
                   ", rebuilds=" + std::to_string(w.rebuilds_triggered) + ")";
        });

    py::enum_<VectorStructure>(m, "VectorStructure",
        "Which structures a vector column builds.\n\n"
        "The exact index is present either way. It is the storage -- the\n"
        "contiguous float buffer everything else is built from -- and it is\n"
        "the oracle: HNSW is approximate by construction, so recall is the\n"
        "only question that can be asked of it, and recall is undefined\n"
        "without a true answer.")
        .value("Exact", VectorStructure::Exact,
               "Brute force only: exhaustive, and the answer every\n"
               "approximate result is graded against.")
        .value("Graph", VectorStructure::Graph,
               "HNSW as well, for corpora where scanning everything stops\n"
               "paying.");

    py::class_<VectorPlan>(m, "VectorPlan",
        "How a vector column is built.\n\n"
        "Named rather than measured, unlike every scalar column. choose_index\n"
        "times lookups; the question for a vector index is recall at a target,\n"
        "which needs a query workload and a ground truth a table does not\n"
        "have. The recall measurement stays in scripts/bench_vector.py,\n"
        "against the exact index this column keeps for exactly that purpose.")
        .def(py::init([](VectorStructure structure, Metric metric, std::size_t M,
                         std::size_t ef_construction, std::size_t ef_search,
                         std::uint64_t seed) {
                 VectorPlan p;
                 p.structure = structure;
                 p.metric = metric;
                 p.M = M;
                 p.ef_construction = ef_construction;
                 p.ef_search = ef_search;
                 p.seed = seed;
                 return p;
             }),
             py::arg("structure") = VectorStructure::Graph,
             py::arg("metric") = Metric::L2, py::arg("M") = 16,
             py::arg("ef_construction") = 200, py::arg("ef_search") = 50,
             py::arg("seed") = 100)
        .def_readwrite("structure", &VectorPlan::structure)
        .def_readwrite("metric", &VectorPlan::metric)
        .def_readwrite("M", &VectorPlan::M)
        .def_readwrite("ef_construction", &VectorPlan::ef_construction)
        .def_readwrite("ef_search", &VectorPlan::ef_search)
        .def_readwrite("seed", &VectorPlan::seed,
                       "Fixed, so a rebuild from the same insertion order\n"
                       "reproduces the graph exactly rather than approximately.\n"
                       "That is what makes 'reopen returns the same neighbours'\n"
                       "an assertion instead of a hope.")
        .def("__repr__", [](const VectorPlan& p) {
            return std::string("VectorPlan(") +
                   hylis::query::to_string(p.structure) + ", " +
                   hylis::index::to_string(p.metric) +
                   ", M=" + std::to_string(p.M) + ")";
        });

    py::class_<VectorMatch>(m, "VectorMatch", "One neighbour, in the table's terms.")
        .def_readonly("key", &VectorMatch::key, "The record key.")
        .def_readonly("row", &VectorMatch::row,
                      "Its position inside the vector index. Kept because it is\n"
                      "what a second vector call wants, and because comparing\n"
                      "two plans should not need a round trip through the store.")
        .def_readonly("score", &VectorMatch::score,
                      "Whatever the column's metric measures: a distance for\n"
                      "L2, a dot product for InnerProduct, a cosine similarity\n"
                      "for Cosine. Best first regardless of which.")
        .def("__repr__", [](const VectorMatch& m) {
            return "VectorMatch(key=" + std::to_string(m.key) +
                   ", score=" + std::to_string(m.score) + ")";
        });

    py::class_<VectorInfo>(m, "VectorInfo", "A vector column, and what it holds.")
        .def_readonly("name", &VectorInfo::name)
        .def_readonly("dim", &VectorInfo::dim)
        .def_readonly("indexed", &VectorInfo::indexed)
        .def_readonly("rows", &VectorInfo::rows, "Rows with a live embedding.")
        .def_readonly("orphans", &VectorInfo::orphans,
                      "Rows deletion could not reclaim. HNSW cannot give a node\n"
                      "back, so a delete is a mask rather than a removal; the\n"
                      "space returns at the next compaction. Visible because it\n"
                      "costs twice: memory, and every search taking the masked\n"
                      "path.")
        .def_readonly("structure", &VectorInfo::structure)
        .def_readonly("metric", &VectorInfo::metric)
        .def_readonly("has_graph", &VectorInfo::has_graph)
        .def_readonly("rows_are_keys", &VectorInfo::rows_are_keys,
                      "Whether row id i is record key i. The precondition for\n"
                      "using a table bitmap as a vector mask: bit position i\n"
                      "has to mean the same row in both.")
        .def_readonly("memory_bytes", &VectorInfo::memory_bytes,
                      "Including *both* copies of the corpus when a graph is\n"
                      "built. The exact index keeps its own, which is the price\n"
                      "of brute force staying independently searchable.")
        .def("__repr__", [](const VectorInfo& v) {
            return "VectorInfo('" + v.name + "', dim=" + std::to_string(v.dim) +
                   ", rows=" + std::to_string(v.rows) +
                   ", orphans=" + std::to_string(v.orphans) + ")";
        });

    py::class_<QueryTrace>(m, "QueryTrace", "Why a query executed as it did.")
        .def_readonly("reason", &QueryTrace::reason)
        .def_readonly("used_index", &QueryTrace::used_index)
        .def_readonly("scanned", &QueryTrace::scanned,
                      "Rows read directly from the store. Non-zero exactly\n"
                      "when no index could serve the predicate.")
        .def_readonly("matched", &QueryTrace::matched)
        .def("__repr__", [](const QueryTrace& t) {
            return "QueryTrace(" + std::string(t.used_index ? "index" : "scan") +
                   ", matched=" + std::to_string(t.matched) + ")";
        });

    py::class_<HybridTrace>(m, "HybridTrace",
        "What a hybrid query did: the structured half, the plan, and the join.")
        .def_readonly("structured", &HybridTrace::structured,
                      "How the predicates were resolved.")
        .def_readonly("plan", &HybridTrace::plan, "What the planner chose, and why.")
        .def_readonly("without_vector", &HybridTrace::without_vector,
                      "Rows the predicate matched that carry no embedding.\n"
                      "Dropped, because a row with no vector cannot be a\n"
                      "nearest neighbour -- and counted, because returning\n"
                      "fewer rows than the predicate matched must not be silent.")
        .def_readonly("mask_used", &HybridTrace::mask_used,
                      "Whether the structured half was answered by a popcount,\n"
                      "so no row id was materialised before the search began.")
        .def("__repr__", [](const HybridTrace& t) {
            return std::string("HybridTrace(") +
                   hylis::query::to_string(t.plan.kind) +
                   ", matched=" + std::to_string(t.plan.matched_rows) + ")";
        });

    py::class_<Table>(m, "Table",
        "A table: records in a store, indexes over their columns.\n\n"
        "Owns nothing but the connection. The records belong to the\n"
        "RecordStore, the structures to ColumnIndex, the decisions to\n"
        "IndexCatalog and the types to Schema; this adds extraction,\n"
        "write-path maintenance, and the reopen story.")
        .def(py::init<RecordStore&, Schema>(), py::arg("store"), py::arg("schema"),
             // The store is borrowed and must outlive the table.
             py::keep_alive<1, 2>())
        .def_static("open", &Table::open, py::arg("store"),
                    py::keep_alive<0, 1>(),
                    "Reopen using the schema on disk. A stored plan is\n"
                    "uninterpretable without it: 'encoding: composite' says\n"
                    "nothing unless the key type is known.")

        // --- DDL ---
        .def_property_readonly("schema", &Table::schema)
        .def_property_readonly("catalog", &Table::catalog,
                               py::return_value_policy::reference_internal,
                               "The persisted decisions. Plans, never the\n"
                               "fitted models: a model rebuilds in milliseconds,\n"
                               "while producing a plan means building and timing\n"
                               "every candidate.")
        .def("add_column", &Table::add_column, py::arg("column"))
        .def("create_index", [](Table& self, const std::string& name,
                                double write_fraction) {
            Workload workload;
            workload.write_fraction = write_fraction;
            return self.create_index(name, workload);
        }, py::arg("column"), py::arg("write_fraction") = 0.0,
           "Build or re-tune an index.\n\n"
           "The write rate is an input, not a refinement: on lookups alone\n"
           "the static RMI wins nearly everything, so without it that\n"
           "build-only structure would be chosen for columns about to be\n"
           "written to, which then cost a rebuild on the first write.")
        .def("create_index_as", [](Table& self, const std::string& name,
                                   hylis::index::IndexKind kind,
                                   double write_fraction) {
            Workload workload;
            workload.write_fraction = write_fraction;
            return self.create_index_as(name, kind, workload);
        }, py::arg("column"), py::arg("kind"), py::arg("write_fraction") = 0.0,
           "Build a named structure, skipping the measurement.\n\n"
           "choose_index times lookups and writes and nothing else, so a\n"
           "column whose workload is dominated by count() or by conjunctions\n"
           "is judged on the one thing a bitmap is worst at. This is how an\n"
           "experiment gets the family it wants to compare, and how a user who\n"
           "knows their query mix overrides a lookup benchmark.")
        .def("drop_index", &Table::drop_index, py::arg("column"))
        .def("has_index", &Table::has_index, py::arg("column"))
        .def("describe", &Table::describe)
        .def("info", &Table::info, py::arg("column"))
        .def("explain_column", &Table::explain_column, py::arg("column"),
             "The plan and its measured evidence: ns/lookup, bytes, error bound.")

        // --- DML ---
        .def("put", &Table::put, py::arg("record"),
             "Insert or update. Type-checked against the schema before\n"
             "anything is written, so a record that would half-load is refused.")
        .def("put_batch", &Table::put_batch, py::arg("records"),
             "Load many records with one index-maintenance pass.\n\n"
             "Not an optimisation but the only usable bulk-load path: a column\n"
             "that falls back to a rebuild would pay one per record otherwise.")
        .def("update", [](Table& self, std::int64_t key, const py::dict& changes) {
            std::map<std::string, Datum> typed;
            for (auto item : changes) {
                typed[item.first.cast<std::string>()] = datum_from_py(item.second);
            }
            return self.update(key, typed);
        }, py::arg("key"), py::arg("changes"))
        .def("erase", &Table::erase, py::arg("key"))
        .def("get", [](const Table& self, std::int64_t key) -> py::object {
            const auto r = self.get(key);
            return r ? py::cast(*r) : py::none();
        }, py::arg("key"))

        // --- queries ---
        .def("select_keys", [](const Table& self, const std::string& column,
                               PredOp op, py::handle value, py::handle value2) {
            QueryTrace trace;
            auto keys = self.select_keys(
                predicate_from_py(column, op, value, value2), &trace);
            return py::make_tuple(std::move(keys), trace);
        }, py::arg("column"), py::arg("op"), py::arg("value") = py::none(),
           py::arg("value2") = py::none(),
           "Matching record keys, ascending, with the trace.\n\n"
           "Sorted deliberately: an index returns rows in the column's order\n"
           "and a scan returns them in the store's hash order, so without it\n"
           "the ordering would depend on which structure answered.")
        .def("select", [](const Table& self, const std::string& column, PredOp op,
                          py::handle value, py::handle value2) {
            QueryTrace trace;
            auto rows = self.select(
                predicate_from_py(column, op, value, value2), &trace);
            return py::make_tuple(std::move(rows), trace);
        }, py::arg("column"), py::arg("op"), py::arg("value") = py::none(),
           py::arg("value2") = py::none(),
           "Matching records, with the trace.")
        .def("count", [](const Table& self, const std::string& column, PredOp op,
                         py::handle value, py::handle value2) {
            return self.count(predicate_from_py(column, op, value, value2));
        }, py::arg("column"), py::arg("op"), py::arg("value") = py::none(),
           py::arg("value2") = py::none(),
           "How many rows match, without producing them.\n\n"
           "A bitmap column answers by popcount over n/64 words; every other\n"
           "family builds the row list and measures it.")
        .def("select_all", [](const Table& self, const py::list& predicates) {
            QueryTrace trace;
            auto keys = self.select_all(predicates_from_py(predicates), &trace);
            return py::make_tuple(std::move(keys), trace);
        }, py::arg("predicates"),
           "Record keys matching every predicate, ascending, with the trace.\n\n"
           "Each predicate is a (column, op, value[, value2]) tuple. When all\n"
           "of them are served by bitmap columns over the same rows this is a\n"
           "word-parallel AND touching n/64 words whatever matches; otherwise\n"
           "it is a sorted merge, most selective predicate first. The trace\n"
           "says which ran.")
        .def("select_all_records", [](const Table& self, const py::list& predicates) {
            QueryTrace trace;
            auto rows = self.select_all_records(predicates_from_py(predicates),
                                                &trace);
            return py::make_tuple(std::move(rows), trace);
        }, py::arg("predicates"))
        .def("select_any", [](const Table& self, const py::list& predicates) {
            QueryTrace trace;
            auto keys = self.select_any(predicates_from_py(predicates), &trace);
            return py::make_tuple(std::move(keys), trace);
        }, py::arg("predicates"),
           "Record keys matching any predicate, ascending and deduplicated.")
        .def("scan", &Table::scan, py::arg("limit") = 0, py::arg("offset") = 0)

        // --- vectors ---
        .def("create_vector_index", &Table::create_vector_index,
             py::arg("column"), py::arg("plan") = VectorPlan{},
             "Build (or re-tune) the structures behind a vector column.\n\n"
             "Re-tuning keeps every embedding, except that the metric cannot\n"
             "change on a non-empty column: Cosine stores the normalised form,\n"
             "so the vectors handed in are no longer recoverable.")
        .def("drop_vector_index", &Table::drop_vector_index, py::arg("column"))
        .def("has_vector_index", &Table::has_vector_index, py::arg("column"))
        .def("describe_vectors", &Table::describe_vectors)
        .def("vector_info", &Table::vector_info, py::arg("column"))
        .def("put_vector", [](Table& self, std::int64_t key,
                              const std::string& column, const FloatArray& vec) {
            return self.put_vector(key, column, query_vector(vec));
        }, py::arg("key"), py::arg("column"), py::arg("vector"),
           "Attach an embedding to a record, returning the row it occupies.\n\n"
           "The record must already exist: an embedding belongs to a row, and\n"
           "knn() answers in record keys the caller is going to fetch.\n\n"
           "**Not write-ahead logged.** The floats reach disk at\n"
           "save_vectors() and nowhere else, so a crash before that loses\n"
           "every embedding attached since the last one while the records\n"
           "themselves survive. That is the cost of keeping a 128-float vector\n"
           "out of a JSON WAL.")
        .def("put_vectors", [](Table& self, const std::string& column,
                               const std::vector<std::int64_t>& keys,
                               const FloatArray& data) {
            if (data.ndim() != 2) {
                throw std::invalid_argument(
                    "put_vectors: expected an (n, dim) array, got " +
                    std::to_string(data.ndim()) + "-D");
            }
            if (static_cast<std::size_t>(data.shape(0)) != keys.size()) {
                throw std::invalid_argument(
                    "put_vectors: " + std::to_string(keys.size()) + " keys but " +
                    std::to_string(data.shape(0)) + " rows of vectors");
            }
            const std::vector<float> flat(
                data.data(), data.data() + data.shape(0) * data.shape(1));
            return self.put_vectors(column, keys, flat);
        }, py::arg("column"), py::arg("keys"), py::arg("vectors"),
           "Attach many embeddings at once, keys against an (n, dim) array.")
        .def("erase_vector", &Table::erase_vector, py::arg("key"), py::arg("column"))
        .def("has_vector", &Table::has_vector, py::arg("key"), py::arg("column"))
        .def("get_vector", [](const Table& self, std::int64_t key,
                              const std::string& column) -> py::object {
            const std::vector<float> v = self.get_vector(key, column);
            if (v.empty()) return py::none();
            return py::array_t<float>(static_cast<py::ssize_t>(v.size()), v.data());
        }, py::arg("key"), py::arg("column"),
           "The stored embedding, or None. For Cosine this is the normalised\n"
           "form, which is what the index compares against.")
        .def("vector_keys", &Table::vector_keys, py::arg("column"),
             "Every record key carrying an embedding, in row order.")
        .def("compact_vectors", &Table::compact_vectors, py::arg("column"),
             "Reclaim the rows deletion could not, and stop every search\n"
             "paying the masked path for them. Returns how many were freed.")

        .def("knn", [](const Table& self, const std::string& column,
                       const FloatArray& query, std::size_t k, std::size_t ef,
                       bool exact) {
            return self.knn(column, query_vector(query), k, ef, exact);
        }, py::arg("column"), py::arg("query"), py::arg("k") = 10,
           py::arg("ef") = 0, py::arg("exact") = false,
           "k nearest, best first, in record keys.\n\n"
           "`exact` forces the brute-force scan. Selectable rather than a\n"
           "fallback: it is the only exact answer here, and it is what every\n"
           "approximate result is graded against.")
        .def("knn_by_key", [](const Table& self, const std::string& column,
                              std::int64_t key, std::size_t k, std::size_t ef,
                              bool exact) {
            return self.knn_by_key(column, key, k, ef, exact);
        }, py::arg("column"), py::arg("key"), py::arg("k") = 10,
           py::arg("ef") = 0, py::arg("exact") = false,
           "More-like-this, seeded by a row already in the table.\n\n"
           "The seed is excluded: a row is always its own nearest neighbour,\n"
           "so returning it would spend one of the k on something the caller\n"
           "already has.")
        .def("hybrid", [](const Table& self, const py::list& predicates,
                          const std::string& column, const FloatArray& query,
                          std::size_t k, std::size_t ef) {
            HybridTrace trace;
            auto matches = self.hybrid(predicates_from_py(predicates), column,
                                       query_vector(query), k, ef, &trace);
            return py::make_tuple(std::move(matches), trace);
        }, py::arg("predicates"), py::arg("column"), py::arg("query"),
           py::arg("k") = 10, py::arg("ef") = 0,
           "k nearest among rows satisfying the predicates, with the trace.\n\n"
           "Every plan returns the same rows; the planner chooses between\n"
           "costs, never between answers. When every predicate is served by a\n"
           "bitmap column whose positions are this column's row ids, the\n"
           "structured half is a popcount and no row id is materialised at\n"
           "all; otherwise the predicates resolve to record keys which are\n"
           "then translated to rows.")
        .def("hybrid_with", [](const Table& self, PlanKind kind,
                               const py::list& predicates,
                               const std::string& column, const FloatArray& query,
                               std::size_t k, std::size_t ef) {
            HybridTrace trace;
            auto matches = self.hybrid_with(kind, predicates_from_py(predicates),
                                            column, query_vector(query), k, ef,
                                            &trace);
            return py::make_tuple(std::move(matches), trace);
        }, py::arg("plan"), py::arg("predicates"), py::arg("column"),
           py::arg("query"), py::arg("k") = 10, py::arg("ef") = 0,
           "Force a plan. Exists so a test can assert every legal plan returns\n"
           "the same rows -- without it the planner could pick a fast *wrong*\n"
           "plan and nothing would notice.")
        .def("explain_hybrid", [](const Table& self, const py::list& predicates,
                                  const std::string& column, std::size_t k) {
            return self.explain_hybrid(predicates_from_py(predicates), column, k);
        }, py::arg("predicates"), py::arg("column"), py::arg("k") = 10,
           "Decide, without doing any vector work.\n\n"
           "On a bitmap column aligned to the vector rows this costs a\n"
           "popcount; on any other it costs the whole structured query,\n"
           "because executing the predicate is how this planner knows its\n"
           "selectivity. The trace says which one happened.")
        .def("calibrate", [](Table& self, const std::string& column,
                             const FloatArray& queries, std::size_t k,
                             std::size_t ef, std::size_t samples) {
            if (queries.ndim() != 2) {
                throw std::invalid_argument(
                    "calibrate: expected an (n, dim) array of queries, got " +
                    std::to_string(queries.ndim()) + "-D");
            }
            const std::vector<float> flat(
                queries.data(), queries.data() + queries.shape(0) * queries.shape(1));
            return self.calibrate(column, flat, k, ef, samples);
        }, py::arg("column"), py::arg("queries"), py::arg("k") = 10,
           py::arg("ef") = 0, py::arg("samples") = 12,
           "Measure this corpus's own pre-filter crossover and adopt it.\n\n"
           "The 50% default came from one measurement on one corpus at one ef,\n"
           "and the crossover is not a constant of the algorithm: it moves with\n"
           "n, with the dimensionality, with ef, and with how fast this\n"
           "machine's cache is. No scalar column is involved -- the synthetic\n"
           "filters are cut from the vector column's own rows.")
        .def("plan_available", [](const Table& self, PlanKind kind,
                                  const py::list& predicates,
                                  const std::string& column) {
            return self.plan_available(kind, predicates_from_py(predicates), column);
        }, py::arg("plan"), py::arg("predicates"), py::arg("column"))
        .def_property("prefilter_threshold", &Table::prefilter_threshold,
                      &Table::set_prefilter_threshold,
                      "Selectivity at or below which a filtered exhaustive scan\n"
                      "is preferred. A single measured constant, and that is the\n"
                      "point: it is the honest baseline a learned cost model has\n"
                      "to beat.")

        // --- persistence ---
        .def("save", &Table::save,
             "Write the schema, the vectors and the catalog, atomically.")
        .def("save_vectors", &Table::save_vectors,
             "Write the embeddings and the metadata that interprets them.\n\n"
             "Separate from save() because it is the one durability point\n"
             "vectors have: they never reach the write-ahead log. What lands\n"
             "on disk is the compacted form -- live rows only -- whether or\n"
             "not memory has caught up, so a reopen is always clean.")
        .def("checkpoint", &Table::checkpoint,
             "Flush pending rebuilds, write every derived file, then snapshot\n"
             "the store and truncate the WAL.\n\n"
             "The derived files go first, which is the reverse of the obvious\n"
             "order and is deliberate: the snapshot truncates the log, and the\n"
             "embeddings are the one thing here with no log to replay from.")
        .def("freshness", &Table::freshness, py::arg("column"))
        .def("rebuild", &Table::rebuild, py::arg("column"))
        .def("rebuilds", &Table::rebuilds,
             "How often a write forced a whole-column rebuild, cumulatively.")
        .def("validate", &Table::validate,
             "Every index agrees with the store, for every indexed column.\n"
             "Spans the store, the indexes and the schema together, where each\n"
             "component's own validate() can only speak for itself.")
        .def("__len__", &Table::size)
        .def("__repr__", [](const Table& t) {
            return "Table(" + std::to_string(t.size()) + " records, " +
                   std::to_string(t.schema().size()) + " columns)";
        });
}
