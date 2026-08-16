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
using hylis::index::Workload;
using hylis::query::ColumnInfo;
using hylis::query::PredOp;
using hylis::query::Predicate;
using hylis::query::QueryTrace;
using hylis::query::Schema;
using hylis::query::Table;
using hylis::query::WriteResult;
using hylis::storage::Record;
using hylis::storage::RecordStore;

namespace {

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

    py::enum_<PredOp>(m, "PredOp", "What a predicate asks of a column.")
        .value("Eq", PredOp::Eq)
        .value("Lt", PredOp::Lt)
        .value("Le", PredOp::Le)
        .value("Gt", PredOp::Gt)
        .value("Ge", PredOp::Ge)
        .value("Between", PredOp::Between, "Inclusive at both ends.")
        .value("Prefix", PredOp::Prefix,
               "String columns. One descent and a leaf walk, and impossible\n"
               "under any integer encoding of the string.")
        .value("Contains", PredOp::Contains,
               "String columns. No index here can serve an infix match, so\n"
               "this is executed by a full scan and the trace says so.")
        .value("IsNull", PredOp::IsNull,
               "Rows with no value for the column, which by construction\n"
               "appear in no index at all.");

    m.def("op_is_indexable", &hylis::query::op_is_indexable, py::arg("op"),
          "Whether an ordered index can answer this operator directly.");

    // Bound here rather than in _planner because query/predicate.hpp is the
    // query layer's shared type: the planner and the table used to have one
    // Predicate each, which meant they could never appear in the same program.
    py::class_<Predicate>(m, "Predicate", "One structured constraint.")
        .def(py::init([](std::string column, PredOp op, py::handle value,
                         py::handle value2) {
                 return predicate_from_py(std::move(column), op, value, value2);
             }),
             py::arg("column"), py::arg("op"), py::arg("value") = py::none(),
             py::arg("value2") = py::none())
        .def_readonly("column", &Predicate::column)
        .def_readonly("op", &Predicate::op)
        .def("__repr__", [](const Predicate& p) {
            return "Predicate(" + p.column + " " +
                   std::string(hylis::query::to_string(p.op)) + ")";
        });

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

        // --- persistence ---
        .def("save", &Table::save, "Write the schema and the catalog, atomically.")
        .def("checkpoint", &Table::checkpoint,
             "Flush pending rebuilds, snapshot the store, truncate the WAL,\n"
             "and save the schema and catalog.")
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
