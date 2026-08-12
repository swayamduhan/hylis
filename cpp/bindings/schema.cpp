// bindings/schema.cpp
//
// pybind11 binding for the typed column layer, importable as `hylis._schema`.
//
// Exposes ColumnDef and Schema, plus the value parse/format pair.
//
// LogicalType itself is registered by `hylis._rmi`, because that is where
// ColumnIndex lives and a type used in two modules must be defined in exactly
// one. This module imports _rmi at load so registration order is guaranteed
// rather than left to whichever import a caller happens to write first.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "index/logical_type.hpp"
#include "query/schema.hpp"
#include "storage/record.hpp"

namespace py = pybind11;

using hylis::index::Datum;
using hylis::index::LogicalType;
using hylis::query::ColumnDef;
using hylis::query::Schema;
using hylis::storage::Record;

namespace {

py::object datum_to_py(const Datum& value) {
    switch (value.index()) {
        case 0: return py::cast(std::get<std::int64_t>(value));
        case 1: return py::cast(std::get<double>(value));
        case 2: return py::cast(std::get<std::string>(value));
        default: return py::cast(std::get<bool>(value));
    }
}

Datum datum_from_py(py::handle value) {
    // bool before int: in Python bool is a subclass of int, so the natural
    // order would silently retype a Bool column's value into an Int64 one.
    if (py::isinstance<py::bool_>(value)) return Datum{value.cast<bool>()};
    if (py::isinstance<py::int_>(value)) return Datum{value.cast<std::int64_t>()};
    if (py::isinstance<py::float_>(value)) return Datum{value.cast<double>()};
    if (py::isinstance<py::str>(value)) return Datum{value.cast<std::string>()};
    throw std::invalid_argument(
        "a column value must be an int, float, str or bool; got " +
        std::string(py::str(py::type::of(value))));
}

}  // namespace

PYBIND11_MODULE(_schema, m) {
    m.doc() =
        "hylis typed column layer (C++): logical types, column defs, schema";
    hylis::bindings::attach_build_info(m);

    // LogicalType is registered here. Importing it explicitly rather than
    // relying on the caller means `import hylis._schema` alone is enough.
    py::module_::import("hylis._rmi");

    m.def("parse_value", [](LogicalType type, const std::string& text) {
              return datum_to_py(hylis::index::parse_datum(type, text));
          }, py::arg("type"), py::arg("text"),
          "Parse text into a typed value, or raise.\n\n"
          "Trailing junk is a failure rather than a value: accepting '12abc'\n"
          "as 12 would turn a data-entry typo into a silently wrong index.");

    m.def("format_value", [](LogicalType type, py::handle value) {
              return hylis::index::format_datum(type, datum_from_py(value));
          }, py::arg("type"), py::arg("value"),
          "Render a typed value back to the text the record store holds.\n"
          "Timestamps normalise to ISO-8601 with a Z, so a checkpoint is\n"
          "canonical whichever way the value arrived.");

    m.def("prefix_upper_bound", [](const std::string& p) -> py::object {
              std::string upper;
              if (!hylis::index::prefix_upper_bound(p, &upper)) return py::none();
              return py::bytes(upper);
          }, py::arg("prefix"),
          "The exclusive upper bound of a prefix range, or None when the\n"
          "prefix has no successor (empty, or all 0xFF bytes) and the caller\n"
          "must fall back to an unbounded >= scan.\n\n"
          "Works on **bytes**, and returns bytes, because that is the ordering\n"
          "the B+ tree uses -- a str argument is taken as its UTF-8 encoding.\n"
          "Incrementing the last byte of a multi-byte character can produce a\n"
          "sequence that is not valid UTF-8, which is correct as a bound and\n"
          "would raise if it were handed back as a str.");

    py::class_<ColumnDef>(m, "ColumnDef", "One declared column.")
        .def(py::init<std::string, LogicalType, std::size_t>(),
             py::arg("name"), py::arg("type"), py::arg("dim") = 0)
        .def_readwrite("name", &ColumnDef::name)
        .def_readwrite("type", &ColumnDef::type)
        .def_readwrite("dim", &ColumnDef::dim, "Vector columns only.")
        .def("__repr__", [](const ColumnDef& c) {
            std::string out = "ColumnDef('" + c.name + "', " +
                              hylis::index::to_string(c.type);
            if (c.dim) out += ", dim=" + std::to_string(c.dim);
            return out + ")";
        });

    py::class_<Schema>(m, "Schema",
        "What a table's columns are, and the enforcement that makes this a\n"
        "schema rather than a convention.\n\n"
        "Two rules, asymmetric on purpose:\n"
        "  * an unknown column is an error -- catching the typo is the point;\n"
        "  * a missing column is not -- the row is simply absent from that\n"
        "    column's index and matches no predicate on it.")
        .def(py::init<>())
        .def(py::init<std::vector<ColumnDef>>(), py::arg("columns"))
        .def("add", &Schema::add, py::arg("column"),
             "Declare a column. Adding is the only schema evolution supported:\n"
             "dropping or retyping would invalidate every index and stored plan\n"
             "over the column.")
        .def("has", &Schema::has, py::arg("name"))
        .def("column", &Schema::column, py::arg("name"),
             py::return_value_policy::copy)
        .def("type_of", &Schema::type_of, py::arg("name"))
        .def("columns", &Schema::columns)
        .def("scalar_columns", &Schema::scalar_columns)
        .def("vector_columns", &Schema::vector_columns)
        .def("parse", [](const Schema& self, const std::string& name,
                         const std::string& text) {
            return datum_to_py(self.parse(name, text));
        }, py::arg("column"), py::arg("text"))
        .def("format", [](const Schema& self, const std::string& name,
                          py::handle value) {
            return self.format(name, datum_from_py(value));
        }, py::arg("column"), py::arg("value"))
        .def("validate", &Schema::validate, py::arg("record"),
             "Raise if a record does not fit the schema, naming the column.\n"
             "A record that half-loads is worse than one that is refused.")
        .def("accepts", &Schema::accepts, py::arg("record"),
             "Whether a record would be accepted, without the message. For\n"
             "callers counting rejects across a bulk load.")
        .def("serialize", &Schema::serialize)
        .def_static("parse_json", &Schema::parse_json, py::arg("blob"))
        .def("__len__", &Schema::size)
        .def("__contains__", &Schema::has)
        .def("__repr__", [](const Schema& s) {
            return "Schema(" + std::to_string(s.size()) + " columns)";
        });
}
