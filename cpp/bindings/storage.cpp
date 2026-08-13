// bindings/storage.cpp
//
// pybind11 binding for the storage layer.
// Produces an importable module `hylis._storage` exposing Record and RecordStore.
//
// Design: we bind the C++ types as closely as possible. Record is exposed as
// a simple value type with .key (int) and .columns (dict[str,str]).
// RecordStore gives put/get/delete/checkpoint/close as methods.
// This keeps the Python API thin — no clever abstractions to hide behind.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "storage/record.hpp"
#include "storage/store.hpp"

namespace py = pybind11;
using namespace hylis::storage;

PYBIND11_MODULE(_storage, m) {
    m.doc() = "hylis storage core (C++): Record + RecordStore with WAL durability";

    // --- Record ---
    py::class_<Record>(m, "Record", "A single row: primary key + column map.")
        .def(py::init<std::int64_t, std::map<std::string,std::string>>(),
             py::arg("key"), py::arg("columns") = std::map<std::string,std::string>{},
             "Create a record with an integer primary key and optional column dict.")
        .def_readwrite("key", &Record::key, "Primary key.")
        .def_readwrite("columns", &Record::columns, "Column map (str -> str).")
        .def("get", [](const Record& r, const std::string& name,
                        const std::string& default_val) -> std::string {
                 return r.get(name, default_val);
             },
             py::arg("name"), py::arg("default") = std::string{},
             "Get a column value, returning default if absent.")
        // Necessary, not sugar.
        //
        // `record.columns` converts the C++ map to a *new* Python dict on every
        // read, so `record.columns["price"] = "40"` mutates a temporary and is
        // silently lost. That has already made one test vacuous. This is the
        // way to change a column in place.
        .def("set", [](Record& r, const std::string& name,
                       const std::string& value) { r.columns[name] = value; },
             py::arg("name"), py::arg("value"),
             "Set one column. Use this rather than assigning into .columns,\n"
             "which returns a copy.")
        .def("unset", [](Record& r, const std::string& name) {
                 return r.columns.erase(name) > 0;
             },
             py::arg("name"), "Remove one column. Returns True if it was there.")
        .def("__repr__", [](const Record& r) {
            return "Record(key=" + std::to_string(r.key) + ", columns="
                   + std::to_string(r.columns.size()) + " cols)";
        });

    // --- RecordStore ---
    py::class_<RecordStore>(m, "RecordStore",
        "In-memory record store with write-ahead log durability.")
        .def(py::init<const std::string&, bool>(),
             py::arg("directory"), py::arg("recover") = true,
             "Open (or create) a store backed by `directory`. "
             "If recover=True (default), replay the WAL on open.")
        .def("put", &RecordStore::put, py::arg("record"),
             "Insert or update a record (WAL first, then memory).")
        .def("get", [](const RecordStore& self, std::int64_t key) -> py::object {
             const Record* r = self.get(key);
             return r ? py::cast(*r) : py::none();
        }, py::arg("key"),
           "Point lookup by key. Returns Record or None.")
        // Exposed as `delete`, not `del`: `del` is a reserved keyword in
        // Python, so `store.del(k)` would be a SyntaxError at the call site.
        .def("delete", &RecordStore::del, py::arg("key"),
             "Delete by key. Returns True if a record was removed.")
        .def("contains", &RecordStore::contains, py::arg("key"),
             "True if the key exists.")
        // Returned by value, not as an iterator.
        //
        // These previously built a local vector and handed back
        // py::make_iterator over it, with keep_alive<0,1> pinning the *store*.
        // The store staying alive does nothing for the vector, which is
        // destroyed when the lambda returns — so the iterator dangled and read
        // freed memory, usually without visible symptoms. Returning the vector
        // lets pybind11's stl.h convert it to the list both docstrings already
        // promised.
        .def("keys", &RecordStore::keys, "List of all primary keys (unsorted).")
        .def("records", &RecordStore::records, "All records (unsorted).")
        .def("checkpoint", &RecordStore::checkpoint,
             "Snapshot state to disk and truncate the WAL.")
        .def("close", &RecordStore::close, "Close the WAL file handle.")
        .def("__len__", &RecordStore::size, "Number of live records.")
        .def("__contains__", &RecordStore::contains);
}
