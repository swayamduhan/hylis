// bindings/btree.cpp
//
// pybind11 binding for the B+ tree, importable as `hylis._btree`.
//
// Only the default int64 -> int64 instantiation is exposed: keys are record
// primary keys and values are record ids, which is what the query planner
// works with. The template stays generic on the C++ side for the tests.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>

#include "index/btree.hpp"

namespace py = pybind11;
using hylis::index::BPlusTree;
using hylis::index::CompareOp;

using Tree = BPlusTree<std::int64_t, std::int64_t>;

PYBIND11_MODULE(_btree, m) {
    m.doc() = "hylis B+ tree core (C++): ordered index over int64 keys";

    py::enum_<CompareOp>(m, "CompareOp",
        "Predicate for range_query. Shared with the learned index so the "
        "query planner can treat either the same way.")
        .value("Eq", CompareOp::Eq, "key == value")
        .value("Lt", CompareOp::Lt, "key <  value")
        .value("Le", CompareOp::Le, "key <= value")
        .value("Gt", CompareOp::Gt, "key >  value")
        .value("Ge", CompareOp::Ge, "key >= value");

    py::class_<Tree>(m, "BPlusTree",
        "B+ tree mapping int64 keys to int64 values, with ordered range scans.")
        .def(py::init<std::size_t>(), py::arg("order") = 32,
             "Create a tree with the given branching order (max children per "
             "internal node). Must be >= 3.")
        .def("insert", &Tree::insert, py::arg("key"), py::arg("value"),
             "Insert or overwrite. Returns True if the key was new.")
        .def("find", [](const Tree& self, std::int64_t key) -> py::object {
             const std::int64_t* v = self.find(key);
             return v ? py::cast(*v) : py::none();
        }, py::arg("key"), "Look up a key. Returns the value or None.")
        .def("erase", &Tree::erase, py::arg("key"),
             "Remove a key. Returns True if it was present.")
        .def("contains", &Tree::contains, py::arg("key"), "True if the key exists.")
        .def("range", &Tree::range, py::arg("lo"), py::arg("hi"),
             "Values for all keys in [lo, hi], ascending by key.")
        .def("range_query", &Tree::range_query, py::arg("op"), py::arg("value"),
             "Values matching `op` against `value`, ascending by key.")
        .def("keys", &Tree::keys, "All keys, ascending.")
        .def("items", &Tree::items, "All (key, value) pairs, ascending by key.")
        .def("height", &Tree::height, "Height in nodes; a leaf-only tree is 1.")
        .def("order", &Tree::order, "The branching order this tree was built with.")
        .def("clear", &Tree::clear, "Remove all entries.")
        .def("validate", &Tree::validate,
             "Check all B+ tree invariants. Raises RuntimeError describing the "
             "first violation found.")
        .def("__len__", &Tree::size, "Number of entries.")
        .def("__contains__", &Tree::contains)
        .def("__repr__", [](const Tree& self) {
            return "BPlusTree(order=" + std::to_string(self.order()) +
                   ", size=" + std::to_string(self.size()) +
                   ", height=" + std::to_string(self.height()) + ")";
        });
}
