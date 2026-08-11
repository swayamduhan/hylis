// bindings/rmi.cpp
//
// pybind11 binding for the learned index, importable as `hylis._rmi`.
//
// Exposes three layers:
//   RMIndex      the learned index itself
//   ColumnIndex  auto-tuned per-column choice between a tree and a model
//   IndexCatalog which columns chose what, persisted
//
// Lookup timings come from measure_plan() on the C++ side rather than being
// timed from Python. A per-call bridge crossing costs on the order of a
// microsecond while a lookup costs single-digit nanoseconds, so timing it
// through pybind11 would measure pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "index/column_index.hpp"
#include "index/index_catalog.hpp"
#include "index/rmi.hpp"

namespace py = pybind11;

using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::ColumnValue;
using hylis::index::CompareOp;
using hylis::index::IndexCatalog;
using hylis::index::IndexKind;
using hylis::index::IndexPlan;
using hylis::index::LinearModel;

using Learned = hylis::index::RMIndex<std::int64_t, std::int64_t>;

PYBIND11_MODULE(_rmi, m) {
    m.doc() = "hylis learned index core (C++): two-stage RMI, per-column selection";
    hylis::bindings::attach_build_info(m);

    // ---------------------------------------------------------------- model

    py::class_<LinearModel>(m, "LinearModel",
        "y = slope*x + intercept, fitted by least squares.\n\n"
        "There is no training loop and no neural network anywhere in the RMI:\n"
        "both stages are closed-form fits computed in a single pass.")
        .def(py::init<double, double>(), py::arg("slope"), py::arg("intercept"))
        .def_static("fit", [](const std::vector<double>& xs,
                              const std::vector<double>& ys) {
            if (xs.size() != ys.size()) {
                throw std::invalid_argument("fit: xs and ys must be the same length");
            }
            return LinearModel::fit(xs.data(), ys.data(), xs.size());
        }, py::arg("xs"), py::arg("ys"), "Least-squares fit of ys onto xs.")
        .def("predict", &LinearModel::predict, py::arg("x"))
        .def_property_readonly("slope", &LinearModel::slope)
        .def_property_readonly("intercept", &LinearModel::intercept)
        .def("__repr__", [](const LinearModel& lm) {
            return "LinearModel(slope=" + std::to_string(lm.slope()) +
                   ", intercept=" + std::to_string(lm.intercept()) + ")";
        });

    // ---------------------------------------------------------------- stats

    py::class_<Learned::Stats>(m, "RMIStats",
        "What a built index cost and how well it fitted.")
        .def_readonly("size", &Learned::Stats::size)
        .def_readonly("models", &Learned::Stats::models)
        .def_readonly("empty_models", &Learned::Stats::empty_models,
                      "Second-stage models routed no keys at all. A large\n"
                      "fraction means the model budget is being wasted.")
        .def_readonly("max_error", &Learned::Stats::max_error,
                      "Worst |predicted - actual| position, in records.")
        .def_readonly("mean_error", &Learned::Stats::mean_error,
                      "Average prediction error; what lookup cost tracks.")
        .def_readonly("max_window", &Learned::Stats::max_window,
                      "Widest search interval any lookup can be forced into.")
        .def_readonly("model_bytes", &Learned::Stats::model_bytes,
                      "The index overhead proper — independent of key count.")
        .def_readonly("total_bytes", &Learned::Stats::total_bytes)
        .def("__repr__", [](const Learned::Stats& s) {
            return "RMIStats(size=" + std::to_string(s.size) +
                   ", models=" + std::to_string(s.models) +
                   ", max_error=" + std::to_string(s.max_error) +
                   ", mean_error=" + std::to_string(s.mean_error) + ")";
        });

    // -------------------------------------------------------------- RMIndex

    py::class_<Learned>(m, "RMIndex",
        "Two-stage Recursive Model Index over sorted int64 keys.\n\n"
        "Exact, not approximate: each second-stage model records the worst\n"
        "prediction error it made at build time, so a lookup searches only a\n"
        "provably sufficient window. A bad model costs time, never accuracy.\n\n"
        "Immutable — build() only, no insert or erase. That is the real\n"
        "tradeoff against a B+ tree.")
        .def(py::init<std::size_t, std::size_t>(),
             py::arg("models") = 1024, py::arg("search_threshold") = 64,
             "models is M, the second-stage size — the knob that buys accuracy\n"
             "on a curved distribution. search_threshold is where the final\n"
             "search switches from a linear scan to a binary search.")
        .def("build", &Learned::build, py::arg("keys"), py::arg("values"),
             "Fit the index. Keys must be strictly ascending; raises\n"
             "ValueError otherwise.")
        .def("find", [](const Learned& self, std::int64_t key) -> py::object {
            const std::int64_t* v = self.find(key);
            return v ? py::cast(*v) : py::none();
        }, py::arg("key"), "Look up a key. Returns the value or None.")
        .def("contains", &Learned::contains, py::arg("key"))
        .def("probes", &Learned::probes, py::arg("key"),
             "How many key comparisons a find() would perform. Lets the\n"
             "graceful-degradation claim be checked rather than trusted.")
        .def("range", &Learned::range, py::arg("lo"), py::arg("hi"),
             "Values for all keys in [lo, hi], ascending.")
        .def("range_query", &Learned::range_query, py::arg("op"), py::arg("value"),
             "Values matching `op`, ascending by key. Same interface the B+\n"
             "tree exposes, so a query planner can hold either.")
        .def("keys", &Learned::keys, "All keys, ascending.")
        .def("items", &Learned::items, "All (key, value) pairs, ascending.")
        .def("stats", &Learned::stats)
        .def("clear", &Learned::clear)
        .def("validate", &Learned::validate,
             "Replay every key through the real lookup path and confirm its\n"
             "position lies inside the predicted window. Raises RuntimeError\n"
             "if any bound is wrong.")
        .def_property_readonly("models", &Learned::model_count)
        .def_property_readonly("search_threshold", &Learned::search_threshold)
        .def("__len__", &Learned::size)
        .def("__contains__", &Learned::contains)
        .def("__repr__", [](const Learned& self) {
            return "RMIndex(models=" + std::to_string(self.model_count()) +
                   ", size=" + std::to_string(self.size()) +
                   ", max_error=" + std::to_string(self.stats().max_error) + ")";
        });

    // ----------------------------------------------------------- selection

    py::enum_<IndexKind>(m, "IndexKind", "Which structure a column was given.")
        .value("BPlusTree", IndexKind::BPlusTree)
        .value("RMI", IndexKind::RMI);

    py::class_<IndexPlan>(m, "IndexPlan",
        "A per-column index decision, and the measurements behind it.\n\n"
        "The measured fields are why a plan is worth persisting: reproducing\n"
        "them means rebuilding and re-timing every candidate structure.")
        .def(py::init<>())
        .def_readwrite("kind", &IndexPlan::kind)
        .def_readwrite("rmi_models", &IndexPlan::rmi_models)
        .def_readwrite("search_threshold", &IndexPlan::search_threshold)
        .def_readwrite("btree_order", &IndexPlan::btree_order)
        .def_readwrite("ns_per_lookup", &IndexPlan::ns_per_lookup,
                       "Measured on this machine at selection time, in C++ —\n"
                       "not timed across the Python bridge.")
        .def_readwrite("max_error", &IndexPlan::max_error)
        .def_readwrite("index_bytes", &IndexPlan::index_bytes)
        .def_readwrite("n_keys", &IndexPlan::n_keys)
        .def_readwrite("key_min", &IndexPlan::key_min)
        .def_readwrite("key_max", &IndexPlan::key_max)
        .def("matches", &IndexPlan::matches, py::arg("keys"),
             "Whether this plan is still evidence about the given column.\n"
             "A mismatch only means the choice may now be suboptimal —\n"
             "correctness never depends on it.")
        .def("__repr__", [](const IndexPlan& p) {
            return std::string("IndexPlan(kind=") + hylis::index::to_string(p.kind) +
                   ", rmi_models=" + std::to_string(p.rmi_models) +
                   ", ns_per_lookup=" + std::to_string(p.ns_per_lookup) +
                   ", index_bytes=" + std::to_string(p.index_bytes) + ")";
        });

    m.def("choose_index", &hylis::index::choose_index,
          py::arg("keys"), py::arg("values"),
          py::arg("size_budget") = std::numeric_limits<std::size_t>::max(),
          "Build every candidate structure, time real lookups on each, and\n"
          "return the plan for the fastest that fits the budget.\n\n"
          "Deliberately empirical: an analytic cost model over assumed\n"
          "cache-miss constants would need retuning per machine and could not\n"
          "be checked. Costs seconds once per column.");

    m.def("measure_plan", &hylis::index::measure_plan,
          py::arg("keys"), py::arg("values"), py::arg("plan"),
          "Build one specific plan and time it, returning the plan with its\n"
          "measured fields filled in. The honest way to benchmark a single\n"
          "structure from Python.");

    py::class_<ColumnIndex>(m, "ColumnIndex",
        "A column's index, with the choice of structure hidden.\n\n"
        "Presents the same find/range_query pair whether a B+ tree or a\n"
        "learned index is inside, so a query planner can ask for rows\n"
        "matching a predicate without learning what answered.")
        .def_static("build", &ColumnIndex::build,
                    py::arg("keys"), py::arg("values"),
                    py::arg("size_budget") = std::numeric_limits<std::size_t>::max(),
                    "Auto-tune: measure every candidate and keep the best.")
        .def_static("build_with", &ColumnIndex::build_with,
                    py::arg("keys"), py::arg("values"), py::arg("plan"),
                    "Replay a decision made earlier, skipping the measurement.")
        .def("find", [](const ColumnIndex& self, std::int64_t key) -> py::object {
            const std::int64_t* v = self.find(key);
            return v ? py::cast(*v) : py::none();
        }, py::arg("key"))
        .def("contains", &ColumnIndex::contains, py::arg("key"))
        .def("range", &ColumnIndex::range, py::arg("lo"), py::arg("hi"))
        .def("range_query", &ColumnIndex::range_query, py::arg("op"), py::arg("value"))
        .def("plan", &ColumnIndex::plan)
        .def("validate", &ColumnIndex::validate)
        .def_property_readonly("kind", &ColumnIndex::kind)
        .def("__len__", &ColumnIndex::size)
        .def("__contains__", &ColumnIndex::contains)
        .def("__repr__", [](const ColumnIndex& self) {
            return std::string("ColumnIndex(kind=") +
                   hylis::index::to_string(self.kind()) +
                   ", size=" + std::to_string(self.size()) + ")";
        });

    // ------------------------------------------------------------- catalog

    py::class_<IndexCatalog> catalog(m, "IndexCatalog",
        "Which index each column got, persisted.\n\n"
        "Stores plans, not fitted models: models rebuild in milliseconds,\n"
        "while producing a plan means building and timing every candidate.\n"
        "Saved with the same atomic temp-file-and-rename the record store's\n"
        "checkpoint uses, so a crash mid-save leaves the previous catalog.");

    py::enum_<IndexCatalog::Freshness>(catalog, "Freshness")
        .value("Missing", IndexCatalog::Freshness::Missing)
        .value("Stale", IndexCatalog::Freshness::Stale)
        .value("Fresh", IndexCatalog::Freshness::Fresh);

    catalog.def(py::init<>())
        .def("set", &IndexCatalog::set, py::arg("column"), py::arg("plan"))
        .def("get", [](const IndexCatalog& self, const std::string& column) -> py::object {
            std::optional<IndexPlan> plan = self.get(column);
            return plan ? py::cast(*plan) : py::none();
        }, py::arg("column"), "The stored plan for a column, or None.")
        .def("erase", &IndexCatalog::erase, py::arg("column"))
        .def("columns", &IndexCatalog::columns)
        .def("clear", &IndexCatalog::clear)
        .def("freshness", &IndexCatalog::freshness, py::arg("column"), py::arg("keys"),
             "Whether a stored plan is still evidence about this column.")
        .def("build_column", &IndexCatalog::build_column,
             py::arg("column"), py::arg("keys"), py::arg("values"),
             "Replay the stored plan when it still applies, re-tune when it\n"
             "does not, and record whatever was actually used.")
        .def("serialize", &IndexCatalog::serialize)
        .def_static("parse", &IndexCatalog::parse, py::arg("blob"))
        .def("save", &IndexCatalog::save, py::arg("path"))
        .def_static("load", &IndexCatalog::load, py::arg("path"),
                    "Returns an empty catalog if the file is absent; raises\n"
                    "RuntimeError if it exists but is corrupt.")
        .def("__len__", &IndexCatalog::size)
        .def("__repr__", [](const IndexCatalog& self) {
            return "IndexCatalog(" + std::to_string(self.size()) + " columns)";
        });
}
