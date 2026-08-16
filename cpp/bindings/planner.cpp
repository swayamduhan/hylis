// bindings/planner.cpp
//
// pybind11 binding for the hybrid query planner, importable as
// `hylis._planner`.
//
// This module holds *borrowed* pointers to a FlatIndex and an HnswIndex that
// Python owns, so every setter uses keep_alive to tie their lifetimes to the
// planner's. Without it, `planner.set_graph(HnswIndex(...))` would bind a
// temporary, the temporary would be collected, and the planner would search a
// freed graph — a crash that would look like an index bug rather than a
// binding one.
//
// hylis._flat and hylis._hnsw are imported at module init so their types are
// present in pybind11's registry before anything tries to cast one. They are
// separate extension modules, and the registry is only populated by import.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "index/column_index.hpp"
#include "query/planner.hpp"

namespace py = pybind11;

using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::CompareOp;
using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Neighbor;
using hylis::query::HybridPlanner;
using hylis::query::PlanKind;
using hylis::query::Predicate;
using hylis::query::QueryPlan;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

const float* as_vector(const FloatArray& arr, const char* what) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(
            std::string(what) + ": expected a 1-D array, got " +
            std::to_string(arr.ndim()) + "-D");
    }
    return arr.data();
}

}  // namespace

PYBIND11_MODULE(_planner, m) {
    m.doc() =
        "hylis hybrid query planner (C++): one query, two indexes.\n\n"
        "Joins a structured predicate to a vector similarity search. The\n"
        "structured index returns row ids; the vector index takes row ids as\n"
        "its filter. This picks how to combine them.";
    hylis::bindings::attach_build_info(m);

    // Bring the vector index types into the registry. Without these imports
    // the first set_exact() would fail with an unregistered-type error.
    py::module_::import("hylis._btree");  // CompareOp lives here
    py::module_::import("hylis._rmi");    // IndexKind lives here
    py::module_::import("hylis._flat");
    py::module_::import("hylis._hnsw");

    py::enum_<PlanKind>(m, "PlanKind", "How a hybrid query is executed.")
        .value("NoPredicate", PlanKind::NoPredicate,
               "Unconstrained similarity search.")
        .value("PreFilter", PlanKind::PreFilter,
               "Run the predicate, then scan only the survivors exactly.\n"
               "O(|allowed|), so it gets cheaper as a predicate tightens.")
        .value("FilteredGraph", PlanKind::FilteredGraph,
               "Run the predicate, then beam-search the graph rejecting\n"
               "non-matches. Gets more expensive as a predicate tightens,\n"
               "because the graph must step through rejects to stay connected.")
        .value("BitmapFilteredGraph", PlanKind::BitmapFilteredGraph,
               "The same traversal, with a bitmap column's bit set used\n"
               "directly as the mask. Saves decoding every matching row id\n"
               "into a list and then stamping it -- O(matches) of setup per\n"
               "query -- not the membership test, which was already O(1).")
        .value("PostFilter", PlanKind::PostFilter,
               "Search unfiltered, then drop non-matches.\n\n"
               "Included because it is the trap, not because it wins: it is\n"
               "what a system without a planner does, and it can silently\n"
               "return fewer than k rows.");

    // Predicate is bound by hylis._table, which owns query/predicate.hpp's
    // type. Registering it twice would be a runtime conflict, and the two were
    // separate structs until phase D unified them -- which is what let the
    // planner and the table finally appear in one program.
    py::module_::import("hylis._table");

    py::class_<QueryPlan>(m, "QueryPlan",
        "The decision, and the evidence behind it.")
        .def_readonly("kind", &QueryPlan::kind)
        .def_readonly("matched_rows", &QueryPlan::matched_rows,
                      "Rows satisfying the predicate. Exact, not estimated --\n"
                      "the planner runs the predicate before deciding.")
        .def_readonly("corpus_rows", &QueryPlan::corpus_rows)
        .def_readonly("selectivity", &QueryPlan::selectivity)
        .def_readonly("threshold", &QueryPlan::threshold)
        .def_readonly("selectivity_was_free", &QueryPlan::selectivity_was_free,
                      "Whether the selectivity was known without executing the\n"
                      "predicate. True only for a bitmap column aligned to the\n"
                      "corpus, which answers by popcount -- the one case where\n"
                      "this planner's execute-then-decide weakness does not\n"
                      "apply.")
        .def_readonly("reason", &QueryPlan::reason,
                      "Why this plan, in words. A planner that cannot say why\n"
                      "is not defensible in a report.")
        .def("__repr__", [](const QueryPlan& p) {
            return std::string("QueryPlan(") + hylis::query::to_string(p.kind) +
                   ", matched=" + std::to_string(p.matched_rows) + "/" +
                   std::to_string(p.corpus_rows) + ")";
        });

    py::class_<HybridPlanner>(m, "HybridPlanner",
        "Routes a hybrid query to the cheapest correct plan.\n\n"
        "The decision rule is measured, not assumed: a filtered exact scan is\n"
        "O(|allowed|) and gets cheaper as a predicate tightens, while a\n"
        "filtered graph search must step through non-matching nodes and gets\n"
        "more expensive. They cross at ~50% selectivity, and below it the\n"
        "scan wins by up to 50x.\n\n"
        "Every plan returns the same rows. The planner chooses between costs,\n"
        "never between answers.")
        .def(py::init<double>(), py::arg("prefilter_threshold") = 0.5,
             "prefilter_threshold is the selectivity at or below which an\n"
             "exact scan is preferred. It defaults to the measured crossover,\n"
             "and being one honest constant is the point: it is the baseline\n"
             "a learned cost model would have to beat.")
        .def_property("prefilter_threshold", &HybridPlanner::prefilter_threshold,
                      &HybridPlanner::set_prefilter_threshold)
        // ColumnIndex owns unique_ptrs and is move-only, so it cannot cross
        // the binding boundary by value. Rather than add a shared_ptr holder
        // purely to satisfy Python, the column is built *inside* the planner
        // from the keys and values — which is also the clearer API, because
        // there is then only one owner and no question of what happens to the
        // caller's copy.
        .def("set_column", [](HybridPlanner& self, const std::string& name,
                              const std::vector<ColumnKey>& keys,
                              const std::vector<hylis::index::ColumnValue>& values,
                              double write_fraction) {
            hylis::index::Workload workload;
            workload.write_fraction = write_fraction;
            self.set_column(name, ColumnIndex::build(
                keys, values, std::numeric_limits<std::size_t>::max(), workload));
        }, py::arg("name"), py::arg("keys"), py::arg("values"),
           py::arg("write_fraction") = 0.0,
           "Attach a column, auto-tuning its index.\n\n"
           "keys must be strictly ascending; values are the row ids a\n"
           "predicate over this column returns, which is what the vector\n"
           "index takes as its filter. That correspondence is the join this\n"
           "whole module exists to make.")
        .def("set_column_kind", [](HybridPlanner& self, const std::string& name,
                                   const std::vector<ColumnKey>& keys,
                                   const std::vector<hylis::index::ColumnValue>& values,
                                   hylis::index::IndexKind kind,
                                   std::size_t rmi_models) {
            hylis::index::IndexPlan plan;
            plan.kind = kind;
            plan.rmi_models = rmi_models;
            self.set_column(name, ColumnIndex::build_with(keys, values, plan));
        }, py::arg("name"), py::arg("keys"), py::arg("values"), py::arg("kind"),
           py::arg("rmi_models") = 1024,
           "Attach a column with the structure named rather than measured.\n"
           "For showing that the planner's answers do not depend on which\n"
           "index answered.")
        .def("set_column_index", [](HybridPlanner& self, const std::string& name,
                                    hylis::index::LogicalType type,
                                    const std::vector<ColumnKey>& keys,
                                    const std::vector<hylis::index::ColumnValue>& values,
                                    const hylis::index::IndexPlan& plan,
                                    py::object row_space) {
            std::vector<hylis::index::ColumnValue> space;
            const std::vector<hylis::index::ColumnValue>* space_ptr = nullptr;
            if (!row_space.is_none()) {
                space = row_space.cast<std::vector<hylis::index::ColumnValue>>();
                space_ptr = &space;
            }
            self.set_column(name, ColumnIndex::build_typed_with(type, keys, values,
                                                                plan, space_ptr));
        }, py::arg("name"), py::arg("type"), py::arg("keys"), py::arg("values"),
           py::arg("plan"), py::arg("row_space") = py::none(),
           "Attach a column built to an exact plan.\n\n"
           "`row_space` is every row the corpus holds, and a bitmap column\n"
           "needs it: bit position i must mean row id i for the graph to take\n"
           "the bit set as a mask, which is only true when the bitmap covers\n"
           "the whole corpus densely. Passing None leaves the column covering\n"
           "only the rows that carry a value, which is correct but cannot be\n"
           "used as a mask.")
        .def("plan_available", &HybridPlanner::plan_available,
             py::arg("kind"), py::arg("predicate"),
             "Whether a plan can run for this predicate at all.\n"
             "BitmapFilteredGraph needs a bitmap column aligned to the corpus.")
        .def("search_rows", [](const HybridPlanner& self,
                               const std::vector<hylis::index::ColumnValue>& rows,
                               py::array_t<float, py::array::c_style |
                                                  py::array::forcecast> query,
                               std::size_t k, std::size_t ef) {
            hylis::query::QueryPlan plan;
            auto out = self.search_rows(rows, query.data(), k, ef, &plan);
            return py::make_tuple(std::move(out), plan);
        }, py::arg("rows"), py::arg("query"), py::arg("k") = 10, py::arg("ef") = 0,
           "Plan and run over row ids the caller already has.\n\n"
           "The seam for everything the planner cannot express itself: a\n"
           "conjunction Table resolved, a Contains it had to scan for, or a\n"
           "hand-built filter. Same plan choice, different source of rows.")
        .def("has_column", &HybridPlanner::has_column, py::arg("name"))
        .def("columns", &HybridPlanner::columns)
        .def("set_exact", &HybridPlanner::set_exact, py::arg("index"),
             py::keep_alive<1, 2>(),
             "Attach the exact index. Borrowed, not copied -- keep_alive ties\n"
             "its lifetime to the planner's.")
        .def("set_graph", &HybridPlanner::set_graph, py::arg("index"),
             py::keep_alive<1, 2>(),
             "Attach the graph index. Borrowed; see set_exact.")
        .def("matching_rows", &HybridPlanner::matching_rows, py::arg("predicate"),
             "Row ids satisfying the predicate, ascending. The structured\n"
             "half alone, which is also the answer to a plain SQL query.")
        .def("calibrate", [](HybridPlanner& self, const std::string& column,
                              const FloatArray& queries, std::size_t k,
                              std::size_t ef, std::size_t samples) {
            if (queries.ndim() != 2) {
                throw std::invalid_argument(
                    "calibrate: expected a 2-D (n_queries, dim) array");
            }
            return self.calibrate(column, queries.data(),
                                  static_cast<std::size_t>(queries.shape(0)),
                                  k, ef, samples);
        }, py::arg("column"), py::arg("queries"), py::arg("k") = 10,
           py::arg("ef") = 0, py::arg("samples") = 12,
           "Measure this machine's actual crossover and adopt it.\n\n"
           "The ~50% default came from one corpus at one ef, and the crossover\n"
           "moves with n, dimensionality, ef and cache speed. Same answer\n"
           "choose_index() gives one level down: build both, time both, keep\n"
           "what won. Returns the new threshold.")
        .def("explain", &HybridPlanner::explain,
             py::arg("predicate"), py::arg("k") = 10,
             "Run the predicate and decide, without doing the vector work.")
        .def("search", [](const HybridPlanner& self, const Predicate& predicate,
                          const FloatArray& query, std::size_t k, std::size_t ef) {
            QueryPlan plan;
            std::vector<Neighbor> found = self.search(
                predicate, as_vector(query, "search"), k, ef, &plan);
            return py::make_tuple(std::move(found), plan);
        }, py::arg("predicate"), py::arg("query"), py::arg("k") = 10,
           py::arg("ef") = 0,
           "Answer a hybrid query. Returns (neighbours, plan) so the caller\n"
           "can see which plan ran without asking twice.")
        .def("search_with", [](const HybridPlanner& self, PlanKind kind,
                               const Predicate& predicate, const FloatArray& query,
                               std::size_t k, std::size_t ef) {
            return self.search_with(kind, predicate,
                                    as_vector(query, "search_with"), k, ef);
        }, py::arg("plan"), py::arg("predicate"), py::arg("query"),
           py::arg("k") = 10, py::arg("ef") = 0,
           "Force a plan.\n\n"
           "Exists so tests can assert every plan returns the same rows.\n"
           "Without it the planner could pick a fast *wrong* plan and nothing\n"
           "would notice, which is the one failure mode a query optimiser\n"
           "must never have.")
        .def("__repr__", [](const HybridPlanner& self) {
            return "HybridPlanner(threshold=" +
                   std::to_string(self.prefilter_threshold()) + ", columns=" +
                   std::to_string(self.columns().size()) + ")";
        });
}
