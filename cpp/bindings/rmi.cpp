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
#include "index/dynamic_rmi.hpp"
#include "index/index_catalog.hpp"
#include "index/rmi.hpp"

namespace py = pybind11;

using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::ColumnShape;
using hylis::index::ColumnValue;
using hylis::index::CompareOp;
using hylis::index::Datum;
using hylis::index::IndexCatalog;
using hylis::index::IndexKind;
using hylis::index::KeyEncoding;
using hylis::index::LogicalType;
using hylis::index::Workload;
using hylis::index::IndexPlan;
using hylis::index::LinearModel;

using Learned = hylis::index::RMIndex<std::int64_t, std::int64_t>;
using Dynamic = hylis::index::DynamicRMIndex<std::int64_t, std::int64_t>;

namespace {

// A Python scalar as a typed column value.
//
// bool is checked first because in Python bool is a subclass of int, so the
// natural order would turn every True into 1 and silently retype a Bool
// column's predicate into an Int64 one.
Datum datum_from_py(py::handle value) {
    if (py::isinstance<py::bool_>(value)) return Datum{value.cast<bool>()};
    if (py::isinstance<py::int_>(value)) return Datum{value.cast<std::int64_t>()};
    if (py::isinstance<py::float_>(value)) return Datum{value.cast<double>()};
    if (py::isinstance<py::str>(value)) return Datum{value.cast<std::string>()};
    throw std::invalid_argument(
        "a column value must be an int, float, str or bool; got " +
        std::string(py::str(py::type::of(value))));
}

// Dispatch a typed build on the logical type, which is what says how to read
// the key list. Keys stay in a concrete vector<T> rather than a vector of
// variants: a variant carrying std::string is 40 bytes, and the ingest path
// is exactly where that would be paid a million times over.
ColumnIndex build_typed_from_py(LogicalType type, const py::object& keys,
                                const std::vector<ColumnValue>& values,
                                std::size_t size_budget, double write_fraction) {
    Workload workload;
    workload.write_fraction = write_fraction;
    switch (type) {
        case LogicalType::Int64:
        case LogicalType::Timestamp:
            return ColumnIndex::build_typed(
                type, keys.cast<std::vector<std::int64_t>>(), values,
                size_budget, workload);
        case LogicalType::Double:
            return ColumnIndex::build_typed(
                type, keys.cast<std::vector<double>>(), values, size_budget,
                workload);
        case LogicalType::String:
            return ColumnIndex::build_typed(
                type, keys.cast<std::vector<std::string>>(), values, size_budget,
                workload);
        case LogicalType::Bool:
            throw std::invalid_argument(
                "a bool column has two distinct values, so an ordered index "
                "over it degenerates to a sorted list of every row in the "
                "table. It is served by a bitmap instead (module phase C).");
        case LogicalType::Vector:
            break;
    }
    throw std::invalid_argument(
        "a vector column is served by the graph indexes, not by a scalar one");
}

// Replay a typed plan without re-measuring. The dispatch is the same shape as
// build_typed_from_py; both exist because the logical type is what says how to
// read the key list.
ColumnIndex build_typed_with_from_py(LogicalType type, const py::object& keys,
                                     const std::vector<ColumnValue>& values,
                                     const IndexPlan& plan) {
    switch (type) {
        case LogicalType::Int64:
        case LogicalType::Timestamp:
            return ColumnIndex::build_typed_with(
                type, keys.cast<std::vector<std::int64_t>>(), values, plan);
        case LogicalType::Double:
            return ColumnIndex::build_typed_with(
                type, keys.cast<std::vector<double>>(), values, plan);
        case LogicalType::String:
            return ColumnIndex::build_typed_with(
                type, keys.cast<std::vector<std::string>>(), values, plan);
        default:
            break;
    }
    throw std::invalid_argument(
        "build_typed_with: only scalar columns have a plan to replay");
}

// Build one specific typed plan and time it. The comparison an experiment
// needs: choose_index reports only its winner, so measuring a structure it
// rejected has to be possible from outside.
IndexPlan measure_typed_from_py(LogicalType type, const py::object& keys,
                                const std::vector<ColumnValue>& values,
                                const IndexPlan& plan, double write_fraction) {
    Workload workload;
    workload.write_fraction = write_fraction;
    switch (type) {
        case LogicalType::Int64:
        case LogicalType::Timestamp:
            return hylis::index::measure_plan_for(
                type, keys.cast<std::vector<std::int64_t>>(), values, plan,
                workload);
        case LogicalType::Double:
            return hylis::index::measure_plan_for(
                type, keys.cast<std::vector<double>>(), values, plan, workload);
        case LogicalType::String:
            return hylis::index::measure_plan_for(
                type, keys.cast<std::vector<std::string>>(), values, plan,
                workload);
        default:
            break;
    }
    throw std::invalid_argument(
        "measure_plan_typed: only scalar columns have a plan to measure");
}

}  // namespace

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

    // ------------------------------------------------------- DynamicRMIndex

    py::class_<Dynamic::Config>(m, "DynamicConfig",
        "Tuning for a writable learned index.\n\n"
        "score_threshold defaults to infinity — the Cook's distance trigger\n"
        "off — and that is a measured default, not a placeholder. See\n"
        "scripts/experiment_merge_threshold.py.")
        .def(py::init<>())
        .def_readwrite("second_stage_size", &Dynamic::Config::second_stage_size)
        .def_readwrite("search_threshold", &Dynamic::Config::search_threshold)
        .def_readwrite("delta_order", &Dynamic::Config::delta_order)
        .def_readwrite("merge_ratio", &Dynamic::Config::merge_ratio,
                       "Merge once pending changes reach this fraction of the base.")
        .def_readwrite("score_threshold", &Dynamic::Config::score_threshold,
                       "Merge once a disturbed model's Cook's distance passes\n"
                       "this. Infinity disables it.")
        .def_readwrite("score_check_interval", &Dynamic::Config::score_check_interval)
        .def_readwrite("rebuild_error_ratio", &Dynamic::Config::rebuild_error_ratio,
                       "Refit stage 1 when mean error passes this multiple of\n"
                       "its value at the last full build.");

    py::class_<Dynamic::Stats>(m, "DynamicStats",
        "What the index holds and what maintaining it has cost.")
        .def_readonly("size", &Dynamic::Stats::size)
        .def_readonly("base_size", &Dynamic::Stats::base_size)
        .def_readonly("delta_size", &Dynamic::Stats::delta_size)
        .def_readonly("tombstones", &Dynamic::Stats::tombstones)
        .def_readonly("merges", &Dynamic::Stats::merges)
        .def_readonly("full_rebuilds", &Dynamic::Stats::full_rebuilds)
        .def_readonly("models_shifted", &Dynamic::Stats::models_shifted,
                      "Cumulative models updated in O(1) by an intercept shift.")
        .def_readonly("models_refitted", &Dynamic::Stats::models_refitted,
                      "Cumulative models that had to be refitted over their\n"
                      "whole segment. The ratio against models_shifted is how\n"
                      "localised the write pattern was.")
        .def_readonly("keys_rescanned", &Dynamic::Stats::keys_rescanned)
        .def_readonly("last_merge_seconds", &Dynamic::Stats::last_merge_seconds)
        .def_readonly("total_merge_seconds", &Dynamic::Stats::total_merge_seconds)
        .def_readonly("index_bytes", &Dynamic::Stats::index_bytes)
        .def_readonly("mean_error", &Dynamic::Stats::mean_error)
        .def_readonly("baseline_mean_error", &Dynamic::Stats::baseline_mean_error)
        .def_readonly("max_error", &Dynamic::Stats::max_error)
        .def("__repr__", [](const Dynamic::Stats& s) {
            return "DynamicStats(size=" + std::to_string(s.size) +
                   ", base=" + std::to_string(s.base_size) +
                   ", delta=" + std::to_string(s.delta_size) +
                   ", tombstones=" + std::to_string(s.tombstones) +
                   ", merges=" + std::to_string(s.merges) + ")";
        });

    py::class_<Dynamic>(m, "DynamicRMIndex",
        "A learned index you can write to.\n\n"
        "An immutable RMI over an out-of-place delta buffer, with deletions\n"
        "tombstoned rather than compacted. Tombstoning is what keeps the base\n"
        "positions fixed, which is what keeps every surviving key's error\n"
        "bound valid and makes withdrawing a key from its model's statistics\n"
        "exact and O(1) — machine unlearning in the DynaMind sense.\n\n"
        "Same find/range/range_query contract as RMIndex and BPlusTree, so a\n"
        "query planner can hold any of the three. Reads pay a delta probe and\n"
        "a tombstone check that a static RMIndex does not: this is slower\n"
        "read-only, and that is the price of being writable.")
        .def(py::init<Dynamic::Config>(), py::arg("config") = Dynamic::Config{})
        .def("build", &Dynamic::build, py::arg("keys"), py::arg("values"),
             "Fit from a sorted, unique key set. Keys must be strictly\n"
             "ascending; raises ValueError otherwise.")
        .def("insert", &Dynamic::insert, py::arg("key"), py::arg("value"),
             "False if the key is already present.")
        .def("erase", &Dynamic::erase, py::arg("key"),
             "False if the key was not present.")
        .def("find", [](const Dynamic& self, std::int64_t key) -> py::object {
            const std::int64_t* v = self.find(key);
            return v ? py::cast(*v) : py::none();
        }, py::arg("key"), "Look up a key. Returns the value or None.")
        .def("contains", &Dynamic::contains, py::arg("key"))
        .def("range", &Dynamic::range, py::arg("lo"), py::arg("hi"))
        .def("range_query", &Dynamic::range_query, py::arg("op"), py::arg("value"))
        .def("merge", &Dynamic::merge,
             "Fold the delta buffer and tombstones back into the base now.")
        .def("merge_due", &Dynamic::merge_due,
             "Whether a merge would happen if the triggers were checked now.")
        .def("score", &Dynamic::score,
             "Worst Cook's distance over the models disturbed since the last\n"
             "merge: how far the deletions have pulled them from their data.")
        .def("stats", &Dynamic::stats)
        .def("validate", &Dynamic::validate,
             "Check every invariant, including that the underlying RMI is\n"
             "still exact — every stored key inside its predicted window.\n"
             "Raises RuntimeError otherwise.")
        .def("__len__", &Dynamic::size)
        .def("__contains__", &Dynamic::contains)
        .def("__repr__", [](const Dynamic& self) {
            const auto s = self.stats();
            return "DynamicRMIndex(size=" + std::to_string(s.size) +
                   ", base=" + std::to_string(s.base_size) +
                   ", delta=" + std::to_string(s.delta_size) +
                   ", tombstones=" + std::to_string(s.tombstones) + ")";
        });

    // ----------------------------------------------------------- selection

    // --------------------------------------------------------- logical types
    //
    // What a column *is*, as opposed to how it is stored. This is a hard
    // filter on which structures are candidates: RMIndex fits models to
    // static_cast<double>(key), so a learned index over strings would be
    // fitting a model to an ordering the model itself imposed.
    py::enum_<LogicalType>(m, "LogicalType",
        "A column's logical type. Decides which index families can serve it.")
        .value("Int64", LogicalType::Int64)
        .value("Double", LogicalType::Double)
        .value("String", LogicalType::String,
               "Indexed natively by the B+ tree. Never encoded to an integer.")
        .value("Bool", LogicalType::Bool,
               "Two distinct values; an ordered index over it is pointless.")
        .value("Timestamp", LogicalType::Timestamp,
               "Physically int64 epoch millis, logically distinct: different\n"
               "parse and format, plus a near-monotonicity prior.")
        .value("Vector", LogicalType::Vector,
               "Served by the graph indexes, never by a scalar one.");

    py::enum_<KeyEncoding>(m, "KeyEncoding",
        "How a column's values become index keys.")
        .value("Native", KeyEncoding::Native, "The value itself. Unique columns.")
        .value("Composite", KeyEncoding::Composite,
               "(value, row id). What makes a duplicated column indexable.")
        .value("Position", KeyEncoding::Position,
               "Sorted rank. Reserved for a learned index over a duplicated\n"
               "numeric column; see experiment E3.")
        .value("Dictionary", KeyEncoding::Dictionary,
               "A dense code per distinct value, with one bitmap each.");

    m.def("type_supports_rmi", &hylis::index::type_supports_rmi, py::arg("type"),
          "Whether a learned index can serve this type at all. False for\n"
          "String and Bool, and that is a fact about learned indexes rather\n"
          "than a limitation of this implementation.");

    py::class_<ColumnShape>(m, "ColumnShape",
        "What one pass over a sorted column tells the candidate filter.")
        .def_readonly("rows", &ColumnShape::rows)
        .def_readonly("distinct", &ColumnShape::distinct)
        .def_readonly("unique", &ColumnShape::unique)
        .def_readonly("monotone", &ColumnShape::monotone,
                      "Row ids ascending in key order: the column arrived in\n"
                      "order, which is what makes an append-only write path legal.")
        .def_property_readonly("duplicate_fraction",
                               &ColumnShape::duplicate_fraction)
        .def("__repr__", [](const ColumnShape& s) {
            return "ColumnShape(rows=" + std::to_string(s.rows) +
                   ", distinct=" + std::to_string(s.distinct) +
                   ", unique=" + std::string(s.unique ? "True" : "False") +
                   ", monotone=" + std::string(s.monotone ? "True" : "False") + ")";
        });

    py::enum_<IndexKind>(m, "IndexKind", "Which structure a column was given.")
        .value("BPlusTree", IndexKind::BPlusTree)
        .value("RMI", IndexKind::RMI,
               "Build-only. A legal answer for a read-only column and no other.")
        .value("Bitmap", IndexKind::Bitmap,
               "Dictionary plus one bitmap per distinct value. For columns\n"
               "with few distinct values, where an ordered index's ordering\n"
               "does no work -- and the only family that answers count()\n"
               "without materialising a row id.")
        .value("DynamicRMI", IndexKind::DynamicRMI,
               "The learned index made writable: an immutable RMI over a delta\n"
               "buffer, with deletions tombstoned.");

    py::class_<IndexPlan>(m, "IndexPlan",
        "A per-column index decision, and the measurements behind it.\n\n"
        "The measured fields are why a plan is worth persisting: reproducing\n"
        "them means rebuilding and re-timing every candidate structure.")
        .def(py::init<>())
        .def_readwrite("kind", &IndexPlan::kind)
        .def_readwrite("type", &IndexPlan::type,
                       "The column's logical type. Recorded because it is what\n"
                       "decided which structures were candidates: a plan without\n"
                       "it records the answer but not the question.")
        .def_readwrite("encoding", &IndexPlan::encoding)
        .def_readwrite("distinct", &IndexPlan::distinct)
        .def_readwrite("monotone", &IndexPlan::monotone)
        .def_readwrite("rmi_models", &IndexPlan::rmi_models)
        .def_readwrite("search_threshold", &IndexPlan::search_threshold)
        .def_readwrite("btree_order", &IndexPlan::btree_order)
        .def_readwrite("merge_ratio", &IndexPlan::merge_ratio,
                       "DynamicRMI only: when to fold the delta buffer back in.")
        .def_readwrite("write_fraction", &IndexPlan::write_fraction,
                       "The write rate this plan was chosen for. A plan picked\n"
                       "for a read-only column is not evidence about one that\n"
                       "has started taking writes, whatever the keys look like.")
        .def_readwrite("ns_per_write", &IndexPlan::ns_per_write,
                       "Measured on a 50/50 insert/erase stream; 0 when the\n"
                       "column was chosen read-only.")
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

    m.def("choose_index", [](const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values,
                             std::size_t size_budget, double write_fraction) {
              Workload workload;
              workload.write_fraction = write_fraction;
              return hylis::index::choose_index(keys, values, size_budget, workload);
          },
          py::arg("keys"), py::arg("values"),
          py::arg("size_budget") = std::numeric_limits<std::size_t>::max(),
          py::arg("write_fraction") = 0.0,
          "Build every candidate structure, time real lookups on each, and\n"
          "return the plan for the fastest that fits the budget.\n\n"
          "Deliberately empirical: an analytic cost model over assumed\n"
          "cache-miss constants would need retuning per machine and could not\n"
          "be checked. Costs seconds once per column.\n\n"
          "write_fraction is not a refinement. On lookups alone the static\n"
          "RMI wins nearly everything, and it is build-only -- so without\n"
          "knowing the write rate this would hand an immutable structure to\n"
          "a column that is about to be written to. Above zero, only\n"
          "writable candidates are considered and writes are timed too.");

    m.def("measure_plan", [](const std::vector<ColumnKey>& keys,
                             const std::vector<ColumnValue>& values,
                             const IndexPlan& plan, double write_fraction) {
              Workload workload;
              workload.write_fraction = write_fraction;
              return hylis::index::measure_plan(keys, values, plan, workload);
          },
          py::arg("keys"), py::arg("values"), py::arg("plan"),
          py::arg("write_fraction") = 0.0,
          "Build one specific plan and time it, returning the plan with its\n"
          "measured fields filled in. The honest way to benchmark a single\n"
          "structure from Python.");

    py::class_<ColumnIndex>(m, "ColumnIndex",
        "A column's index, with the choice of structure hidden.\n\n"
        "Presents the same find/range_query pair whether a B+ tree or a\n"
        "learned index is inside, so a query planner can ask for rows\n"
        "matching a predicate without learning what answered.")
        .def_static("build", [](const std::vector<ColumnKey>& keys,
                                const std::vector<ColumnValue>& values,
                                std::size_t size_budget, double write_fraction) {
                        Workload workload;
                        workload.write_fraction = write_fraction;
                        return ColumnIndex::build(keys, values, size_budget, workload);
                    },
                    py::arg("keys"), py::arg("values"),
                    py::arg("size_budget") = std::numeric_limits<std::size_t>::max(),
                    py::arg("write_fraction") = 0.0,
                    "Auto-tune: measure every candidate and keep the best.")
        .def_static("build_with", &ColumnIndex::build_with,
                    py::arg("keys"), py::arg("values"), py::arg("plan"),
                    "Replay a decision made earlier, skipping the measurement.")
        .def_static("build_typed_with", &build_typed_with_from_py,
                    py::arg("type"), py::arg("keys"), py::arg("values"),
                    py::arg("plan"),
                    "Replay a typed decision made earlier, skipping the\n"
                    "measurement. What catalog replay does on reopen.")
        .def_static("build_typed", &build_typed_from_py,
                    py::arg("type"), py::arg("keys"), py::arg("values"),
                    py::arg("size_budget") = std::numeric_limits<std::size_t>::max(),
                    py::arg("write_fraction") = 0.0,
                    "Auto-tune a column of a given logical type.\n\n"
                    "`keys` must be ascending and the same length as `values`,\n"
                    "which are row ids. Repeated keys are allowed and select\n"
                    "the composite encoding; a string column is indexed as a\n"
                    "string and is never encoded to an integer.")
        .def("find", [](const ColumnIndex& self, std::int64_t key) -> py::object {
            const std::int64_t* v = self.find(key);
            return v ? py::cast(*v) : py::none();
        }, py::arg("key"))
        .def("contains", &ColumnIndex::contains, py::arg("key"))
        .def("range", &ColumnIndex::range, py::arg("lo"), py::arg("hi"))
        .def("range_query", &ColumnIndex::range_query, py::arg("op"), py::arg("value"))
        // --- typed surface ---
        .def("lookup", [](const ColumnIndex& self, py::handle value) {
            return self.lookup(datum_from_py(value));
        }, py::arg("value"),
           "Every row whose value equals this one.\n\n"
           "The general form of find(), and the only correct one for a column\n"
           "that repeats values, where find() has no single row to point at.")
        .def("query", [](const ColumnIndex& self, CompareOp op, py::handle value) {
            return self.query(op, datum_from_py(value));
        }, py::arg("op"), py::arg("value"),
           "Rows matching a predicate, with the value typed to match the column.")
        .def("query_range", [](const ColumnIndex& self, py::handle lo, py::handle hi) {
            return self.query_range(datum_from_py(lo), datum_from_py(hi));
        }, py::arg("lo"), py::arg("hi"), "Rows with values in [lo, hi].")
        .def("count", [](const ColumnIndex& self, CompareOp op, py::handle value) {
            return self.count(op, datum_from_py(value));
        }, py::arg("op"), py::arg("value"),
           "How many rows match, without producing them.\n\n"
           "A bitmap answers by popcount over n/64 words and materialises\n"
           "nothing. Every other family builds the row list and measures it,\n"
           "which for a permissive predicate is the whole cost of the query\n"
           "paid to learn one number.")
        .def_property_readonly("has_bitmap", &ColumnIndex::has_bitmap,
                               "Whether this column can hand out a bit set for\n"
                               "combining with another predicate.")
        .def_property_readonly("row_space", &ColumnIndex::row_space,
                               "Rows this index has positions for. Two bitmaps\n"
                               "are combinable only if these agree.")
        .def("query_prefix", &ColumnIndex::query_prefix, py::arg("prefix"),
             "Rows whose value begins with `prefix`. String columns only.\n\n"
             "Exact, byte-order, and a single leaf-chain walk. Impossible under\n"
             "any integer encoding of the string, which is the clearest reason\n"
             "to index strings as strings.")
        .def("insert", &ColumnIndex::insert, py::arg("key"), py::arg("value"),
             "Insert by int64 key. The pre-typed API, kept working.")
        .def("erase", &ColumnIndex::erase, py::arg("key"),
             "Erase by value alone. Legal only where one value names one row;\n"
             "a composite column raises, because picking one of several rows\n"
             "would delete one the caller never named.")
        .def("insert_row", [](ColumnIndex& self, py::handle value, std::int64_t row) {
            return self.insert_row(datum_from_py(value), row);
        }, py::arg("value"), py::arg("row"))
        .def("erase_row", [](ColumnIndex& self, py::handle value, std::int64_t row) {
            return self.erase_row(datum_from_py(value), row);
        }, py::arg("value"), py::arg("row"),
           "A composite key is (value, row), so the row id is required there:\n"
           "a value alone names several rows and picking one would delete a\n"
           "row the caller never named.")
        .def("plan", &ColumnIndex::plan)
        .def("validate", &ColumnIndex::validate)
        .def_property_readonly("kind", &ColumnIndex::kind)
        .def_property_readonly("type", &ColumnIndex::type)
        .def_property_readonly("encoding", &ColumnIndex::encoding)
        .def_property_readonly("is_mutable", &ColumnIndex::is_mutable)
        .def_property_readonly("is_native", &ColumnIndex::is_native,
                               "Whether one key names at most one row. False\n"
                               "for a composite encoding, where lookup() is the\n"
                               "answer and find() returns None by design.")
        .def("__len__", &ColumnIndex::size)
        .def("__contains__", &ColumnIndex::contains)
        .def("__repr__", [](const ColumnIndex& self) {
            return std::string("ColumnIndex(kind=") +
                   hylis::index::to_string(self.kind()) +
                   ", type=" + hylis::index::to_string(self.type()) +
                   ", encoding=" + hylis::index::to_string(self.encoding()) +
                   ", size=" + std::to_string(self.size()) + ")";
        });

    m.def("measure_plan_typed", &measure_typed_from_py,
          py::arg("type"), py::arg("keys"), py::arg("values"), py::arg("plan"),
          py::arg("write_fraction") = 0.0,
          "Build one specific typed plan and time it in C++.\n\n"
          "choose_index reports only the candidate that won, so measuring a\n"
          "structure it rejected -- which is what an experiment comparing\n"
          "families has to do -- needs this. Timed on the C++ side because a\n"
          "pybind11 crossing costs ~1 us and a lookup costs a few ns.");

    m.def("candidates_for", [](LogicalType type, const ColumnShape& shape,
                               double write_fraction) {
              Workload workload;
              workload.write_fraction = write_fraction;
              return hylis::index::candidates_for(type, shape, workload);
          },
          py::arg("type"), py::arg("shape"), py::arg("write_fraction") = 0.0,
          "Which structures can serve this column at all.\n\n"
          "Stage one of the choice, and a hard filter rather than a\n"
          "preference: it runs before anything is built or timed, so the\n"
          "expensive path is never spent on a structure that could not have\n"
          "answered the query.");

    m.def("measure_shape", [](const std::vector<ColumnKey>& keys,
                              const std::vector<ColumnValue>& values) {
              return hylis::index::measure_shape(keys, values);
          }, py::arg("keys"), py::arg("values"),
          "Rows, distinct values, uniqueness and monotonicity, in one pass.");

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
        .def("build_column", [](IndexCatalog& self, const std::string& column,
                                const std::vector<ColumnKey>& keys,
                                const std::vector<ColumnValue>& values,
                                double write_fraction) {
            Workload workload;
            workload.write_fraction = write_fraction;
            return self.build_column(column, keys, values, workload);
        }, py::arg("column"), py::arg("keys"), py::arg("values"),
           py::arg("write_fraction") = 0.0,
           "Replay the stored plan when it still applies, re-tune when it\n"
           "does not, and record whatever was actually used.\n\n"
           "A stale plan is re-timed before being thrown away: re-tuning\n"
           "means rebuilding every candidate, and most changes to a column\n"
           "do not change which structure is right for it.")
        .def("explain_column", [](IndexCatalog& self, const std::string& column,
                                  const std::vector<ColumnKey>& keys,
                                  const std::vector<ColumnValue>& values,
                                  double write_fraction) {
            Workload workload;
            workload.write_fraction = write_fraction;
            IndexCatalog::Decision decision;
            ColumnIndex index =
                self.build_column(column, keys, values, workload, &decision);
            py::dict out;
            out["action"] = decision.action == IndexCatalog::Action::Replayed
                                ? "replayed"
                                : decision.action == IndexCatalog::Action::Retuned
                                      ? "retuned"
                                      : "chosen";
            out["freshness"] = decision.freshness == IndexCatalog::Freshness::Fresh
                                   ? "fresh"
                                   : decision.freshness == IndexCatalog::Freshness::Stale
                                         ? "stale"
                                         : "missing";
            out["recorded_ns"] = decision.recorded_ns;
            out["measured_ns"] = decision.measured_ns;
            out["kind"] = std::string(to_string(index.kind()));
            return out;
        }, py::arg("column"), py::arg("keys"), py::arg("values"),
           py::arg("write_fraction") = 0.0,
           "build_column, but reporting why the column got what it got.")
        .def_property("retune_threshold", &IndexCatalog::retune_threshold,
                      &IndexCatalog::set_retune_threshold,
                      "How much slower a replayed stale plan may be before a\n"
                      "full re-tune is worth paying for. Default 1.5.")
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
