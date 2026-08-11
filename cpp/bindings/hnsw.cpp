// bindings/hnsw.cpp
//
// pybind11 binding for the HNSW graph index, importable as `hylis._hnsw`.
//
// Mirrors the flat index's binding style: numpy arrays in, (ids, scores)
// arrays out for batch search, Neighbor objects for single queries.
//
// HNSW does not replace FlatIndex — both stay, and expose the same
// search/search_filtered pair so the query planner can hold either. Which one
// is right depends on the selectivity of a query's filter, so it is a
// query-time decision rather than a build-time one.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "index/hnsw.hpp"
#include "index/router.hpp"

namespace py = pybind11;

using hylis::index::HnswIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;
using hylis::index::NeuralRouter;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

const float* as_vector(const FloatArray& arr, std::size_t dim, const char* what) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(
            std::string(what) + ": expected a 1-D array, got " +
            std::to_string(arr.ndim()) + " dimensions");
    }
    if (static_cast<std::size_t>(arr.shape(0)) != dim) {
        throw std::invalid_argument(
            std::string(what) + ": expected " + std::to_string(dim) +
            " values, got " + std::to_string(arr.shape(0)));
    }
    return arr.data();
}

void require_matrix(const FloatArray& arr, std::size_t dim, const char* what) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(
            std::string(what) + ": expected a 2-D array, got " +
            std::to_string(arr.ndim()) + " dimensions");
    }
    if (static_cast<std::size_t>(arr.shape(1)) != dim) {
        throw std::invalid_argument(
            std::string(what) + ": expected " + std::to_string(dim) +
            " columns, got " + std::to_string(arr.shape(1)));
    }
}

}  // namespace

PYBIND11_MODULE(_hnsw, m) {
    m.doc() = "hylis HNSW graph index core (C++): approximate nearest neighbours";
    hylis::bindings::attach_build_info(m);

    // Metric and Neighbor are already bound by hylis._flat, and pybind11
    // registers C++ types globally — importing that module first makes those
    // registrations visible here rather than duplicating them.
    py::module_::import("hylis._flat");

    py::class_<NeuralRouter>(m, "NeuralRouter",
        "A learned replacement for HNSW's hierarchical descent.\n\n"
        "The upper layers exist only to hand the layer-0 search a starting\n"
        "node near the query. This predicts those entry points directly:\n"
        "a classifier from query vector to k-means cluster, whose medoids\n"
        "become the entry nodes.\n\n"
        "Trained by python/hylis/router.py and loaded here — inference must\n"
        "be in C++ because a query costs ~30us end to end, and a Python\n"
        "callback per query would cost more than the query does.")
        .def(py::init<>())
        .def_static("from_json", &NeuralRouter::from_json, py::arg("blob"))
        .def_static("load", &NeuralRouter::load, py::arg("path"))
        .def("predict", [](const NeuralRouter& self, const FloatArray& query,
                           std::size_t p) {
            std::vector<std::uint32_t> out;
            self.predict(as_vector(query, self.dim(), "predict"), p, out);
            return out;
        }, py::arg("query"), py::arg("p") = 1,
           "Top-p cluster ids, best first.")
        .def("entry_points", [](const NeuralRouter& self, const FloatArray& query,
                                std::size_t p) {
            std::vector<std::uint32_t> out;
            self.entry_points(as_vector(query, self.dim(), "entry_points"), p, out);
            return out;
        }, py::arg("query"), py::arg("p") = 2,
           "Node ids to seed the beam with: the medoids of the top-p clusters.")
        .def("logits", [](const NeuralRouter& self, const FloatArray& query) {
            std::vector<std::uint32_t> ignored;
            self.predict(as_vector(query, self.dim(), "logits"), 1, ignored);
            const std::vector<float>& logits = self.last_logits();
            return py::array_t<float>(static_cast<py::ssize_t>(logits.size()),
                                      logits.data());
        }, py::arg("query"),
           "Raw pre-softmax scores. Exposed so the Python and C++ forward\n"
           "passes can be compared directly — a transposed weight matrix\n"
           "would otherwise give a router that 'works' while routing badly.")
        .def("medoid", &NeuralRouter::medoid, py::arg("cluster"))
        .def("set_medoid", &NeuralRouter::set_medoid,
             py::arg("cluster"), py::arg("node"),
             "Repoint a cluster at a different node. Medoid repair leaves the\n"
             "classifier alone and moves only the nodes it lands on.")
        .def_property_readonly("dim", &NeuralRouter::dim)
        .def_property_readonly("hidden", &NeuralRouter::hidden)
        .def_property_readonly("clusters", &NeuralRouter::clusters)
        .def_property_readonly("trained_on", &NeuralRouter::trained_on,
                               "Corpus size when this router was fitted; 0 if\n"
                               "it was written before staleness tracking.")
        .def_property_readonly("baseline_entry_distance",
                               &NeuralRouter::baseline_entry_distance,
                               "Mean distance from a training vector to the\n"
                               "medoid it routes to, at training time.")
        .def_property_readonly("has_baseline", &NeuralRouter::has_baseline,
                               "False when drift cannot be assessed, which is\n"
                               "reported rather than papered over.")
        .def("__repr__", [](const NeuralRouter& self) {
            return "NeuralRouter(dim=" + std::to_string(self.dim()) +
                   ", hidden=" + std::to_string(self.hidden()) +
                   ", clusters=" + std::to_string(self.clusters()) + ")";
        });

    py::class_<HnswIndex::RouterHealth>(m, "RouterHealth",
        "How far a router's entry points have drifted from the data.\n\n"
        "A stale router still returns correct results, just from worse\n"
        "starting points -- so nothing fails, and the benefit erodes quietly.\n"
        "This is what makes that visible.")
        .def_readonly("sampled", &HnswIndex::RouterHealth::sampled)
        .def_readonly("trained_on", &HnswIndex::RouterHealth::trained_on)
        .def_readonly("current_size", &HnswIndex::RouterHealth::current_size)
        .def_readonly("growth_ratio", &HnswIndex::RouterHealth::growth_ratio,
                      "current_size / trained_on.")
        .def_readonly("mean_entry_distance",
                      &HnswIndex::RouterHealth::mean_entry_distance)
        .def_readonly("baseline", &HnswIndex::RouterHealth::baseline,
                      "The same figure, recorded at training time.")
        .def_readonly("drift_ratio", &HnswIndex::RouterHealth::drift_ratio,
                      "mean_entry_distance / baseline. Above 1 means entry\n"
                      "points have moved away from the vectors they serve.")
        .def_readonly("comparable", &HnswIndex::RouterHealth::comparable,
                      "False when the router carries no baseline, in which\n"
                      "case drift_ratio is meaningless and says so.")
        .def("__repr__", [](const HnswIndex::RouterHealth& h) {
            return "RouterHealth(drift=" + std::to_string(h.drift_ratio) +
                   ", growth=" + std::to_string(h.growth_ratio) +
                   ", comparable=" + (h.comparable ? "True" : "False") + ")";
        });

    py::class_<HnswIndex::Stats>(m, "HnswStats", "Shape and cost of a built graph.")
        .def_readonly("nodes", &HnswIndex::Stats::nodes)
        .def_readonly("levels", &HnswIndex::Stats::levels,
                      "Number of layers in use; 1 + the highest.")
        .def_readonly("edges", &HnswIndex::Stats::edges,
                      "Directed edges summed over every layer.")
        .def_readonly("layer0_edges", &HnswIndex::Stats::layer0_edges)
        .def_readonly("mean_degree_l0", &HnswIndex::Stats::mean_degree_l0)
        .def_readonly("max_degree_l0", &HnswIndex::Stats::max_degree_l0)
        .def_readonly("entry_point", &HnswIndex::Stats::entry_point)
        .def_readonly("layer_population", &HnswIndex::Stats::layer_population,
                      "Nodes reaching each layer. Should decay by roughly 1/M\n"
                      "per level — that decay is what makes the upper layers a\n"
                      "cheap highway rather than a second full index.")
        .def_readonly("reachable", &HnswIndex::Stats::reachable,
                      "Nodes reachable from the entry point through layer 0.\n"
                      "A node with no in-edges can never be returned, so this\n"
                      "is the practical ceiling on recall. Not guaranteed to\n"
                      "equal `nodes`: pruning a full neighbour list can drop a\n"
                      "back-link, and at very small M some nodes get stranded.")
        .def_readonly("graph_bytes", &HnswIndex::Stats::graph_bytes,
                      "Links only — the index overhead over the raw vectors.")
        .def_readonly("total_bytes", &HnswIndex::Stats::total_bytes)
        .def("__repr__", [](const HnswIndex::Stats& s) {
            return "HnswStats(nodes=" + std::to_string(s.nodes) +
                   ", levels=" + std::to_string(s.levels) +
                   ", mean_degree_l0=" + std::to_string(s.mean_degree_l0) +
                   ", reachable=" + std::to_string(s.reachable) + ")";
        });

    py::class_<HnswIndex>(m, "HnswIndex",
        "Hierarchical Navigable Small World graph (Malkov & Yashunin, 2018).\n\n"
        "Approximate: it may miss neighbours, and no error bound makes it\n"
        "exact. The meaningful questions are what recall it reaches and how\n"
        "fast — measure against FlatIndex, which is the exact oracle.\n\n"
        "Not thread-safe: search uses mutable scratch so it does not have to\n"
        "clear O(n) state per query. Use one index per thread.\n\n"
        "No deletion. A node cannot be removed without degrading the graph\n"
        "around it; production systems tombstone and periodically rebuild.")
        .def(py::init<std::size_t, Metric, std::size_t, std::size_t, std::uint64_t,
                      bool>(),
             py::arg("dim"), py::arg("metric") = Metric::L2,
             py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("seed") = 100, py::arg("flat_only") = false,
             "M is links selected per node per layer (2*M allowed at layer 0).\n"
             "ef_construction is the build-time beam width — it buys graph\n"
             "quality, and costs nothing at query time.\n\n"
             "flat_only builds no layers above 0 at all: the full-replacement\n"
             "configuration, where a router is the only way in. The RNG stream\n"
             "is unchanged, so the layer-0 graph is identical to a normal\n"
             "build's and the two remain comparable.")

        .def("add", [](HnswIndex& self, const FloatArray& vec) {
            return self.add(as_vector(vec, self.dim(), "add"));
        }, py::arg("vector"), "Insert one vector. Returns its assigned id.")

        .def("add_batch", [](HnswIndex& self, const FloatArray& arr) {
            require_matrix(arr, self.dim(), "add_batch");
            const auto n = static_cast<std::size_t>(arr.shape(0));
            const float* data = arr.data();
            py::gil_scoped_release unlock;
            self.add_batch(data, n);
        }, py::arg("vectors"), "Insert n vectors from an (n, dim) array.")

        .def("search", [](const HnswIndex& self, const FloatArray& query,
                          std::size_t k, std::size_t ef, bool use_router) {
            const float* q = as_vector(query, self.dim(), "search");
            py::gil_scoped_release unlock;
            return self.search(q, k, ef, use_router);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 0,
           py::arg("use_router") = false,
           "Approximate search. ef is the beam width — the quality knob.\n"
           "Raising it must never lower recall. 0 uses the index default.\n\n"
           "use_router=True replaces the hierarchical descent with the loaded\n"
           "router's entry points. Same graph, same seed, one variable — so a\n"
           "difference is attributable to routing and nothing else.")

        .def("search_filtered", [](const HnswIndex& self, const FloatArray& query,
                                   std::size_t k,
                                   const std::vector<std::int64_t>& allowed,
                                   std::size_t ef, bool use_router) {
            const float* q = as_vector(query, self.dim(), "search_filtered");
            py::gil_scoped_release unlock;
            return self.search_filtered(q, k, allowed, ef, use_router);
        }, py::arg("query"), py::arg("k"), py::arg("allowed"), py::arg("ef") = 0,
           py::arg("use_router") = false,
           "Search restricted to `allowed` ids.\n\n"
           "Non-matching nodes are traversed *through* but never collected —\n"
           "skipping them would disconnect the graph. Cost is therefore about\n"
           "ef/selectivity visits, capped at n, so a tight filter pushes this\n"
           "toward a full scan while FlatIndex's filtered scan gets cheaper.\n"
           "Where those cross is what the query planner has to predict.")

        .def("search_batch", [](const HnswIndex& self, const FloatArray& queries,
                                std::size_t k, std::size_t ef, bool use_router) {
            require_matrix(queries, self.dim(), "search_batch");
            const auto nq = static_cast<std::size_t>(queries.shape(0));
            const std::size_t width = std::min(k, self.size());

            py::array_t<std::int64_t> ids({nq, width});
            py::array_t<float> scores({nq, width});
            std::int64_t* id_out = ids.mutable_data();
            float* score_out = scores.mutable_data();
            const float* q = queries.data();
            const std::size_t dim = self.dim();
            {
                py::gil_scoped_release unlock;
                for (std::size_t i = 0; i < nq; ++i) {
                    const std::vector<Neighbor> found =
                        self.search(q + i * dim, width, ef, use_router);
                    for (std::size_t j = 0; j < width; ++j) {
                        // A graph search can return fewer than k when the beam
                        // runs dry; pad so the array stays rectangular, using
                        // the -1 sentinel ann-benchmarks and FAISS expect.
                        if (j < found.size()) {
                            id_out[i * width + j] = found[j].id;
                            score_out[i * width + j] = found[j].score;
                        } else {
                            id_out[i * width + j] = -1;
                            score_out[i * width + j] =
                                std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
            return py::make_tuple(ids, scores);
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 0,
           py::arg("use_router") = false,
           "Search many queries. Returns (ids, scores) as\n"
           "(n_queries, min(k, len(index))) arrays. Short rows are padded with\n"
           "id -1, which recall_at_k treats as a miss.")

        .def("set_router", &HnswIndex::set_router, py::arg("router"),
             "Attach a trained router. Raises ValueError if it was trained for\n"
             "a different dimensionality or a larger corpus.")
        .def("clear_router", &HnswIndex::clear_router)
        .def("router", &HnswIndex::router, py::return_value_policy::reference_internal)
        .def_property_readonly("has_router", &HnswIndex::has_router)
        .def_property("router_top_p", &HnswIndex::router_top_p,
                      &HnswIndex::set_router_top_p,
                      "How many clusters' medoids seed the beam. More entry\n"
                      "points cost distance computations but give the search\n"
                      "several places to start when the router is unsure.")
        .def_property("router_keeps_global_entry",
                      &HnswIndex::router_keeps_global_entry,
                      &HnswIndex::set_router_keeps_global_entry,
                      "Also seed with the graph's own entry point. One extra\n"
                      "distance computation as insurance against a badly\n"
                      "routed query.")

        .def("router_health", &HnswIndex::router_health,
             py::arg("sample") = 2000, py::arg("seed") = 0,
             "How far the router's entry points have drifted from the data.\n\n"
             "Drift costs recall or latency but produces no error, so nothing\n"
             "else would ever report it. Measured against the medoids rather\n"
             "than centroids: the medoid is the node the beam actually starts\n"
             "from, and this router format deliberately does not persist\n"
             "centroids.")
        .def("repair_router_medoids", &HnswIndex::repair_router_medoids,
             "Move every cluster's medoid back onto the data it now covers,\n"
             "returning how many moved.\n\n"
             "The cheap repair tier: one pass, classifier untouched. Compare\n"
             "a full retrain, which is k-means to convergence plus tens of\n"
             "epochs of gradient descent.")
        .def("rebaseline_router", &HnswIndex::rebaseline_router,
             py::arg("sample") = 2000, py::arg("seed") = 0,
             "Accept the current state as the new normal. Without this a\n"
             "repaired router keeps reporting the drift it just fixed.")

        .def("vector_at", [](const HnswIndex& self, std::int64_t id) {
            const float* v = self.vector_at(id);
            return py::array_t<float>(static_cast<py::ssize_t>(self.dim()), v);
        }, py::arg("id"), "A stored vector; normalised form for Cosine.")

        .def("reachable", &HnswIndex::reachable,
             "Nodes reachable from the entry point at layer 0.")
        .def("stats", &HnswIndex::stats)
        .def("validate", &HnswIndex::validate,
             "Check structural invariants: degree caps, level consistency, no\n"
             "self-loops or duplicate links. Raises RuntimeError on violation.\n"
             "Deliberately does NOT check reachability — see `reachable`.")
        .def("reserve", &HnswIndex::reserve, py::arg("n"))
        .def("clear", &HnswIndex::clear)

        .def_property_readonly("dim", &HnswIndex::dim)
        .def_property_readonly("metric", &HnswIndex::metric)
        .def_property_readonly("M", &HnswIndex::M)
        .def_property_readonly("ef_construction", &HnswIndex::ef_construction)
        .def_property_readonly("max_level", &HnswIndex::max_level)
        .def_property_readonly("entry_point", &HnswIndex::entry_point)
        .def_property_readonly("last_visited", &HnswIndex::last_visited,
                               "Nodes scored by the most recent search — the\n"
                               "number that explains why a graph beats a scan.")
        .def_property_readonly("last_routing_visited",
                               &HnswIndex::last_routing_visited,
                               "Of those, how many were spent just getting to\n"
                               "layer 0: the hierarchy walk, or 0 when a router\n"
                               "supplied the entry points. This is the work the\n"
                               "router is trying to eliminate.")
        .def_property("ef_search", &HnswIndex::ef_search, &HnswIndex::set_ef_search,
                      "Default beam width when a search does not name one.")
        .def_property("use_heuristic", &HnswIndex::uses_heuristic,
                      &HnswIndex::set_use_heuristic,
                      "Whether insertion uses the paper's neighbour-selection\n"
                      "heuristic (Algorithm 4) or simply keeps the M nearest.\n"
                      "Exposed so the difference can be measured.")
        .def("__len__", &HnswIndex::size)
        .def("__repr__", [](const HnswIndex& self) {
            return "HnswIndex(dim=" + std::to_string(self.dim()) +
                   ", M=" + std::to_string(self.M()) +
                   ", size=" + std::to_string(self.size()) +
                   ", levels=" + std::to_string(self.max_level() + 1) + ")";
        });
}
