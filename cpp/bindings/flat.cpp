// bindings/flat.cpp
//
// pybind11 binding for the flat vector index, importable as `hylis._flat`.
//
// Vectors arrive as numpy arrays rather than Python lists: every vector in
// this project already is one, and copying a 128-float row through a list per
// query would dominate the measurement this index exists to provide.
//
// Two result shapes, on purpose. Single-query search returns Neighbor objects,
// which read well interactively. Batch search returns the (ids, scores) array
// pair that FAISS and ann-benchmarks use, so results drop straight into a
// recall computation with no reshaping.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "index/flat.hpp"

namespace py = pybind11;
using hylis::index::FlatIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Validate a 1-D query and hand back a raw pointer into it.
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

}  // namespace

PYBIND11_MODULE(_flat, m) {
    m.doc() = "hylis flat (brute-force) vector index core (C++)";
    hylis::bindings::attach_build_info(m);

    py::enum_<Metric>(m, "Metric", "Similarity measure used by a FlatIndex.")
        .value("L2", Metric::L2, "Euclidean distance; smaller is nearer")
        .value("InnerProduct", Metric::InnerProduct, "dot product; larger is more similar")
        .value("Cosine", Metric::Cosine,
               "inner product on normalised vectors; larger is more similar");

    py::class_<Neighbor>(m, "Neighbor", "One search result.")
        .def_readonly("id", &Neighbor::id, "Row index of the matched vector.")
        .def_readonly("score", &Neighbor::score,
                      "Distance for L2, similarity for InnerProduct/Cosine.")
        .def("__repr__", [](const Neighbor& n) {
            return "Neighbor(id=" + std::to_string(n.id) +
                   ", score=" + std::to_string(n.score) + ")";
        })
        // Unpackable as `for id, score in results`.
        .def("__iter__", [](const Neighbor& n) {
            return py::iter(py::make_tuple(n.id, n.score));
        });

    py::class_<FlatIndex>(m, "FlatIndex",
        "Exhaustive k-nearest-neighbour search over dense float32 vectors.\n\n"
        "Exact by construction: it scores every stored vector, so its answers\n"
        "are the ground truth an approximate index is graded against.")
        .def(py::init<std::size_t, Metric>(),
             py::arg("dim"), py::arg("metric") = Metric::L2,
             "Create an empty index over `dim`-dimensional vectors.")

        .def("add", [](FlatIndex& self, const FloatArray& vec) {
            return self.add(as_vector(vec, self.dim(), "add"));
        }, py::arg("vector"), "Append one vector. Returns its assigned id.")

        .def("add_batch", [](FlatIndex& self, const FloatArray& arr) {
            if (arr.ndim() != 2) {
                throw std::invalid_argument(
                    "add_batch: expected a 2-D (n, dim) array, got " +
                    std::to_string(arr.ndim()) + " dimensions");
            }
            if (static_cast<std::size_t>(arr.shape(1)) != self.dim()) {
                throw std::invalid_argument(
                    "add_batch: expected " + std::to_string(self.dim()) +
                    " columns, got " + std::to_string(arr.shape(1)));
            }
            const auto n = static_cast<std::size_t>(arr.shape(0));
            const float* data = arr.data();
            {
                py::gil_scoped_release unlock;
                self.add_batch(data, n);
            }
        }, py::arg("vectors"), "Append n vectors from an (n, dim) array.")

        .def("search", [](const FlatIndex& self, const FloatArray& query, std::size_t k) {
            const float* q = as_vector(query, self.dim(), "search");
            py::gil_scoped_release unlock;
            return self.search(q, k);
        }, py::arg("query"), py::arg("k") = 10,
           "Exact search over every stored vector. Returns up to k Neighbors,\n"
           "best first, with ties broken by lower id.")

        .def("search_filtered", [](const FlatIndex& self, const FloatArray& query,
                                   std::size_t k,
                                   const std::vector<std::int64_t>& allowed) {
            const float* q = as_vector(query, self.dim(), "search_filtered");
            py::gil_scoped_release unlock;
            return self.search_filtered(q, k, allowed);
        }, py::arg("query"), py::arg("k"), py::arg("allowed"),
           "Search restricted to `allowed` ids -- the pre-filter query plan.\n"
           "Costs O(len(allowed)) rather than O(len(index)), which is what\n"
           "makes it the cheaper plan when a predicate is selective.\n"
           "Raises IndexError if any id is out of range.")

        .def("search_batch", [](const FlatIndex& self, const FloatArray& queries,
                                std::size_t k) {
            if (queries.ndim() != 2) {
                throw std::invalid_argument(
                    "search_batch: expected a 2-D (n_queries, dim) array, got " +
                    std::to_string(queries.ndim()) + " dimensions");
            }
            if (static_cast<std::size_t>(queries.shape(1)) != self.dim()) {
                throw std::invalid_argument(
                    "search_batch: expected " + std::to_string(self.dim()) +
                    " columns, got " + std::to_string(queries.shape(1)));
            }
            const auto nq = static_cast<std::size_t>(queries.shape(0));
            // Every row scans the same corpus, so every row returns the same
            // count; clamping once keeps the output rectangular and avoids
            // the -1 padding FAISS has to use.
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
                    const std::vector<Neighbor> found = self.search(q + i * dim, width);
                    for (std::size_t j = 0; j < width; ++j) {
                        id_out[i * width + j] = found[j].id;
                        score_out[i * width + j] = found[j].score;
                    }
                }
            }
            return py::make_tuple(ids, scores);
        }, py::arg("queries"), py::arg("k") = 10,
           "Search many queries at once. Returns (ids, scores), each an\n"
           "(n_queries, min(k, len(index))) array -- the layout ann-benchmarks\n"
           "and FAISS use, so it feeds a recall computation directly.")

        .def("vector_at", [](const FlatIndex& self, std::int64_t id) {
            const float* v = self.vector_at(id);
            return py::array_t<float>(static_cast<py::ssize_t>(self.dim()), v);
        }, py::arg("id"),
           "Return a stored vector as a numpy array. For Cosine this is the\n"
           "normalised form, not what was originally added.")

        .def("reserve", &FlatIndex::reserve, py::arg("n"),
             "Pre-allocate room for n vectors.")
        .def("clear", &FlatIndex::clear, "Remove every vector; keeps dim and metric.")
        .def_property_readonly("dim", &FlatIndex::dim, "Vector dimensionality.")
        .def_property_readonly("metric", &FlatIndex::metric, "Metric in use.")
        .def("__len__", &FlatIndex::size, "Number of stored vectors.")
        .def("__repr__", [](const FlatIndex& self) {
            const char* name = self.metric() == Metric::L2 ? "L2"
                             : self.metric() == Metric::InnerProduct ? "InnerProduct"
                             : "Cosine";
            return "FlatIndex(dim=" + std::to_string(self.dim()) +
                   ", metric=" + name +
                   ", size=" + std::to_string(self.size()) + ")";
        });
}
