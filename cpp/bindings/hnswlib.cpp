// bindings/hnswlib.cpp
//
// nmslib/hnswlib exposed as `hylis._hnswlib`, wrapped to our interface.
//
// Benchmark baseline only — nothing in the engine imports this. It exists to
// answer the one question the project cannot answer from the inside: is our
// HNSW competitive, or a strawman? A router that beats a slow baseline has
// proved nothing.
//
// The module always exists so `import hylis._hnswlib` never fails; when
// hnswlib was not fetched, `available` is False and constructing the index
// raises with an explanation rather than an ImportError three frames away.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "build_info.hpp"

#ifdef HYLIS_HAS_HNSWLIB
#include "index/hnswlib_adapter.hpp"
#endif

namespace py = pybind11;

namespace {
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
}  // namespace

PYBIND11_MODULE(_hnswlib, m) {
    m.doc() = "nmslib/hnswlib wrapped to the hylis vector-index interface "
              "(benchmark baseline only)";
    hylis::bindings::attach_build_info(m);

#ifdef HYLIS_HAS_HNSWLIB
    m.attr("available") = true;

    using hylis::index::HnswlibIndex;
    using hylis::index::Metric;
    using hylis::index::Neighbor;

    py::module_::import("hylis._flat");  // registers Metric and Neighbor

    auto as_vector = [](const FloatArray& arr, std::size_t dim, const char* what) {
        if (arr.ndim() != 1 || static_cast<std::size_t>(arr.shape(0)) != dim) {
            throw std::invalid_argument(
                std::string(what) + ": expected a 1-D array of " +
                std::to_string(dim) + " values");
        }
        return arr.data();
    };

    py::class_<HnswlibIndex>(m, "HnswlibIndex",
        "The reference HNSW implementation, behind our interface.\n\n"
        "Scores are converted to match ours exactly: hnswlib returns squared\n"
        "L2 where we return true distance, and 1-dot where we return the dot\n"
        "product. Without that, a score comparison between the two would be\n"
        "comparing different quantities.")
        .def(py::init<std::size_t, Metric, std::size_t, std::size_t, std::size_t,
                      std::uint64_t>(),
             py::arg("dim"), py::arg("metric") = Metric::L2,
             py::arg("capacity") = 1000, py::arg("M") = 16,
             py::arg("ef_construction") = 200, py::arg("seed") = 100,
             "capacity is hnswlib's preallocated element count; it grows\n"
             "automatically, but sizing it up front avoids repeated reallocation.")

        .def("add", [as_vector](HnswlibIndex& self, const FloatArray& vec) {
            return self.add(as_vector(vec, self.dim(), "add"));
        }, py::arg("vector"))

        .def("add_batch", [](HnswlibIndex& self, const FloatArray& arr) {
            if (arr.ndim() != 2 ||
                static_cast<std::size_t>(arr.shape(1)) != self.dim()) {
                throw std::invalid_argument(
                    "add_batch: expected a 2-D (n, " + std::to_string(self.dim()) +
                    ") array");
            }
            const auto n = static_cast<std::size_t>(arr.shape(0));
            const float* data = arr.data();
            py::gil_scoped_release unlock;
            self.add_batch(data, n);
        }, py::arg("vectors"))

        .def("search", [as_vector](const HnswlibIndex& self, const FloatArray& query,
                                   std::size_t k, std::size_t ef) {
            const float* q = as_vector(query, self.dim(), "search");
            py::gil_scoped_release unlock;
            return self.search(q, k, ef);
        }, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 0)

        .def("search_filtered", [as_vector](const HnswlibIndex& self,
                                            const FloatArray& query, std::size_t k,
                                            const std::vector<std::int64_t>& allowed,
                                            std::size_t ef) {
            const float* q = as_vector(query, self.dim(), "search_filtered");
            py::gil_scoped_release unlock;
            return self.search_filtered(q, k, allowed, ef);
        }, py::arg("query"), py::arg("k"), py::arg("allowed"), py::arg("ef") = 0)

        .def("search_batch", [](const HnswlibIndex& self, const FloatArray& queries,
                                std::size_t k, std::size_t ef) {
            if (queries.ndim() != 2 ||
                static_cast<std::size_t>(queries.shape(1)) != self.dim()) {
                throw std::invalid_argument("search_batch: expected a 2-D array");
            }
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
                    const std::vector<Neighbor> found = self.search(q + i * dim, width, ef);
                    for (std::size_t j = 0; j < width; ++j) {
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
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 0)

        .def("memory_bytes", &HnswlibIndex::memory_bytes)
        .def_property_readonly("dim", &HnswlibIndex::dim)
        .def_property_readonly("metric", &HnswlibIndex::metric)
        .def("__len__", &HnswlibIndex::size)
        .def("__repr__", [](const HnswlibIndex& self) {
            return "HnswlibIndex(dim=" + std::to_string(self.dim()) +
                   ", size=" + std::to_string(self.size()) + ")";
        });
#else
    m.attr("available") = false;
    m.def("HnswlibIndex", [](py::args, py::kwargs) -> py::object {
        throw std::runtime_error(
            "hnswlib was not available when hylis was built. Reconfigure with "
            "-DHYLIS_WITH_HNSWLIB=ON and a working network connection, or run "
            "the benchmarks without the external baseline.");
    }, "Unavailable in this build — see `available`.");
#endif
}
