// index/router.hpp
//
// The neural router: a learned replacement for HNSW's hierarchical descent.
//
// What the hierarchy is for
// -------------------------
// Strip HNSW down and its upper layers do exactly one job — hand the layer-0
// beam search a starting node near the query. The descent walks layers
// max_level..1 with a beam of 1 and produces a single entry point. Everything
// that actually finds neighbours happens at layer 0.
//
// So the router's job is stated precisely: given a query vector, predict good
// layer-0 entry points *directly*, without walking a graph to find them.
//
// Why classification over clusters
// --------------------------------
// The obvious formulation — regress a node id from the query — cannot work.
// Node ids are arbitrary labels; node 5000 is not "between" 4999 and 5001 in
// any geometric sense, so there is no continuous target to fit.
//
// Instead the corpus is partitioned by k-means into C clusters, each with a
// precomputed *medoid* — the real node nearest that cluster's centroid. The
// router is a classifier from query vector to cluster, and the medoids of its
// top-p clusters become the entry points. Clusters carry geometry, so the
// target is learnable, and the output maps straight onto concrete nodes.
//
// Trained in Python, run here
// ---------------------------
// Training is offline and one-time, so it lives in python/hylis/router.py
// where loss curves can be inspected. Inference has to be here: a query costs
// ~30us end to end, and a Python callback per query would cost more than the
// query does — the benchmark would be measuring pybind11 rather than the idea.
//
// Weights cross the boundary as JSON, parsed with the same primitives the
// record store and index catalog use.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/json_detail.hpp"

namespace hylis::index {

class NeuralRouter {
public:
    NeuralRouter() = default;

    std::size_t dim() const { return dim_; }
    std::size_t hidden() const { return hidden_; }
    std::size_t clusters() const { return clusters_; }
    bool empty() const { return clusters_ == 0; }

    // The node standing in for a cluster — its medoid.
    std::uint32_t medoid(std::size_t cluster) const {
        if (cluster >= clusters_) {
            throw std::out_of_range("NeuralRouter: cluster " + std::to_string(cluster) +
                                    " out of range [0, " + std::to_string(clusters_) + ")");
        }
        return medoids_[cluster];
    }

    // Top-p cluster ids for a query, best first.
    //
    // Ranked on raw logits, with no softmax: the softmax is monotonic, so it
    // cannot change the ordering, and skipping it saves C calls to exp() on a
    // path that runs once per query.
    void predict(const float* query, std::size_t p,
                 std::vector<std::uint32_t>& out) const {
        out.clear();
        if (clusters_ == 0 || p == 0) return;
        p = std::min(p, clusters_);

        forward(query);

        // p is 1 or 2 in practice, so a partial selection beats sorting C.
        for (std::size_t rank = 0; rank < p; ++rank) {
            std::size_t best = clusters_;
            float best_logit = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < clusters_; ++c) {
                if (std::find(out.begin(), out.end(), static_cast<std::uint32_t>(c)) !=
                    out.end()) {
                    continue;
                }
                if (logits_[c] > best_logit) {
                    best_logit = logits_[c];
                    best = c;
                }
            }
            if (best == clusters_) break;
            out.push_back(static_cast<std::uint32_t>(best));
        }
    }

    // Entry nodes for a query: the medoids of its top-p clusters, deduplicated
    // (two clusters can share a medoid on degenerate data).
    void entry_points(const float* query, std::size_t p,
                      std::vector<std::uint32_t>& out) const {
        predict(query, p, scratch_clusters_);
        out.clear();
        for (std::uint32_t cluster : scratch_clusters_) {
            const std::uint32_t node = medoids_[cluster];
            if (std::find(out.begin(), out.end(), node) == out.end()) {
                out.push_back(node);
            }
        }
    }

    // Raw logits for the last predict()/entry_points() call. Exposed so the
    // Python and C++ forward passes can be compared directly — a transposed
    // weight matrix would otherwise produce a router that "works" while
    // routing badly, and every downstream number would measure the bug.
    const std::vector<float>& last_logits() const { return logits_; }

    // ---- staleness ------------------------------------------------------
    //
    // A router is fitted to a snapshot. Vectors added afterwards fall in
    // clusters that did not exist when it was trained, and the medoids drift
    // away from the regions they are supposed to represent. Neither shows up
    // as an error: the search still returns correct results, just from worse
    // starting points, so nothing complains and the benefit quietly erodes.
    //
    // What is measured, and why not the obvious thing
    // -----------------------------------------------
    // The obvious drift statistic is distance from a new vector to its
    // assigned *centroid*. This router deliberately does not store centroids
    // — they exist during training only, and keeping them would mean two
    // things that answer the same question and could disagree.
    //
    // So drift is measured against the **medoids**, which are stored. That is
    // not a workaround, it is the better statistic: the medoid is the node the
    // beam actually starts from, so distance-to-medoid is the quantity that
    // matters directly, where distance-to-centroid would be a proxy for it.

    std::size_t trained_on() const { return trained_on_; }

    // Mean distance from a training vector to the medoid this router sends it
    // to, recorded at training time. Zero if the router predates this field,
    // in which case drift cannot be assessed and says so rather than
    // inventing a baseline.
    double baseline_entry_distance() const { return baseline_entry_; }
    bool has_baseline() const { return baseline_entry_ > 0.0; }

    // Repoint a cluster at a different node. The whole of medoid repair: the
    // classifier is left alone and only the nodes it lands on are moved.
    void set_medoid(std::size_t cluster, std::uint32_t node) {
        if (cluster >= clusters_) {
            throw std::out_of_range("NeuralRouter::set_medoid: cluster " +
                                    std::to_string(cluster) + " of " +
                                    std::to_string(clusters_));
        }
        medoids_[cluster] = node;
    }

    void set_baseline_entry_distance(double distance) { baseline_entry_ = distance; }
    void set_trained_on(std::size_t n) { trained_on_ = n; }

    // Confirm this router actually belongs to the index about to use it.
    void require_compatible(std::size_t index_dim, std::size_t node_count) const {
        if (clusters_ == 0) throw std::logic_error("NeuralRouter: no weights loaded");
        if (dim_ != index_dim) {
            throw std::invalid_argument(
                "NeuralRouter: trained for " + std::to_string(dim_) +
                "-dimensional vectors, index holds " + std::to_string(index_dim));
        }
        for (std::size_t c = 0; c < clusters_; ++c) {
            if (medoids_[c] >= node_count) {
                throw std::invalid_argument(
                    "NeuralRouter: cluster " + std::to_string(c) + " points at node " +
                    std::to_string(medoids_[c]) + ", but the index holds only " +
                    std::to_string(node_count) + " — the router was trained on a "
                    "different corpus");
            }
        }
    }

    static NeuralRouter from_json(const std::string& blob) {
        using namespace hylis::storage::json_detail;
        NeuralRouter router;

        std::vector<double> w1, b1, w2, b2, medoids;
        const char* p = blob.c_str();
        skip_ws(p);
        expect(p, '{');
        skip_ws(p);
        if (*p == '}') throw std::runtime_error("router: empty object");

        while (true) {
            skip_ws(p);
            const std::string key = read_string(p);
            skip_ws(p);
            expect(p, ':');
            skip_ws(p);

            if (key == "dim") router.dim_ = static_cast<std::size_t>(read_int(p));
            else if (key == "hidden") router.hidden_ = static_cast<std::size_t>(read_int(p));
            else if (key == "clusters") router.clusters_ = static_cast<std::size_t>(read_int(p));
            else if (key == "w1") w1 = read_number_array(p);
            else if (key == "b1") b1 = read_number_array(p);
            else if (key == "w2") w2 = read_number_array(p);
            else if (key == "b2") b2 = read_number_array(p);
            else if (key == "medoids") medoids = read_number_array(p);
            // Both optional: a router written before staleness tracking
            // existed still loads, and reports that it has no baseline
            // rather than pretending to one.
            else if (key == "trained_on")
                router.trained_on_ = static_cast<std::size_t>(read_int(p));
            else if (key == "baseline_entry_distance")
                router.baseline_entry_ = read_double(p);
            else skip_value(p);

            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("router: expected , or } at top level");
        }

        router.assign(w1, b1, w2, b2, medoids);
        return router;
    }

    static NeuralRouter load(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            throw std::runtime_error("NeuralRouter::load: cannot open " + path);
        }
        std::string blob;
        char buffer[8192];
        std::size_t got = 0;
        while ((got = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
            blob.append(buffer, got);
        }
        std::fclose(f);
        try {
            return from_json(blob);
        } catch (const std::exception& e) {
            throw std::runtime_error("NeuralRouter::load: " + path + " is not a valid "
                                     "router file: " + e.what());
        }
    }

private:
    // Shapes are checked here rather than trusted, because a silently
    // mis-shaped weight matrix produces plausible-looking nonsense.
    void assign(const std::vector<double>& w1, const std::vector<double>& b1,
                const std::vector<double>& w2, const std::vector<double>& b2,
                const std::vector<double>& medoids) {
        auto expect_size = [](const char* name, std::size_t got, std::size_t want) {
            if (got != want) {
                throw std::runtime_error(
                    std::string("router: ") + name + " has " + std::to_string(got) +
                    " values, expected " + std::to_string(want));
            }
        };
        if (dim_ == 0 || hidden_ == 0 || clusters_ == 0) {
            throw std::runtime_error("router: dim, hidden and clusters must all be > 0");
        }
        expect_size("w1", w1.size(), dim_ * hidden_);
        expect_size("b1", b1.size(), hidden_);
        expect_size("w2", w2.size(), hidden_ * clusters_);
        expect_size("b2", b2.size(), clusters_);
        expect_size("medoids", medoids.size(), clusters_);

        // Narrowed to float on load, and the Python side narrows before
        // exporting too, so both run the same arithmetic on the same values.
        w1_.assign(w1.begin(), w1.end());
        b1_.assign(b1.begin(), b1.end());
        w2_.assign(w2.begin(), w2.end());
        b2_.assign(b2.begin(), b2.end());
        medoids_.clear();
        medoids_.reserve(medoids.size());
        for (double v : medoids) {
            if (v < 0) throw std::runtime_error("router: negative medoid id");
            medoids_.push_back(static_cast<std::uint32_t>(v));
        }

        hidden_buf_.assign(hidden_, 0.0f);
        logits_.assign(clusters_, 0.0f);
    }

    // dim -> hidden (ReLU) -> clusters. Row-major weights, matching numpy's
    // default layout so the export is a plain ravel() with no transpose.
    void forward(const float* query) const {
        for (std::size_t h = 0; h < hidden_; ++h) {
            float acc = b1_[h];
            for (std::size_t i = 0; i < dim_; ++i) acc += query[i] * w1_[i * hidden_ + h];
            hidden_buf_[h] = acc > 0.0f ? acc : 0.0f;
        }
        for (std::size_t c = 0; c < clusters_; ++c) {
            float acc = b2_[c];
            for (std::size_t h = 0; h < hidden_; ++h) {
                acc += hidden_buf_[h] * w2_[h * clusters_ + c];
            }
            logits_[c] = acc;
        }
    }

    std::size_t dim_ = 0;
    std::size_t hidden_ = 0;
    std::size_t clusters_ = 0;
    std::vector<float> w1_, b1_, w2_, b2_;
    std::vector<std::uint32_t> medoids_;
    std::size_t trained_on_ = 0;
    double baseline_entry_ = 0.0;

    // Reused across queries so inference allocates nothing. Makes the router
    // non-thread-safe, matching HnswIndex's search for the same reason.
    mutable std::vector<float> hidden_buf_;
    mutable std::vector<float> logits_;
    mutable std::vector<std::uint32_t> scratch_clusters_;
};

}  // namespace hylis::index
