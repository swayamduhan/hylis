// query/vector_column.hpp
//
// A vector column: embeddings that belong to a table's rows, kept outside the
// records that own them.
//
// Why not in the record
// ---------------------
// Record::columns is map<string,string> and the write-ahead log is JSON text.
// A 128-float embedding base64s to roughly 700 bytes per row, so putting
// embeddings in the payload would make the WAL the dominant cost of the whole
// system and make every checkpoint parse megabytes of base64 to recover data
// the indexes were going to rebuild anyway.
//
// So the floats stay where they already live — the contiguous buffer inside
// FlatIndex and HnswIndex — and this class owns the one thing that was
// missing: the map between a *record key*, which is what the table and every
// predicate speak, and a *row id*, which is what a vector index speaks.
//
// Two id spaces, and they are not the same
// ----------------------------------------
// Scalar indexes in this project map (column value) -> record key, so a
// predicate's answer is directly a set of keys and nothing needs renumbering.
// Vector indexes cannot work that way: their ids are dense positions into a
// float buffer, assigned 0..n-1 in insertion order, because the buffer *is*
// the layout that makes a sequential sweep fast.
//
// The join is therefore a real translation and not a cast:
//
//     key_to_row_   record key -> row id      (unordered; hit on every hybrid query)
//     row_to_key_   row id     -> record key  (dense; hit on every result)
//
// A table whose keys happen to be 0..n-1 in insertion order makes the two
// coincide, which is what rows_are_keys() reports. That is not a curiosity: it
// is the exact precondition for handing a *table* bitmap to a vector search as
// a mask, because bit position i must mean row i in both.
//
// Deletion, and why it leaves a hole
// ----------------------------------
// HNSW has no deletion — removing a node would strand the neighbours that
// point at it, and repairing the graph is roadmap item 3. So erasing a row, or
// replacing its embedding, does not remove anything: the old row becomes an
// **orphan**, and a Bitset of live rows keeps it out of every answer.
//
// That is the same trade the learned index makes for the same reason
// (see the tombstone note in rmi.hpp): a delete is O(1) and correct
// immediately, and the space comes back at a compaction rather than at the
// delete. Here the compaction is compact(), and save() writes the compacted
// form whether or not memory has caught up — so a reopen is always clean.
//
// The cost of an orphan is not only space. With any orphan present, every
// search runs through the *masked* path, which for the graph means stepping
// over rejected nodes to stay connected — precisely the filtered-graph curve
// the planner already models. scripts/experiment_vector_sidecar.py measures
// where that stops being acceptable.
//
// Durability
// ----------
// **Vectors are not write-ahead logged.** They are written at save() — which
// checkpoint() calls — and a crash between two saves loses every embedding
// attached since the last one. The records survive, because they went through
// the WAL; their embeddings do not.
//
// This is a consequence of the decision above, not an oversight, and the fix
// is a typed binary WAL payload (deferred: it is a rewrite of module 1's
// format, versioning and recovery story). Table::save_vectors() exists so a
// caller who wants that durability point can take it without a full
// checkpoint.
//
// On disk
// -------
//   <column>.fvecs   the live vectors, row-major, in row order. Standard
//                    TEXMEX layout — an int32 dimension repeated before every
//                    vector — so python/hylis/datasets.py read_fvecs() reads
//                    it directly and the file is inspectable with tools that
//                    already exist.
//   vectors.json     dim, metric, structure and build parameters, plus the
//                    row -> record key map. Written by Table beside
//                    schema.json and catalog.json, because a sidecar of raw
//                    floats is uninterpretable without all of it: Cosine
//                    stores the *normalised* vector, so reloading under L2
//                    would answer a different question while looking entirely
//                    plausible.
//
// The keys are JSON integers rather than doubles: read_number_array would
// round a key above 2^53 to a different key and report nothing, which in this
// one place would silently attach an embedding to the wrong row.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index/bitmap.hpp"
#include "index/distance.hpp"
#include "index/flat.hpp"
#include "index/hnsw.hpp"
#include "storage/detail.hpp"
#include "storage/json_detail.hpp"

namespace hylis::query {

namespace fs = std::filesystem;

using hylis::index::Bitset;
using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Metric;
using hylis::index::Neighbor;

// Which structures a vector column builds.
//
// The exact index is present either way, and that is deliberate twice over.
// It is the storage — the contiguous float buffer every other structure is
// built from — and it is the oracle: HNSW is approximate by construction, so
// "what recall does it reach" is the only question that can be asked of it,
// and recall is undefined without a true answer. Brute force is never
// replaced here; it stays a selectable plan.
enum class VectorStructure {
    Exact,  // brute force only: exhaustive, and the answer every other is graded against
    Graph,  // HNSW as well, for corpora where scanning everything stops paying
};

inline const char* to_string(VectorStructure s) {
    return s == VectorStructure::Graph ? "graph" : "exact";
}

inline bool try_vector_structure_from_string(const std::string& name,
                                             VectorStructure* out) {
    if (name == "exact") { *out = VectorStructure::Exact; return true; }
    if (name == "graph") { *out = VectorStructure::Graph; return true; }
    return false;
}

// How a vector column is built.
//
// Unlike a scalar column, this is *not* chosen by measurement. choose_index
// times lookups, and the question for a vector index is recall at a target,
// which needs a query workload and a ground truth the table does not have.
// Naming the structure is therefore the honest interface, and the recall
// measurement stays where it already is: scripts/bench_vector.py, against the
// exact index this column keeps for exactly that purpose.
struct VectorPlan {
    VectorStructure structure = VectorStructure::Graph;
    Metric metric = Metric::L2;
    std::size_t M = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 50;
    // Fixed, so a rebuild from the same insertion order reproduces the graph
    // exactly rather than approximately. That is what makes "reopen returns
    // the same neighbours" an assertion instead of a hope — see
    // HnswIndex::clear(), which reseeds.
    std::uint64_t seed = 100;
};

// One neighbour, in the table's terms.
//
// `key` is what the caller asked about and can fetch; `row` is the position
// inside the vector index, kept because it is what a second vector call wants
// and because a test comparing two plans needs to name a row without a
// round-trip through the store.
struct VectorMatch {
    std::int64_t key = 0;
    std::int64_t row = 0;
    float score = 0.0f;
};

class VectorColumn {
public:
    VectorColumn(std::string name, std::size_t dim, VectorPlan plan)
        : name_(std::move(name)), plan_(plan), exact_(dim, plan.metric) {
        if (plan_.structure == VectorStructure::Graph) {
            graph_ = std::make_unique<HnswIndex>(dim, plan_.metric, plan_.M,
                                                 plan_.ef_construction,
                                                 plan_.seed);
            graph_->set_ef_search(plan_.ef_search);
        }
    }

    // Move-only. A vector column is the largest object in the system by a wide
    // margin, and an accidental copy is a silent doubling of the corpus rather
    // than a compile error.
    VectorColumn(const VectorColumn&) = delete;
    VectorColumn& operator=(const VectorColumn&) = delete;
    VectorColumn(VectorColumn&&) noexcept = default;
    VectorColumn& operator=(VectorColumn&&) noexcept = default;

    const std::string& name() const { return name_; }
    std::size_t dim() const { return exact_.dim(); }
    Metric metric() const { return exact_.metric(); }
    const VectorPlan& plan() const { return plan_; }
    bool has_graph() const { return graph_ != nullptr; }
    const FlatIndex& exact() const { return exact_; }
    const HnswIndex* graph() const { return graph_.get(); }

    // Rows with a live embedding.
    std::size_t size() const { return live_count_; }
    // Slots in the underlying indexes, orphans included. The two differ
    // exactly by what deletion could not reclaim.
    std::size_t rows() const { return exact_.size(); }
    std::size_t orphans() const { return rows() - live_count_; }
    bool empty() const { return live_count_ == 0; }

    // Both indexes hold their own copy of the corpus. Stated rather than
    // hidden: it is the price of keeping brute force independently searchable,
    // and it is the largest single memory line in the system.
    std::size_t memory_bytes() const {
        const std::size_t per_row = dim() * sizeof(float);
        const std::size_t floats = rows() * per_row * (graph_ ? 2 : 1);
        return floats + key_to_row_.size() * (sizeof(std::int64_t) * 3) +
               row_to_key_.capacity() * sizeof(std::int64_t) +
               live_.memory_bytes();
    }

    bool contains(std::int64_t key) const {
        return key_to_row_.find(key) != key_to_row_.end();
    }

    // The row a key's embedding occupies, or nothing when it has none.
    std::optional<std::int64_t> row_of(std::int64_t key) const {
        const auto it = key_to_row_.find(key);
        if (it == key_to_row_.end()) return std::nullopt;
        return it->second;
    }

    // The key a row belongs to. Valid only for a live row; an orphan still
    // remembers the key it was written for, which is why liveness is asked of
    // live_ and never inferred from this.
    std::int64_t key_at(std::int64_t row) const {
        require_row(row);
        return row_to_key_[static_cast<std::size_t>(row)];
    }

    bool is_live(std::int64_t row) const {
        return row >= 0 && static_cast<std::size_t>(row) < rows() &&
               live_.test(static_cast<std::size_t>(row));
    }

    // The stored vector for a key, or nullptr. For Cosine this is the
    // normalised form, which is what the index compares against.
    const float* vector_of(std::int64_t key) const {
        const auto row = row_of(key);
        return row ? exact_.vector_at(*row) : nullptr;
    }

    // Whether row id i is record key i for every live row, with no orphans.
    //
    // The precondition for using a table's bitmap as a vector mask: bit
    // position i means "the row at rank i of the table's key space" there and
    // "row i of the float buffer" here, and only when this holds are those the
    // same row. Being wrong would filter on entirely different rows while
    // looking plausible, so it is checked rather than assumed.
    bool rows_are_keys() const { return rows_are_keys_ && orphans() == 0; }

    // --- writes -------------------------------------------------------------

    // Attach an embedding to a record key. Returns the row it occupies.
    //
    // Replacing one appends a new row and orphans the old, because the graph
    // cannot give a node back. The orphan leaves every answer immediately and
    // its space comes back at the next compact() or save().
    std::int64_t put(std::int64_t key, const float* vec) {
        const auto existing = key_to_row_.find(key);
        if (existing != key_to_row_.end()) {
            retire(existing->second);
        }

        const std::int64_t row = exact_.add(vec);
        if (graph_) {
            const std::int64_t graph_row = graph_->add(vec);
            if (graph_row != row) {
                // The two indexes are appended in lockstep and row ids are
                // positions, so a divergence means one of them was written to
                // behind this class's back. Every id in this file would then
                // mean two different vectors.
                throw std::logic_error(
                    "VectorColumn '" + name_ + "': the exact index and the "
                    "graph disagree about row ids (" + std::to_string(row) +
                    " vs " + std::to_string(graph_row) + ")");
            }
        }

        row_to_key_.push_back(key);
        live_.push_back(true);
        ++live_count_;
        key_to_row_[key] = row;
        if (row != key) rows_are_keys_ = false;
        return row;
    }

    std::int64_t put(std::int64_t key, const std::vector<float>& vec) {
        require_dim(vec.size());
        return put(key, vec.data());
    }

    bool erase(std::int64_t key) {
        const auto it = key_to_row_.find(key);
        if (it == key_to_row_.end()) return false;
        retire(it->second);
        key_to_row_.erase(it);
        return true;
    }

    // --- search -------------------------------------------------------------

    // k nearest, best first, in record keys.
    //
    // `exact` forces the brute-force scan. Kept selectable rather than treated
    // as a fallback: it is the only exact answer, and a plan that beats it has
    // to be shown to, not assumed to.
    std::vector<VectorMatch> knn(const float* query, std::size_t k,
                                 std::size_t ef = 0, bool exact = false) const {
        return present(raw_search(query, k, ef, exact, nullptr));
    }

    // More-like-this. The seed is excluded — a row is always its own nearest
    // neighbour, so including it would spend one of the k on a row the caller
    // already has.
    std::vector<VectorMatch> knn_by_key(std::int64_t key, std::size_t k,
                                        std::size_t ef = 0,
                                        bool exact = false) const {
        const auto row = row_of(key);
        if (!row) {
            throw std::invalid_argument(
                "VectorColumn '" + name_ + "': record " + std::to_string(key) +
                " has no embedding, so there is nothing to search from");
        }
        Bitset without = live_;
        without.clear(static_cast<std::size_t>(*row));
        return present(raw_search(exact_.vector_at(*row), k, ef, exact, &without));
    }

    // The live rows behind a set of record keys, in the order given.
    //
    // `without_vector` counts keys that matched a predicate but carry no
    // embedding. They are dropped, because a row with no vector cannot be a
    // nearest neighbour — and counted, because silently returning fewer rows
    // than the predicate matched is the kind of thing a caller must be able to
    // see.
    std::vector<std::int64_t> rows_for(const std::vector<std::int64_t>& keys,
                                       std::size_t* without_vector = nullptr) const {
        std::vector<std::int64_t> out;
        out.reserve(keys.size());
        std::size_t missing = 0;
        for (std::int64_t key : keys) {
            const auto it = key_to_row_.find(key);
            if (it == key_to_row_.end()) { ++missing; continue; }
            out.push_back(it->second);
        }
        if (without_vector) *without_vector = missing;
        return out;
    }

    // Translate raw neighbours into record keys, dropping nothing: every id a
    // search returns is live, because the mask and the corpus are the only two
    // things it can walk.
    std::vector<VectorMatch> present(const std::vector<Neighbor>& raw) const {
        std::vector<VectorMatch> out;
        out.reserve(raw.size());
        for (const Neighbor& n : raw) {
            VectorMatch m;
            m.row = n.id;
            m.key = key_at(n.id);
            m.score = n.score;
            out.push_back(m);
        }
        return out;
    }

    // The live set, for a caller that wants to intersect it with its own mask.
    const Bitset& live() const { return live_; }

    // Every record key carrying an embedding, in row order.
    std::vector<std::int64_t> keys() const {
        std::vector<std::int64_t> out;
        out.reserve(live_count_);
        live_.for_each([&](std::size_t row) { out.push_back(row_to_key_[row]); });
        return out;
    }

    // --- maintenance --------------------------------------------------------

    // Rebuild under different parameters, keeping every embedding.
    //
    // The metric is refused on a non-empty column, and that is not fussiness:
    // Cosine stores the *normalised* vector, so the original is already gone
    // and re-tuning to L2 would measure distances between unit vectors while
    // claiming to measure them between the ones handed in.
    void retune(VectorPlan plan) {
        if (plan.metric != plan_.metric && !empty()) {
            throw std::invalid_argument(
                "VectorColumn '" + name_ + "': the metric cannot change on a "
                "column holding vectors. Cosine stores the normalised form, so "
                "the vectors handed in are no longer recoverable; drop the "
                "column and re-attach the embeddings instead.");
        }

        const std::size_t d = dim();
        std::vector<std::int64_t> keys;
        std::vector<float> data;
        keys.reserve(live_count_);
        data.reserve(live_count_ * d);
        live_.for_each([&](std::size_t row) {
            keys.push_back(row_to_key_[row]);
            const float* v = exact_.vector_at(static_cast<std::int64_t>(row));
            data.insert(data.end(), v, v + d);
        });

        plan_ = plan;
        exact_ = FlatIndex(d, plan_.metric);
        graph_.reset();
        if (plan_.structure == VectorStructure::Graph) {
            graph_ = std::make_unique<HnswIndex>(d, plan_.metric, plan_.M,
                                                 plan_.ef_construction,
                                                 plan_.seed);
            graph_->set_ef_search(plan_.ef_search);
        }
        reset_from(keys, data);
    }

    // Reclaim the space deletion could not. Rebuilds both indexes from the
    // live rows in row order, which renumbers every row after the first hole.
    //
    // Callers hold record keys, so renumbering is invisible to them — that is
    // the whole reason the key/row split exists rather than exposing row ids as
    // the table's identity.
    std::size_t compact() {
        const std::size_t reclaimed = orphans();
        if (reclaimed == 0) return 0;

        std::vector<std::int64_t> keys;
        std::vector<float> data;
        keys.reserve(live_count_);
        data.reserve(live_count_ * dim());
        live_.for_each([&](std::size_t row) {
            keys.push_back(row_to_key_[row]);
            const float* v = exact_.vector_at(static_cast<std::int64_t>(row));
            data.insert(data.end(), v, v + dim());
        });
        reset_from(keys, data);
        return reclaimed;
    }

    // --- persistence --------------------------------------------------------

    // The live vectors, row-major, in row order — the compacted form whether
    // or not memory has caught up, so a reopen is always clean.
    //
    // Written straight to the file rather than through detail::atomic_write:
    // that takes the whole blob as a std::string, which for a vector corpus is
    // a second full copy of the largest object in the system. The temp-file,
    // fsync, rename sequence is the same.
    void save(const fs::path& path) const {
        const fs::path tmp = fs::path(path).concat(".tmp");
        {
            std::FILE* out = std::fopen(tmp.string().c_str(), "wb");
            if (out == nullptr) {
                throw std::runtime_error("VectorColumn::save: cannot open " +
                                         tmp.string());
            }
            const std::int32_t d = static_cast<std::int32_t>(dim());
            bool ok = true;
            live_.for_each([&](std::size_t row) {
                if (!ok) return;
                const float* v = exact_.vector_at(static_cast<std::int64_t>(row));
                ok = std::fwrite(&d, sizeof(d), 1, out) == 1 &&
                     std::fwrite(v, sizeof(float), dim(), out) == dim();
            });
            if (ok) storage::detail::fsync_file(out);
            std::fclose(out);
            if (!ok) {
                throw std::runtime_error("VectorColumn::save: short write to " +
                                         tmp.string());
            }
        }
        fs::rename(tmp, path);
        storage::detail::fsync_path(path.string());
    }

    // Everything needed to interpret that file, as a JSON object. Table
    // gathers these into vectors.json beside the schema and the catalog.
    std::string metadata() const {
        using hylis::storage::json_detail::escape_string;
        std::string out = "{\"name\":\"" + escape_string(name_) + "\",";
        out += "\"dim\":" + std::to_string(dim()) + ",";
        out += "\"metric\":\"" + std::string(index::to_string(metric())) + "\",";
        out += "\"structure\":\"" + std::string(to_string(plan_.structure)) + "\",";
        out += "\"m\":" + std::to_string(plan_.M) + ",";
        out += "\"ef_construction\":" + std::to_string(plan_.ef_construction) + ",";
        out += "\"ef_search\":" + std::to_string(plan_.ef_search) + ",";
        out += "\"seed\":" + std::to_string(plan_.seed) + ",";
        out += "\"keys\":[";
        bool first = true;
        live_.for_each([&](std::size_t row) {
            if (!first) out += ",";
            first = false;
            out += std::to_string(row_to_key_[row]);
        });
        out += "]}";
        return out;
    }

    // Rebuild from a saved sidecar. `keys` is the row -> record key map, in
    // row order, and must be exactly as long as the file.
    //
    // The graph is *not* stored — only the vectors are, and the graph is
    // reconstructed by replaying the same insertions in the same order. With
    // the seed fixed that reproduces it exactly, not approximately, so a
    // reopen returns the same neighbours rather than merely similar ones. What
    // it costs is measured in scripts/experiment_vector_sidecar.py, because
    // "the graph is rebuilt on open" is the kind of claim that should carry a
    // number.
    void load(const fs::path& path, const std::vector<std::int64_t>& keys) {
        std::vector<float> data;
        const std::size_t n = read_fvecs(path, dim(), &data);
        if (n != keys.size()) {
            throw std::runtime_error(
                "VectorColumn '" + name_ + "': " + path.string() + " holds " +
                std::to_string(n) + " vectors but the metadata names " +
                std::to_string(keys.size()) + " rows");
        }
        reset_from(keys, data);
    }

    // Every invariant this class maintains, checked against itself.
    void validate() const {
        if (row_to_key_.size() != rows() || live_.size() != rows()) {
            throw std::logic_error(
                "VectorColumn '" + name_ + "': " + std::to_string(rows()) +
                " rows but " + std::to_string(row_to_key_.size()) +
                " key entries and " + std::to_string(live_.size()) + " bits");
        }
        if (graph_ && graph_->size() != rows()) {
            throw std::logic_error(
                "VectorColumn '" + name_ + "': the graph holds " +
                std::to_string(graph_->size()) + " rows and the exact index " +
                std::to_string(rows()));
        }
        if (live_.count() != live_count_) {
            throw std::logic_error(
                "VectorColumn '" + name_ + "': " + std::to_string(live_count_) +
                " live rows counted but " + std::to_string(live_.count()) +
                " bits set");
        }
        if (key_to_row_.size() != live_count_) {
            throw std::logic_error(
                "VectorColumn '" + name_ + "': " +
                std::to_string(key_to_row_.size()) + " keys mapped but " +
                std::to_string(live_count_) + " live rows");
        }
        // The two maps must be inverses over the live set. An orphan is
        // deliberately not required to appear in key_to_row_ — that is what
        // makes it an orphan.
        for (const auto& [key, row] : key_to_row_) {
            if (!is_live(row)) {
                throw std::logic_error(
                    "VectorColumn '" + name_ + "': record " +
                    std::to_string(key) + " points at row " +
                    std::to_string(row) + ", which is not live");
            }
            if (row_to_key_[static_cast<std::size_t>(row)] != key) {
                throw std::logic_error(
                    "VectorColumn '" + name_ + "': row " + std::to_string(row) +
                    " belongs to record " +
                    std::to_string(row_to_key_[static_cast<std::size_t>(row)]) +
                    " but record " + std::to_string(key) + " points at it");
            }
        }
    }

    // Read a TEXMEX .fvecs file, checking the dimension it claims against the
    // one expected. Public so the load path and its tests share one reader.
    static std::size_t read_fvecs(const fs::path& path, std::size_t dim,
                                  std::vector<float>* out) {
        std::FILE* in = std::fopen(path.string().c_str(), "rb");
        if (in == nullptr) {
            throw std::runtime_error("VectorColumn: cannot open " + path.string());
        }
        out->clear();
        std::size_t n = 0;
        std::int32_t stored = 0;
        try {
            while (std::fread(&stored, sizeof(stored), 1, in) == 1) {
                if (stored <= 0 || static_cast<std::size_t>(stored) != dim) {
                    throw std::runtime_error(
                        path.string() + ": vector " + std::to_string(n) +
                        " claims dimension " + std::to_string(stored) +
                        ", but the column is " + std::to_string(dim) +
                        "-dimensional");
                }
                const std::size_t before = out->size();
                out->resize(before + dim);
                if (std::fread(out->data() + before, sizeof(float), dim, in) != dim) {
                    throw std::runtime_error(path.string() + ": truncated at vector " +
                                             std::to_string(n));
                }
                ++n;
            }
        } catch (...) {
            std::fclose(in);
            throw;
        }
        std::fclose(in);
        return n;
    }

private:
    // Take a live row out of every answer without touching the structures.
    void retire(std::int64_t row) {
        if (!is_live(row)) return;
        live_.clear(static_cast<std::size_t>(row));
        --live_count_;
    }

    std::vector<Neighbor> raw_search(const float* query, std::size_t k,
                                     std::size_t ef, bool exact,
                                     const Bitset* mask) const {
        const bool use_graph = graph_ != nullptr && !exact;
        // No mask and no orphans means nothing to reject, so the unfiltered
        // path runs — which for the graph is the difference between a beam
        // search and a beam search that has to step over dead nodes.
        if (mask == nullptr && orphans() == 0) {
            return use_graph ? graph_->search(query, k, ef) : exact_.search(query, k);
        }
        const Bitset& allowed = mask ? *mask : live_;
        return use_graph ? graph_->search_masked(query, k, allowed, ef)
                         : exact_.search_masked(query, k, allowed);
    }

    // Discard both structures and replay the given rows into them, in order.
    void reset_from(const std::vector<std::int64_t>& keys,
                    const std::vector<float>& data) {
        if (keys.size() * dim() != data.size()) {
            throw std::invalid_argument(
                "VectorColumn '" + name_ + "': " + std::to_string(keys.size()) +
                " keys against " + std::to_string(data.size()) + " floats at " +
                std::to_string(dim()) + " dimensions");
        }
        exact_.clear();
        if (graph_) graph_->clear();
        key_to_row_.clear();
        row_to_key_.clear();
        live_ = Bitset();
        live_count_ = 0;
        rows_are_keys_ = true;

        exact_.reserve(keys.size());
        if (graph_) graph_->reserve(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            put(keys[i], data.data() + i * dim());
        }
    }

    void require_dim(std::size_t got) const {
        if (got != dim()) {
            throw std::invalid_argument(
                "VectorColumn '" + name_ + "': expected a " +
                std::to_string(dim()) + "-dimensional vector, got " +
                std::to_string(got));
        }
    }

    void require_row(std::int64_t row) const {
        if (row < 0 || static_cast<std::size_t>(row) >= rows()) {
            throw std::out_of_range(
                "VectorColumn '" + name_ + "': row " + std::to_string(row) +
                " out of range [0, " + std::to_string(rows()) + ")");
        }
    }

    std::string name_;
    VectorPlan plan_;
    FlatIndex exact_;
    std::unique_ptr<HnswIndex> graph_;

    std::unordered_map<std::int64_t, std::int64_t> key_to_row_;
    std::vector<std::int64_t> row_to_key_;
    Bitset live_;
    std::size_t live_count_ = 0;
    bool rows_are_keys_ = true;
};

}  // namespace hylis::query
