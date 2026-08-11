// index/hnsw.hpp
//
// Hierarchical Navigable Small World graph — approximate nearest neighbour
// search. Malkov & Yashunin, TPAMI 2018.
//
// The idea
// --------
// A proximity graph where each node links to its near neighbours can be
// searched greedily: step to whichever neighbour is closer to the query, and
// repeat. That alone gets stuck in local minima. HNSW fixes it with two
// things:
//
//   * A hierarchy. Each node is assigned a random maximum layer from an
//     exponentially decaying distribution, so roughly 1/M of nodes reach each
//     successive layer. The sparse upper layers act as a coarse highway that
//     crosses the whole space in a few hops; the search descends through them
//     before doing fine-grained work at layer 0. That is what makes the
//     expected cost O(log n) rather than O(n).
//   * A neighbour-selection *heuristic* rather than simply keeping the M
//     closest candidates. See select_neighbors — it is the single most
//     important detail in the algorithm.
//
// Approximate, and that changes what tests can say
// ------------------------------------------------
// Unlike the B+ tree, the RMI and the flat index, this structure does not
// return exact answers, and no error bound makes it exact. "Is it correct?"
// is not a question that can be asked of it. The meaningful questions are
// what recall it reaches, at what speed, and how those trade off — so its
// tests assert *bounds and monotonicity* (raising ef must not lower recall)
// and measure recall against FlatIndex, which exists precisely to be the
// exact oracle here.
//
// Not thread-safe
// ---------------
// Search uses a mutable epoch-stamped visited array so it does not have to
// clear O(n) state per query. That makes concurrent searches on one index
// unsafe. Use one index per thread; the alternative — allocating a visited
// set per query — would reintroduce the O(n) per-query cost the whole
// structure exists to avoid.
//
// Deletion is out of scope
// ------------------------
// A node cannot be removed without degrading the graph around it: its
// neighbours lose the connectivity it was providing, and there is no local
// repair that restores navigability. Production systems use tombstones plus a
// periodic rebuild. Nothing on the hylis roadmap needs it.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/distance.hpp"
#include "index/router.hpp"

namespace hylis::index {

class HnswIndex {
public:
    struct Stats {
        std::size_t nodes = 0;
        std::size_t levels = 0;          // 1 + the highest layer in use
        std::size_t edges = 0;           // directed, summed over every layer
        std::size_t layer0_edges = 0;
        double mean_degree_l0 = 0.0;
        std::size_t max_degree_l0 = 0;
        std::int64_t entry_point = -1;
        std::vector<std::size_t> layer_population;  // nodes reaching each layer
        // Nodes actually reachable from the entry point through layer 0.
        // Deliberately a statistic and not an invariant — see reachable().
        std::size_t reachable = 0;
        std::size_t graph_bytes = 0;     // links only — the index overhead
        std::size_t total_bytes = 0;     // + the stored vectors
    };

    // M is the number of links selected per node per layer. M_max is the hard
    // degree cap: M at the upper layers, 2*M at layer 0, which carries all the
    // fine-grained connectivity and so is given the wider budget.
    //
    // ef_construction is the beam width used while inserting. It buys graph
    // quality at build time only — it is not a query-time cost.
    // `flat_only` builds the graph with no layers above 0 at all. The
    // hierarchy exists solely to supply an entry point, so once a router is
    // supplying those instead, the upper layers are dead weight — this is the
    // full-replacement configuration, and the difference in graph_bytes
    // against a normal build is what the hierarchy actually costs.
    explicit HnswIndex(std::size_t dim, Metric metric = Metric::L2,
                       std::size_t M = 16, std::size_t ef_construction = 200,
                       std::uint64_t seed = 100, bool flat_only = false)
        : dim_(dim), metric_(metric), m_(M), m_max_(M), m_max0_(2 * M),
          ef_construction_(ef_construction), seed_(seed), rng_(seed),
          flat_only_(flat_only) {
        if (dim_ == 0) throw std::invalid_argument("HnswIndex: dim must be > 0");
        if (m_ < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
        if (ef_construction_ < 1) {
            throw std::invalid_argument("HnswIndex: ef_construction must be >= 1");
        }
        // Level multiplier from the paper: 1/ln(M) makes the expected number
        // of layers ln(n)/ln(M), which is what gives the search its log
        // behaviour.
        level_multiplier_ = 1.0 / std::log(static_cast<double>(m_));
    }

    std::size_t dim() const { return dim_; }
    std::size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    Metric metric() const { return metric_; }
    std::size_t M() const { return m_; }
    std::size_t ef_construction() const { return ef_construction_; }
    std::size_t max_level() const { return count_ ? max_level_ : 0; }
    std::int64_t entry_point() const {
        return count_ ? static_cast<std::int64_t>(entry_point_) : -1;
    }

    // Default beam width when a search does not name one.
    std::size_t ef_search() const { return ef_search_; }
    void set_ef_search(std::size_t ef) { ef_search_ = std::max<std::size_t>(ef, 1); }

    // Whether insertion uses the paper's selection heuristic or simply keeps
    // the M nearest candidates. Exposed so the difference can be measured
    // rather than asserted — see select_neighbors.
    bool uses_heuristic() const { return use_heuristic_; }
    void set_use_heuristic(bool enabled) { use_heuristic_ = enabled; }

    // How many nodes the most recent search actually scored. The number that
    // explains why the graph is fast: a few hundred out of a corpus of
    // millions.
    std::size_t last_visited() const { return last_visited_; }

    // Of those, how many were spent just *getting to* layer 0 — walking the
    // hierarchy, or zero when a router supplied the entry points. This is the
    // work the router is trying to eliminate, so isolating it is what makes
    // the comparison interpretable rather than a single opaque total.
    std::size_t last_routing_visited() const { return last_routing_visited_; }

    // --- neural router ---------------------------------------------------
    //
    // Optional. With no router loaded, or with use_router=false, search
    // behaves exactly as before — the descent is the default and the router
    // is strictly additive, so the baseline cannot be broken by adding one.
    void set_router(NeuralRouter router) {
        router.require_compatible(dim_, count_);
        router_ = std::move(router);
    }
    void clear_router() { router_ = NeuralRouter{}; }
    bool has_router() const { return !router_.empty(); }
    const NeuralRouter& router() const { return router_; }

    // How many clusters' medoids seed the beam. More entry points cost more
    // distance computations but give the search several places to start from,
    // which matters when the router is unsure.
    std::size_t router_top_p() const { return router_top_p_; }
    void set_router_top_p(std::size_t p) { router_top_p_ = std::max<std::size_t>(p, 1); }

    // Whether to also seed the beam with the graph's global entry point.
    // Cheap insurance against a badly routed query: one extra distance
    // computation buys a fallback start that is never worse than random.
    bool router_keeps_global_entry() const { return router_keep_global_; }
    void set_router_keeps_global_entry(bool keep) { router_keep_global_ = keep; }

    void reserve(std::size_t n) {
        data_.reserve(n * dim_);
        levels_.reserve(n);
        layer0_.reserve(n * stride0());
        upper_.reserve(n);
        visited_.reserve(n);
    }

    void clear() {
        data_.clear();
        levels_.clear();
        layer0_.clear();
        upper_.clear();
        visited_.clear();
        count_ = 0;
        max_level_ = 0;
        entry_point_ = 0;
        epoch_ = 0;
        rng_.seed(seed_);
    }

    std::int64_t add(const std::vector<float>& vec) {
        require_dim(vec.size());
        return add(vec.data());
    }

    std::int64_t add(const float* vec) {
        const std::uint32_t id = static_cast<std::uint32_t>(count_);
        const std::size_t level = random_level();

        data_.insert(data_.end(), vec, vec + dim_);
        if (metric_ == Metric::Cosine) normalise(data_.data() + id * dim_, dim_);
        levels_.push_back(static_cast<std::uint32_t>(level));
        layer0_.resize(layer0_.size() + stride0(), 0);
        upper_.emplace_back(level * stride_upper(), 0u);
        visited_.push_back(0);
        ++count_;

        if (id == 0) {
            entry_point_ = 0;
            max_level_ = level;
            return 0;
        }

        // Safe to take a pointer into data_ now: no further insertion happens
        // during this call, and for Cosine this is the normalised form, which
        // is what the stored vectors are compared against.
        const float* query = data_.data() + static_cast<std::size_t>(id) * dim_;

        std::uint32_t entry = entry_point_;
        // Phase 1: greedy descent through the layers above this node's own,
        // with a beam of 1. Cheap, and only ever narrows the entry point.
        for (std::size_t layer = max_level_; layer > level; --layer) {
            const std::vector<Neighbor> found =
                search_layer(query, {entry}, 1, layer, AcceptAll{});
            if (!found.empty()) entry = static_cast<std::uint32_t>(found[0].id);
        }

        // Phase 2: from this node's own level down to 0, find candidates and
        // wire the node in.
        const std::size_t start = std::min(max_level_, level);
        for (std::size_t layer = start + 1; layer-- > 0;) {
            std::vector<Neighbor> candidates =
                search_layer(query, {entry}, ef_construction_, layer, AcceptAll{});
            const std::size_t cap = layer == 0 ? m_max0_ : m_max_;

            const std::vector<std::uint32_t> chosen = select_neighbors(candidates, m_);
            set_links(id, layer, chosen);

            // Links are added in both directions here, but pruning below can
            // drop one side again — so the graph is genuinely directed. See
            // validate(), which deliberately does not assert symmetry.
            for (std::uint32_t neighbour : chosen) connect(neighbour, id, layer, cap);

            if (!candidates.empty()) {
                entry = static_cast<std::uint32_t>(candidates[0].id);
            }
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_point_ = id;
        }
        return static_cast<std::int64_t>(id);
    }

    void add_batch(const float* data, std::size_t n) {
        reserve(count_ + n);
        for (std::size_t i = 0; i < n; ++i) add(data + i * dim_);
    }

    const float* vector_at(std::int64_t id) const {
        require_id(id);
        return data_.data() + static_cast<std::size_t>(id) * dim_;
    }

    std::vector<Neighbor> search(const float* query, std::size_t k,
                                 std::size_t ef = 0, bool use_router = false) const {
        return search_impl(query, k, ef, AcceptAll{}, use_router);
    }

    std::vector<Neighbor> search(const std::vector<float>& query, std::size_t k,
                                 std::size_t ef = 0, bool use_router = false) const {
        require_dim(query.size());
        return search(query.data(), k, ef, use_router);
    }

    // Search restricted to `allowed` ids.
    //
    // The graph is still traversed in full — non-matching nodes are stepped
    // *through* but never collected — because skipping them entirely would
    // disconnect the graph and strand whole regions. The cost is therefore
    // roughly ef/selectivity node visits, capped at n: as the predicate
    // tightens, this degrades toward a full scan while FlatIndex's filtered
    // scan gets cheaper in proportion. Those two curves cross, and that
    // crossover is exactly what the query planner exists to exploit — see
    // scripts/bench_vector.py, which measures where it lands.
    std::vector<Neighbor> search_filtered(const float* query, std::size_t k,
                                          const std::vector<std::int64_t>& allowed,
                                          std::size_t ef = 0,
                                          bool use_router = false) const {
        if (allowed.empty() || k == 0 || count_ == 0) return {};

        // Epoch-stamped like the visited set, so marking costs O(|allowed|)
        // rather than O(n). Clearing per query would make even a two-element
        // filter cost a full pass over the corpus.
        if (++allow_epoch_ == 0) {
            std::fill(allowed_.begin(), allowed_.end(), 0);
            allow_epoch_ = 1;
        }
        allowed_.resize(count_, 0);
        for (std::int64_t id : allowed) {
            require_id(id);
            allowed_[static_cast<std::size_t>(id)] = allow_epoch_;
        }
        const std::uint32_t epoch = allow_epoch_;
        const std::vector<std::uint32_t>& marks = allowed_;
        return search_impl(query, k, ef,
                           [&marks, epoch](std::uint32_t id) {
                               return marks[id] == epoch;
                           },
                           use_router);
    }

    std::vector<Neighbor> search_filtered(const std::vector<float>& query, std::size_t k,
                                          const std::vector<std::int64_t>& allowed,
                                          std::size_t ef = 0,
                                          bool use_router = false) const {
        require_dim(query.size());
        return search_filtered(query.data(), k, allowed, ef, use_router);
    }

    std::vector<std::vector<Neighbor>> search_batch(const float* queries,
                                                    std::size_t n_queries,
                                                    std::size_t k,
                                                    std::size_t ef = 0,
                                                    bool use_router = false) const {
        std::vector<std::vector<Neighbor>> out;
        out.reserve(n_queries);
        for (std::size_t i = 0; i < n_queries; ++i) {
            out.push_back(search(queries + i * dim_, k, ef, use_router));
        }
        return out;
    }

    Stats stats() const {
        Stats s;
        s.nodes = count_;
        s.levels = count_ ? max_level_ + 1 : 0;
        s.entry_point = entry_point();
        s.layer_population.assign(s.levels, 0);

        for (std::size_t node = 0; node < count_; ++node) {
            for (std::size_t layer = 0; layer <= levels_[node]; ++layer) {
                s.layer_population[layer] += 1;
                const std::uint32_t degree = links(node, layer)[0];
                s.edges += degree;
                if (layer == 0) {
                    s.layer0_edges += degree;
                    s.max_degree_l0 = std::max<std::size_t>(s.max_degree_l0, degree);
                }
            }
        }
        s.mean_degree_l0 = count_ ? static_cast<double>(s.layer0_edges) /
                                        static_cast<double>(count_)
                                  : 0.0;
        s.reachable = reachable();

        s.graph_bytes = layer0_.capacity() * sizeof(std::uint32_t) +
                        levels_.capacity() * sizeof(std::uint32_t);
        for (const auto& node_links : upper_) {
            s.graph_bytes += node_links.capacity() * sizeof(std::uint32_t);
        }
        s.total_bytes = s.graph_bytes + data_.capacity() * sizeof(float);
        return s;
    }

    // Structural invariants. Search *quality* is not something that can be
    // asserted here — that is what recall measurement against FlatIndex is
    // for — but the graph's shape absolutely can be.
    void validate() const {
        if (levels_.size() != count_ || upper_.size() != count_) {
            throw std::logic_error("validate: node bookkeeping out of step");
        }
        if (count_ == 0) return;

        if (levels_[entry_point_] != max_level_) {
            throw std::logic_error(
                "validate: entry point " + std::to_string(entry_point_) +
                " is at level " + std::to_string(levels_[entry_point_]) +
                " but the graph's maximum is " + std::to_string(max_level_));
        }

        for (std::size_t node = 0; node < count_; ++node) {
            if (levels_[node] > max_level_) {
                throw std::logic_error("validate: node " + std::to_string(node) +
                                       " exceeds the recorded maximum level");
            }
            for (std::size_t layer = 0; layer <= levels_[node]; ++layer) {
                const std::uint32_t* list = links(node, layer);
                const std::size_t degree = list[0];
                const std::size_t cap = layer == 0 ? m_max0_ : m_max_;
                if (degree > cap) {
                    throw std::logic_error(
                        "validate: node " + std::to_string(node) + " at layer " +
                        std::to_string(layer) + " has degree " +
                        std::to_string(degree) + ", over the cap of " +
                        std::to_string(cap));
                }
                for (std::size_t i = 1; i <= degree; ++i) {
                    const std::uint32_t other = list[i];
                    if (other >= count_) {
                        throw std::logic_error("validate: node " + std::to_string(node) +
                                               " links to nonexistent " +
                                               std::to_string(other));
                    }
                    if (other == node) {
                        throw std::logic_error("validate: node " + std::to_string(node) +
                                               " links to itself at layer " +
                                               std::to_string(layer));
                    }
                    if (levels_[other] < layer) {
                        throw std::logic_error(
                            "validate: node " + std::to_string(node) + " links at layer " +
                            std::to_string(layer) + " to " + std::to_string(other) +
                            ", which does not reach that layer");
                    }
                    for (std::size_t j = i + 1; j <= degree; ++j) {
                        if (list[j] == other) {
                            throw std::logic_error(
                                "validate: node " + std::to_string(node) +
                                " has a duplicate link to " + std::to_string(other));
                        }
                    }
                }
            }
        }

    }

    // How many nodes can actually be reached from the entry point through
    // layer 0. A node nothing links *to* can never be returned by any query,
    // however large ef gets, so this is the practical ceiling on recall.
    //
    // Not part of validate(), and that is the point. HNSW adds links in both
    // directions, but pruning a full neighbour list can drop the back-link
    // again — and if every in-edge to a node is pruned away, it is stranded.
    // Nothing in the algorithm prevents that, so full reachability is an
    // empirical property of sensible parameters rather than a structural
    // guarantee. At M=16 it is reliably 100%; at M=2, where lists fill and
    // prune constantly, a few percent of nodes get stranded. Reporting it as
    // a number rather than asserting it is the honest treatment.
    std::size_t reachable() const {
        if (count_ == 0) return 0;
        std::vector<bool> seen(count_, false);
        std::vector<std::uint32_t> stack{entry_point_};
        seen[entry_point_] = true;
        std::size_t reached = 1;
        while (!stack.empty()) {
            const std::uint32_t node = stack.back();
            stack.pop_back();
            const std::uint32_t* list = links(node, 0);
            for (std::size_t i = 1; i <= list[0]; ++i) {
                if (!seen[list[i]]) {
                    seen[list[i]] = true;
                    ++reached;
                    stack.push_back(list[i]);
                }
            }
        }
        return reached;
    }

private:
    struct AcceptAll {
        bool operator()(std::uint32_t) const { return true; }
    };

    // Min-heap ordering: priority_queue::top() is the *closest* candidate.
    struct FarthestFirst {
        bool operator()(const Neighbor& a, const Neighbor& b) const {
            return better_than(b.score, b.id, a.score, a.id);
        }
    };

    std::size_t stride0() const { return m_max0_ + 1; }        // degree + slots
    std::size_t stride_upper() const { return m_max_ + 1; }

    std::uint32_t* links(std::size_t node, std::size_t layer) {
        if (layer == 0) return layer0_.data() + node * stride0();
        return upper_[node].data() + (layer - 1) * stride_upper();
    }

    const std::uint32_t* links(std::size_t node, std::size_t layer) const {
        if (layer == 0) return layer0_.data() + node * stride0();
        return upper_[node].data() + (layer - 1) * stride_upper();
    }

    // Layer for a new node: floor(-ln(U) * mL), an exponential decay that
    // puts ~1/M of nodes on each successive layer.
    std::size_t random_level() {
        // Still drawn (and discarded) in flat_only mode so the RNG stream —
        // and therefore every other random decision — matches a normal build
        // exactly. Otherwise the two graphs would differ for two reasons at
        // once and the comparison would be worthless.
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        double u = unit(rng_);
        if (u <= 0.0) u = std::numeric_limits<double>::min();
        const std::size_t level = static_cast<std::size_t>(-std::log(u) * level_multiplier_);
        return flat_only_ ? 0 : level;
    }

    float score_to(const float* query, std::size_t node) const {
        return score_vectors(query, data_.data() + node * dim_, dim_, metric_);
    }

    float score_between(std::size_t a, std::size_t b) const {
        return score_vectors(data_.data() + a * dim_, data_.data() + b * dim_,
                             dim_, metric_);
    }

    void new_epoch() const {
        if (++epoch_ == 0) {  // wrapped; every stamp is now ambiguous
            std::fill(visited_.begin(), visited_.end(), 0);
            epoch_ = 1;
        }
        if (visited_.size() < count_) visited_.resize(count_, 0);
    }

    bool already_visited(std::uint32_t id) const {
        if (visited_[id] == epoch_) return true;
        visited_[id] = epoch_;
        return false;
    }

    // The core beam search, used at every layer and by both insertion and
    // query. `accept` decides what may enter the result set; nodes it rejects
    // are still traversed *through*, which is what keeps a filtered search
    // from disconnecting the graph.
    template <typename Accept>
    std::vector<Neighbor> search_layer(const float* query,
                                       const std::vector<std::uint32_t>& entries,
                                       std::size_t ef, std::size_t layer,
                                       Accept accept) const {
        new_epoch();
        TopK results(std::max<std::size_t>(ef, 1));
        std::priority_queue<Neighbor, std::vector<Neighbor>, FarthestFirst> frontier;

        for (std::uint32_t entry : entries) {
            if (already_visited(entry)) continue;
            const float d = score_to(query, entry);
            ++last_visited_;
            frontier.push({static_cast<std::int64_t>(entry), d});
            if (accept(entry)) results.offer(entry, d);
        }

        while (!frontier.empty()) {
            const Neighbor closest = frontier.top();
            // Nothing further out can improve a full result set: the frontier
            // is ordered by distance, so if its best is already worse than our
            // worst keeper, every remaining candidate is too.
            if (results.full() &&
                !better_than(closest.score, closest.id,
                             results.worst().score, results.worst().id)) {
                break;
            }
            frontier.pop();

            const std::uint32_t* list = links(static_cast<std::size_t>(closest.id), layer);
            const std::size_t degree = list[0];
            for (std::size_t i = 1; i <= degree; ++i) {
                const std::uint32_t candidate = list[i];
                if (already_visited(candidate)) continue;
                const float d = score_to(query, candidate);
                ++last_visited_;
                const bool worth_exploring =
                    !results.full() ||
                    better_than(d, candidate, results.worst().score, results.worst().id);
                if (worth_exploring) {
                    frontier.push({static_cast<std::int64_t>(candidate), d});
                    if (accept(candidate)) results.offer(candidate, d);
                }
            }
        }
        return results.drain();
    }

    // Algorithm 4 from the paper, and the detail that makes the graph
    // navigable rather than merely connected.
    //
    // Keeping the M nearest candidates sounds obviously right and is not: in a
    // dense cluster, all M of them are near each other as well as near the new
    // node, so they carry almost the same information and the node ends up
    // with no way out of its own neighbourhood. The heuristic instead drops a
    // candidate when it is closer to an already-selected neighbour than to the
    // node itself — that candidate is already "covered", and the slot is
    // better spent on a direction not yet represented. The result keeps
    // long-range links that let greedy search escape local minima.
    //
    // set_use_heuristic(false) selects the naive M-nearest instead, so the
    // difference shows up as a recall number rather than an assertion.
    // The candidates already carry their distance to the node being wired in,
    // which is everything the heuristic needs — hence no query pointer here.
    std::vector<std::uint32_t> select_neighbors(const std::vector<Neighbor>& candidates,
                                                std::size_t m) const {
        std::vector<std::uint32_t> chosen;
        chosen.reserve(std::min(m, candidates.size()));

        if (!use_heuristic_) {
            for (std::size_t i = 0; i < candidates.size() && chosen.size() < m; ++i) {
                chosen.push_back(static_cast<std::uint32_t>(candidates[i].id));
            }
            return chosen;
        }

        // candidates arrive nearest-first, which the heuristic depends on:
        // each one is judged against the closer neighbours already accepted.
        for (const Neighbor& candidate : candidates) {
            if (chosen.size() >= m) break;
            bool covered = false;
            for (std::uint32_t accepted : chosen) {
                if (score_between(static_cast<std::size_t>(candidate.id), accepted) <
                    candidate.score) {
                    covered = true;
                    break;
                }
            }
            if (!covered) chosen.push_back(static_cast<std::uint32_t>(candidate.id));
        }
        return chosen;
    }

    void set_links(std::size_t node, std::size_t layer,
                   const std::vector<std::uint32_t>& neighbours) {
        std::uint32_t* list = links(node, layer);
        const std::size_t cap = layer == 0 ? m_max0_ : m_max_;
        const std::size_t n = std::min(neighbours.size(), cap);
        list[0] = static_cast<std::uint32_t>(n);
        for (std::size_t i = 0; i < n; ++i) list[i + 1] = neighbours[i];
    }

    // Add `to` to `from`'s neighbour list, re-selecting if that overflows.
    void connect(std::uint32_t from, std::uint32_t to, std::size_t layer,
                 std::size_t cap) {
        std::uint32_t* list = links(from, layer);
        const std::size_t degree = list[0];
        for (std::size_t i = 1; i <= degree; ++i) {
            if (list[i] == to) return;  // already linked
        }

        if (degree < cap) {
            list[degree + 1] = to;
            list[0] = static_cast<std::uint32_t>(degree + 1);
            return;
        }

        // Full: re-run the selection over the existing neighbours plus the new
        // one, so the survivors are the diverse set rather than whichever
        // happened to arrive first. This is where a previously added link can
        // be dropped, leaving the edge asymmetric.
        std::vector<Neighbor> pool;
        pool.reserve(degree + 1);
        for (std::size_t i = 1; i <= degree; ++i) {
            pool.push_back({static_cast<std::int64_t>(list[i]),
                            score_between(from, list[i])});
        }
        pool.push_back({static_cast<std::int64_t>(to), score_between(from, to)});
        std::sort(pool.begin(), pool.end(), [](const Neighbor& a, const Neighbor& b) {
            return better_than(a.score, a.id, b.score, b.id);
        });

        set_links(from, layer, select_neighbors(pool, cap));
    }

    template <typename Accept>
    std::vector<Neighbor> search_impl(const float* query, std::size_t k,
                                      std::size_t ef, Accept accept,
                                      bool use_router) const {
        last_visited_ = 0;
        last_routing_visited_ = 0;
        if (count_ == 0 || k == 0) return {};

        std::vector<float> scratch;
        const float* q = query;
        if (metric_ == Metric::Cosine) {
            scratch.assign(query, query + dim_);
            normalise(scratch.data(), dim_);
            q = scratch.data();
        }

        const std::size_t beam = std::max(ef ? ef : ef_search_, k);

        // Everything above layer 0 exists only to produce these entry points.
        // The router replaces that walk with one forward pass; the descent is
        // what it has to beat.
        std::vector<std::uint32_t> entries;
        if (use_router && !router_.empty()) {
            router_.entry_points(q, router_top_p_, entries);
            if (router_keep_global_ || entries.empty()) entries.push_back(entry_point_);
        } else {
            std::uint32_t entry = entry_point_;
            for (std::size_t layer = max_level_; layer > 0; --layer) {
                const std::vector<Neighbor> found =
                    search_layer(q, {entry}, 1, layer, AcceptAll{});
                if (!found.empty()) entry = static_cast<std::uint32_t>(found[0].id);
            }
            entries.push_back(entry);
        }
        // Whatever was spent above layer 0 is the routing cost — zero graph
        // nodes for the router, since its cost is arithmetic rather than
        // traversal, and that difference is exactly the thing being measured.
        last_routing_visited_ = last_visited_;

        std::vector<Neighbor> found = search_layer(q, entries, beam, 0, accept);
        if (found.size() > k) found.resize(k);
        for (Neighbor& n : found) n.score = present_score(n.score, metric_);
        return found;
    }

    void require_dim(std::size_t got) const {
        if (got != dim_) {
            throw std::invalid_argument(
                "HnswIndex: expected a " + std::to_string(dim_) +
                "-dimensional vector, got " + std::to_string(got));
        }
    }

    void require_id(std::int64_t id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= count_) {
            throw std::out_of_range(
                "HnswIndex: id " + std::to_string(id) + " out of range [0, " +
                std::to_string(count_) + ")");
        }
    }

    std::size_t dim_;
    Metric metric_;
    std::size_t m_, m_max_, m_max0_;
    std::size_t ef_construction_;
    std::uint64_t seed_;
    std::mt19937_64 rng_;
    double level_multiplier_ = 1.0;
    bool use_heuristic_ = true;
    std::size_t ef_search_ = 50;

    std::vector<float> data_;
    std::vector<std::uint32_t> levels_;
    std::vector<std::uint32_t> layer0_;
    std::vector<std::vector<std::uint32_t>> upper_;
    std::size_t count_ = 0;
    std::size_t max_level_ = 0;
    std::uint32_t entry_point_ = 0;

    // Search scratch. Epoch-stamped so a query costs O(visited) rather than
    // O(n) to reset — which is the whole point of a graph index.
    mutable std::vector<std::uint32_t> visited_;
    mutable std::uint32_t epoch_ = 0;
    mutable std::vector<std::uint32_t> allowed_;
    mutable std::uint32_t allow_epoch_ = 0;
    mutable std::size_t last_visited_ = 0;
    mutable std::size_t last_routing_visited_ = 0;

    bool flat_only_ = false;
    NeuralRouter router_;
    std::size_t router_top_p_ = 2;
    bool router_keep_global_ = false;
};

}  // namespace hylis::index
