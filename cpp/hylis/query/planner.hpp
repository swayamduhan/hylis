// query/planner.hpp
//
// The hybrid query planner — the piece that makes this a *system* rather than
// three indexes in one repository.
//
// The query it answers
// --------------------
//     SELECT id FROM t
//     WHERE  category < 40                  -- structured predicate
//     ORDER BY distance(embedding, $q)      -- vector similarity
//     LIMIT  10
//
// Two indexes, one query. Everything needed to serve it already existed
// before this file:
//
//   * ColumnIndex::range_query(op, value) returns matching row ids, from a
//     B+ tree, an RMI or a dynamic RMI, without the caller knowing which.
//   * FlatIndex and HnswIndex both take search_filtered(query, k, allowed).
//
// The row ids the first produces are exactly the `allowed` the second wants.
// Nothing joined them, so every caller of search_filtered built its filter by
// hand. This file is that join, plus the decision of how to execute it.
//
// The decision is measured, not assumed
// -------------------------------------
// Module 5 measured the two curves this planner chooses between:
//
//   * A filtered exhaustive scan is O(|allowed|) and exact, so it gets
//     *cheaper* as a predicate tightens.
//   * A filtered graph search must step through non-matching nodes to stay
//     connected, so it gets *more expensive* — at 0.1% selectivity it visits
//     all 10,004 nodes for 1502 us against the scan's 3 us.
//
// They cross around 50% selectivity, and below it the scan wins outright —
// by 50x at 0.5%. That crossover is the whole cost model, and it came from
// measurement rather than from assumed cache-miss constants. Same stance
// choose_index() takes one level down.
//
// What this planner does not do
// -----------------------------
// It does not estimate selectivity from a histogram. It runs the predicate
// through ColumnIndex first and therefore knows |allowed| exactly. A real
// optimiser estimates because it must plan before executing; here the
// structured lookup is single-digit nanoseconds and the row ids are work the
// query needs regardless, so executing the predicate *is* the estimate. No
// histogram can beat the actual answer.
//
// The cost of that choice is real and worth stating: a predicate matching
// nearly everything is paid for in full before the planner can discover it
// should have post-filtered. That is the one case where estimation would win,
// and scripts/bench_planner.py measures it rather than arguing about it.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/column_index.hpp"
#include "index/compare_op.hpp"
#include "index/distance.hpp"
#include "index/flat.hpp"
#include "index/hnsw.hpp"
#include "query/predicate.hpp"

namespace hylis::query {

using hylis::index::Bitset;
using hylis::index::ColumnIndex;
using hylis::index::ColumnKey;
using hylis::index::ColumnValue;
using hylis::index::CompareOp;
using hylis::index::Datum;
using hylis::index::FlatIndex;
using hylis::index::HnswIndex;
using hylis::index::Neighbor;

// The predicate type is query/predicate.hpp's, shared with Table.
//
// It used to be a separate struct here, holding a CompareOp and a bare int64.
// Two Predicate types in one namespace could not both be included, so the
// planner and the table could not appear in the same translation unit -- which
// is exactly what a demo of the whole system has to do. Unifying them also
// gives the planner Prefix and Between for free, and makes the value typed
// rather than "whatever int64 the caller encoded it as".
//
// The single-predicate restriction is lifted with it. The original reason --
// that two predicates are set intersection and introduce no new decision --
// stopped being true when bitmaps arrived: a word-parallel AND costs O(n/64)
// whatever matches, where a merge costs O(m1 + m2). Table owns that choice
// (see select_all); the planner consumes its answer.

enum class PlanKind {
    NoPredicate,          // unconstrained similarity search
    PreFilter,            // predicate, then an exact scan of the survivors
    FilteredGraph,        // predicate, then a beam search rejecting non-matches
    BitmapFilteredGraph,  // the same, with the bit set used directly as the mask
    PostFilter,           // unfiltered graph search, then drop non-matches
};

inline const char* to_string(PlanKind kind) {
    switch (kind) {
        case PlanKind::NoPredicate: return "no_predicate";
        case PlanKind::PreFilter: return "pre_filter";
        case PlanKind::FilteredGraph: return "filtered_graph";
        case PlanKind::BitmapFilteredGraph: return "bitmap_filtered_graph";
        case PlanKind::PostFilter: return "post_filter";
    }
    return "?";
}

// The decision, and the evidence for it. `reason` exists because a planner
// that cannot say why it chose something is not defensible in a report, and
// because the demo CLI should be able to print it.
struct QueryPlan {
    PlanKind kind = PlanKind::NoPredicate;
    std::size_t matched_rows = 0;
    std::size_t corpus_rows = 0;
    double selectivity = 1.0;
    double threshold = 0.5;
    std::string reason;

    // Whether the selectivity above was known without executing the predicate.
    //
    // This file already states its own weakness: the planner knows selectivity
    // exactly because it *executes* first, so "a predicate matching nearly
    // everything is paid for in full before the planner can discover it should
    // have post-filtered". A bitmap column answers by popcount and materialises
    // nothing, which is the case where that weakness does not apply -- and the
    // flag is how a caller can tell the two apart.
    bool selectivity_was_free = false;
};

class HybridPlanner {
public:
    // Selectivity at or below which a filtered exhaustive scan is preferred.
    //
    // Defaults to the measured ~50% crossover. It is a single constant, and
    // that is the point: it is the honest baseline a learned cost model has
    // to beat, and building it first is what makes the learned version
    // falsifiable rather than merely plausible.
    explicit HybridPlanner(double prefilter_threshold = 0.5)
        : threshold_(prefilter_threshold) {}

    double prefilter_threshold() const { return threshold_; }
    void set_prefilter_threshold(double t) { threshold_ = t; }

    // Attach a column. Copied in, so the planner owns a complete answer to
    // any predicate over it.
    void set_column(const std::string& name, ColumnIndex index) {
        columns_.insert_or_assign(name, std::move(index));
    }
    bool has_column(const std::string& name) const {
        return columns_.count(name) != 0;
    }
    std::vector<std::string> columns() const {
        std::vector<std::string> out;
        out.reserve(columns_.size());
        for (const auto& [name, _] : columns_) out.push_back(name);
        return out;
    }

    // Borrowed, not owned: the vector indexes are large and the caller
    // already holds them. Their lifetime must outlast the planner.
    void set_exact(const FlatIndex* exact) { exact_ = exact; }
    void set_graph(const HnswIndex* graph) { graph_ = graph; }
    const FlatIndex* exact() const { return exact_; }
    const HnswIndex* graph() const { return graph_; }

    // Find this machine's actual crossover and adopt it.
    //
    // The ~50% default came from one measurement on SIFT10K at one ef. The
    // crossover is not a constant of the algorithm: it moves with n, with the
    // dimensionality, with ef, and with how fast the machine's cache is.
    // Measured on 20,000 32-d vectors at ef=64 it sits nearer 15-25%, and a
    // planner carrying the inherited 50% is wrong on rows in between — by up
    // to 2.97x at the worst point.
    //
    // So the same answer choose_index() gives one level down: build both, time
    // both, keep what won. Costs a few hundred searches once, and replaces a
    // constant that was only ever right for the corpus it was measured on.
    //
    // `sample` predicates are tried between 0 and 1; the threshold is set to
    // the midpoint of the last selectivity where the scan won and the first
    // where the graph did.
    double calibrate(const std::string& column, const float* queries,
                     std::size_t n_queries, std::size_t k, std::size_t ef = 0,
                     std::size_t samples = 12) {
        if (!exact_ || !graph_ || n_queries == 0 || samples < 2) return threshold_;
        const auto it = columns_.find(column);
        if (it == columns_.end()) {
            throw std::invalid_argument("HybridPlanner::calibrate: no column '" +
                                        column + "'");
        }
        const std::size_t dim = graph_->dim();
        const std::size_t n = corpus_size();
        if (n == 0) return threshold_;

        // All row ids, so a synthetic filter of any size can be cut from it
        // without depending on what the column's keys happen to look like.
        std::vector<ColumnValue> all = it->second.range_query(
            CompareOp::Ge, std::numeric_limits<ColumnKey>::min());

        double last_scan_win = 0.0;
        double first_graph_win = 1.0;
        bool seen_graph_win = false;

        for (std::size_t s = 1; s <= samples; ++s) {
            const double selectivity = static_cast<double>(s) / static_cast<double>(samples);
            const std::size_t take = static_cast<std::size_t>(
                selectivity * static_cast<double>(all.size()));
            if (take <= k) continue;
            const std::vector<ColumnValue> subset(all.begin(),
                                                  all.begin() + static_cast<std::ptrdiff_t>(take));

            const double scan = time_plan(PlanKind::PreFilter, subset, queries,
                                          n_queries, dim, k, ef);
            const double graph = time_plan(PlanKind::FilteredGraph, subset, queries,
                                           n_queries, dim, k, ef);
            if (scan <= graph) {
                last_scan_win = selectivity;
            } else if (!seen_graph_win) {
                first_graph_win = selectivity;
                seen_graph_win = true;
            }
        }

        // No crossover in range means one plan won everywhere; keeping the
        // current threshold is more honest than inventing a boundary.
        if (!seen_graph_win) return threshold_;
        threshold_ = 0.5 * (last_scan_win + first_graph_win);
        return threshold_;
    }

    // Run the predicate and decide, without doing the vector work.
    QueryPlan explain(const Predicate& predicate, std::size_t k) const {
        std::vector<ColumnValue> matched;
        Bitset mask;
        return plan_for(predicate, k, &matched, &mask);
    }

    // The whole point of the module.
    std::vector<Neighbor> search(const Predicate& predicate, const float* query,
                                 std::size_t k, std::size_t ef = 0,
                                 QueryPlan* out_plan = nullptr) const {
        std::vector<ColumnValue> matched;
        Bitset mask;
        const QueryPlan plan = plan_for(predicate, k, &matched, &mask);
        if (out_plan) *out_plan = plan;
        // Every plan but the bitmap one needs the rows themselves, and a free
        // selectivity means they were never produced. This is where the saving
        // is given back when the plan turns out to want them.
        if (plan.kind != PlanKind::BitmapFilteredGraph && matched.empty() &&
            plan.matched_rows != 0) {
            matched = matching_rows(predicate);
        }
        return execute(plan.kind, matched, mask, query, k, ef);
    }

    // Force a plan. Exists so tests can assert every plan returns the same
    // rows — without it the planner could pick a fast *wrong* plan and
    // nothing in the suite would notice, which is the failure mode a query
    // optimiser must never have.
    std::vector<Neighbor> search_with(PlanKind kind, const Predicate& predicate,
                                      const float* query, std::size_t k,
                                      std::size_t ef = 0) const {
        if (kind == PlanKind::BitmapFilteredGraph) {
            return execute(kind, {}, matching_mask(predicate), query, k, ef);
        }
        const std::vector<ColumnValue> matched = matching_rows(predicate);
        return execute(kind, matched, Bitset(), query, k, ef);
    }

    // Whether a plan can run at all for this predicate. BitmapFilteredGraph
    // needs a bitmap column whose positions are the corpus's row ids.
    bool plan_available(PlanKind kind, const Predicate& predicate) const {
        if (kind != PlanKind::BitmapFilteredGraph) return true;
        return matching_mask(predicate).size() != 0;
    }

    // Row ids satisfying the predicate. The structured half on its own,
    // exposed because it is also the answer to a plain SQL query.
    //
    // Ordered by the column's *key*, which is the attribute — not by row id.
    // For a column built over an encoded attribute those differ, and assuming
    // otherwise is exactly the bug the post-filter plan had. Callers needing
    // row order must sort; the filtered searches do not care, which is why the
    // common path is not made to pay for it.
    std::vector<ColumnValue> matching_rows(const Predicate& predicate) const {
        const ColumnIndex& column = column_for(predicate.column);
        switch (predicate.op) {
            case PredOp::Between:
                return column.query_range(predicate.value, predicate.value2);
            case PredOp::Prefix:
                return column.query_prefix(std::get<std::string>(predicate.value));
            case PredOp::Contains:
            case PredOp::IsNull:
                throw std::invalid_argument(
                    std::string("HybridPlanner: '") + to_string(predicate.op) +
                    "' cannot be served by an index. Table answers it by "
                    "scanning; pass the resulting row ids to search_rows().");
            default:
                return column.query(compare_op_of(predicate.op), predicate.value);
        }
    }

    // The rows as a bit set, when the column can produce one aligned to the
    // vector index's ids. Empty otherwise.
    Bitset matching_mask(const Predicate& predicate) const {
        const ColumnIndex& column = column_for(predicate.column);
        if (!can_mask(column) || !op_is_indexable(predicate.op) ||
            predicate.op == PredOp::Between || predicate.op == PredOp::Prefix) {
            return Bitset();
        }
        return column.bitmap_for(compare_op_of(predicate.op), predicate.value);
    }

    // Search a set of row ids the caller already has.
    //
    // The seam for everything the planner cannot express: a conjunction
    // resolved by Table, a Contains that had to be scanned, a hand-built
    // filter. The plan choice is the same; only the source of the rows differs.
    std::vector<Neighbor> search_rows(const std::vector<ColumnValue>& rows,
                                      const float* query, std::size_t k,
                                      std::size_t ef = 0,
                                      QueryPlan* out_plan = nullptr) const {
        QueryPlan plan = plan_for_rows(rows.size(), k);
        if (out_plan) *out_plan = plan;
        return execute(plan.kind, rows, Bitset(), query, k, ef);
    }

private:
    const ColumnIndex& column_for(const std::string& name) const {
        const auto it = columns_.find(name);
        if (it == columns_.end()) {
            throw std::invalid_argument(
                "HybridPlanner: no column named '" + name +
                "'. Known columns: " + joined_columns());
        }
        return it->second;
    }

    // A bit set is only usable as a vector filter when position i means row id
    // i and the set covers the corpus. Both are checkable, and being wrong
    // would filter on entirely different rows while looking plausible.
    bool can_mask(const ColumnIndex& column) const {
        return column.has_bitmap() && column.rows_are_dense() &&
               column.row_space() == corpus_size();
    }

    std::size_t corpus_size() const {
        if (exact_) return exact_->size();
        if (graph_) return graph_->size();
        return 0;
    }

    std::string joined_columns() const {
        std::string out;
        for (const auto& [name, _] : columns_) {
            if (!out.empty()) out += ", ";
            out += name;
        }
        return out.empty() ? "(none)" : out;
    }

    // Everything below the vector work: how many rows match, and how much it
    // cost to find out.
    //
    // A bitmap column aligned to the corpus answers by popcount and keeps the
    // mask, so nothing is materialised unless a plan turns out to need it.
    // Every other column has to execute, which is the weakness this file
    // states at the top and the one case where estimation would have won.
    QueryPlan plan_for(const Predicate& predicate, std::size_t k,
                       std::vector<ColumnValue>* matched, Bitset* mask) const {
        QueryPlan plan;
        plan.corpus_rows = corpus_size();
        plan.threshold = threshold_;

        *mask = matching_mask(predicate);
        if (mask->size() != 0) {
            plan.matched_rows = mask->count();
            plan.selectivity_was_free = true;
        } else {
            *matched = matching_rows(predicate);
            plan.matched_rows = matched->size();
        }
        plan.selectivity =
            plan.corpus_rows ? static_cast<double>(plan.matched_rows) /
                               static_cast<double>(plan.corpus_rows)
                             : 0.0;

        if (plan.matched_rows == 0) {
            // Nothing matches, so no vector work is worth doing at all. This
            // is the cheapest possible plan and the structured index alone
            // established it.
            plan.kind = PlanKind::PreFilter;
            plan.reason = "predicate matches no rows; no vector search needed";
            return plan;
        }

        if (!graph_) {
            plan.kind = PlanKind::PreFilter;
            plan.reason = "no graph attached; the exact index is the only option";
            return plan;
        }

        // Fewer survivors than the caller wants back: every one of them is in
        // the answer, so the only question is their order, and scanning them
        // is both exact and cheapest.
        if (plan.matched_rows <= k) {
            plan.kind = PlanKind::PreFilter;
            plan.reason = "matches (" + std::to_string(plan.matched_rows) +
                          ") do not exceed k (" + std::to_string(k) +
                          "); every match is in the answer";
            return plan;
        }

        if (plan.selectivity <= threshold_) {
            plan.kind = PlanKind::PreFilter;
            plan.reason = "selectivity " + two_dp(plan.selectivity * 100.0) +
                          "% is at or below the " + two_dp(threshold_ * 100.0) +
                          "% crossover; an exact scan of the survivors is "
                          "cheaper than making the graph step through them";
            return plan;
        }

        // The bit set is already the mark array a filtered traversal wants,
        // so this skips decoding the matches into a vector and stamping every
        // one of them -- O(matches) of setup per query, before the search
        // starts. Not a faster membership test: the id-list path already tests
        // in O(1) against an epoch array.
        if (mask->size() != 0) {
            plan.kind = PlanKind::BitmapFilteredGraph;
            plan.reason = "selectivity " + two_dp(plan.selectivity * 100.0) +
                          "% is above the " + two_dp(threshold_ * 100.0) +
                          "% crossover, and the column is a bitmap aligned to "
                          "the corpus; the graph takes the bit set directly, "
                          "so no row id is materialised at all";
            return plan;
        }

        plan.kind = PlanKind::FilteredGraph;
        plan.reason = "selectivity " + two_dp(plan.selectivity * 100.0) +
                      "% is above the " + two_dp(threshold_ * 100.0) +
                      "% crossover; the graph rejects few enough nodes to "
                      "stay worth its speedup";
        return plan;
    }

    // The same decision for rows a caller already has -- a conjunction Table
    // resolved, or a predicate no index could serve.
    QueryPlan plan_for_rows(std::size_t matched, std::size_t k) const {
        QueryPlan plan;
        plan.corpus_rows = corpus_size();
        plan.threshold = threshold_;
        plan.matched_rows = matched;
        plan.selectivity = plan.corpus_rows
                               ? static_cast<double>(matched) /
                                     static_cast<double>(plan.corpus_rows)
                               : 0.0;
        if (matched == 0 || !graph_ || matched <= k ||
            plan.selectivity <= threshold_) {
            plan.kind = PlanKind::PreFilter;
            plan.reason = "selectivity " + two_dp(plan.selectivity * 100.0) +
                          "% over rows supplied by the caller";
            return plan;
        }
        plan.kind = PlanKind::FilteredGraph;
        plan.reason = "selectivity " + two_dp(plan.selectivity * 100.0) +
                      "% over rows supplied by the caller";
        return plan;
    }

    std::vector<Neighbor> execute(PlanKind kind,
                                  const std::vector<ColumnValue>& matched,
                                  const Bitset& mask, const float* query,
                                  std::size_t k, std::size_t ef) const {
        if (k == 0) return {};

        switch (kind) {
            case PlanKind::NoPredicate:
                if (graph_) return graph_->search(query, k, ef);
                if (exact_) return exact_->search(query, k);
                throw std::logic_error("HybridPlanner: no vector index attached");

            case PlanKind::PreFilter: {
                if (!exact_) {
                    throw std::logic_error(
                        "HybridPlanner: the pre-filter plan needs the exact "
                        "index, which was never attached");
                }
                return exact_->search_filtered(query, k, matched);
            }

            case PlanKind::FilteredGraph: {
                if (!graph_) {
                    throw std::logic_error(
                        "HybridPlanner: the filtered-graph plan needs the "
                        "graph, which was never attached");
                }
                return graph_->search_filtered(query, k, matched, ef);
            }

            case PlanKind::BitmapFilteredGraph: {
                if (!graph_) {
                    throw std::logic_error(
                        "HybridPlanner: the bitmap-filtered-graph plan needs "
                        "the graph, which was never attached");
                }
                if (mask.size() == 0) {
                    // Forced by a caller comparing plans on a column that has
                    // no bit set to give. Saying so beats answering with a
                    // silently different filter.
                    throw std::invalid_argument(
                        "HybridPlanner: this column cannot produce a bit set "
                        "aligned to the corpus, so the bitmap-filtered-graph "
                        "plan is not available for it");
                }
                return graph_->search_masked(query, k, mask, ef);
            }

            case PlanKind::PostFilter: {
                // Included because it is the trap, not because it is expected
                // to win. It is what a system without a planner does, and it
                // can return fewer than k rows however large the corpus is:
                // widening the search until k survive is unbounded work, so
                // this asks for a fixed multiple and accepts the shortfall.
                //
                // Having it available and measured is what makes choosing
                // against it a result rather than an assumption.
                if (!graph_) {
                    throw std::logic_error(
                        "HybridPlanner: the post-filter plan needs the graph, "
                        "which was never attached");
                }
                const std::size_t over = std::min(
                    graph_->size(), std::max<std::size_t>(k * kPostFilterFanout, k));
                const std::vector<Neighbor> raw = graph_->search(query, over, ef);

                // `matched` is ascending by *key*, which is not ascending by
                // row id: a column's values are row ids in attribute order, and
                // only coincidentally in row order. Sorting a copy is what makes
                // the membership test below valid.
                //
                // Found by scripts/bench_planner.py rather than by any test —
                // every C++ fixture happened to use a column whose values were
                // already ascending, so the binary search always succeeded and
                // the bug was invisible until a real attribute column was used.
                std::vector<ColumnValue> sorted = matched;
                std::sort(sorted.begin(), sorted.end());

                std::vector<Neighbor> out;
                out.reserve(k);
                for (const Neighbor& n : raw) {
                    if (out.size() == k) break;
                    if (std::binary_search(sorted.begin(), sorted.end(),
                                           static_cast<ColumnValue>(n.id))) {
                        out.push_back(n);
                    }
                }
                return out;
            }
        }
        return {};
    }

    // Best of three passes over the query set. The minimum is the run least
    // interfered with, which is the fair comparison between two plans whose
    // difference can be smaller than the scheduling noise on one pass.
    double time_plan(PlanKind kind, const std::vector<ColumnValue>& allowed,
                     const float* queries, std::size_t n_queries,
                     std::size_t dim, std::size_t k, std::size_t ef) const {
        double best = std::numeric_limits<double>::infinity();
        for (int pass = 0; pass < 3; ++pass) {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t q = 0; q < n_queries; ++q) {
                const float* query = queries + q * dim;
                if (kind == PlanKind::PreFilter) {
                    volatile auto r = exact_->search_filtered(query, k, allowed);
                    (void)r;
                } else {
                    volatile auto r = graph_->search_filtered(query, k, allowed, ef);
                    (void)r;
                }
            }
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            best = std::min(best, seconds);
        }
        return best;
    }

    static std::string two_dp(double v) {
        // std::to_string on a double gives six decimals, which reads badly in
        // an explanation string. This is a percentage in a sentence, not a
        // measurement, so two places is the right precision.
        const long long scaled = static_cast<long long>(v * 100.0 + (v < 0 ? -0.5 : 0.5));
        std::string s = std::to_string(scaled / 100);
        const long long frac = scaled % 100 < 0 ? -(scaled % 100) : scaled % 100;
        s += ".";
        if (frac < 10) s += "0";
        s += std::to_string(frac);
        return s;
    }

    // How many extra candidates the post-filter plan asks the graph for.
    // Arbitrary, and that is the point: no multiple is correct, which is the
    // structural problem with the plan.
    static constexpr std::size_t kPostFilterFanout = 10;

    double threshold_;
    std::map<std::string, ColumnIndex> columns_;
    const FlatIndex* exact_ = nullptr;
    const HnswIndex* graph_ = nullptr;
};

}  // namespace hylis::query
