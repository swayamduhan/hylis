// index/bitmap.hpp
//
// A dictionary-encoded bitmap index, for columns with few distinct values.
//
// Why a third index family
// ------------------------
// A B+ tree and a learned index both answer "where is this key" by locating a
// position. That is the right question for a column of prices and the wrong
// one for a column of three categories, where the answer is a quarter of the
// table and locating it was never the hard part.
//
// Consider `in_stock` over a million rows. A composite-key tree holds a
// million (value, row) pairs — an ordered structure whose ordering does no
// work, because there are two distinct keys and the row id in the key does all
// the tie-breaking. Worse, answering `count(in_stock = true)` means walking and
// materialising 800,000 row ids to find out how many there are.
//
// This stores the same information as one bitmap per distinct value:
//
//     dictionary   sorted distinct values, each with a dense code 0..d-1
//     bitmaps      one per code, n bits, bit i set iff row i has that value
//
// Equality is then a dictionary lookup and one bitmap. A range is an OR over a
// *contiguous run* of codes, which is what keeping the dictionary sorted buys.
// Conjunction is a word-parallel AND. And cardinality is a popcount, which is
// the asymmetry the whole family exists for: **it answers how many rows match
// without materialising a single row id.**
//
// The cost, stated plainly
// ------------------------
// Memory is `d * n / 8` bytes plus the row table — linear in n, but also
// linear in *d*. At two distinct values that is 250 KB per million rows; at a
// thousand it is 125 MB, and the tree, which is independent of d, wins
// outright. There is a crossover and scripts/experiment_bitmap_cardinality.py
// measures where, rather than this header asserting one.
//
// The row table is the real floor
// -------------------------------
// Bitmaps are indexed by dense position 0..n-1 while this project's row ids
// are record keys, which are arbitrary int64. So a position -> record key
// table is needed, and at 8 bytes per row it dominates the bitmaps themselves
// for any small d. It is kept *sorted*, which makes decoding a bitmap produce
// record keys already in ascending order — exactly what Table wants — and
// makes key -> position a binary search rather than a hash map that would cost
// more than everything else here combined.
//
// When the record keys happen to be exactly 0..n-1, which is what an
// auto-increment id looks like, the table is dropped entirely and position is
// the key. That is the common case and it is worth the ten lines.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "index/key_bytes.hpp"

namespace hylis::index {

namespace bit_detail {

inline int popcount64(std::uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(v);
#else
    // Portable fallback. Never taken on the toolchain this project builds
    // with, and cheaper to carry than to discover missing.
    int n = 0;
    while (v) { v &= v - 1; ++n; }
    return n;
#endif
}

inline int trailing_zeros64(std::uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(v);
#else
    int n = 0;
    while ((v & 1u) == 0) { v >>= 1; ++n; }
    return n;
#endif
}

}  // namespace bit_detail

// A fixed-width bit set over `bits` positions.
//
// Deliberately plain: no compression. Roaring or run-length encoding would
// move the crossover with the tree to the right, and the decision to add one
// should rest on measured density rather than on the general knowledge that
// compression exists. experiment_bitmap_cardinality.py reports bytes per row
// against density for exactly that reason.
class Bitset {
public:
    Bitset() = default;
    explicit Bitset(std::size_t bits)
        : words_((bits + 63) / 64, 0), bits_(bits) {}

    std::size_t size() const { return bits_; }
    bool empty() const { return bits_ == 0; }

    void set(std::size_t i) { words_[i >> 6] |= (std::uint64_t{1} << (i & 63)); }
    void clear(std::size_t i) { words_[i >> 6] &= ~(std::uint64_t{1} << (i & 63)); }
    bool test(std::size_t i) const {
        return (words_[i >> 6] >> (i & 63)) & 1u;
    }

    void push_back(bool value) {
        if ((bits_ & 63) == 0) words_.push_back(0);
        ++bits_;
        if (value) set(bits_ - 1);
        else clear(bits_ - 1);
    }

    // How many rows match, without producing any of them. The operation the
    // whole family exists for.
    std::size_t count() const {
        std::size_t total = 0;
        for (std::uint64_t w : words_) {
            total += static_cast<std::size_t>(bit_detail::popcount64(w));
        }
        return total;
    }

    Bitset& operator|=(const Bitset& other) {
        require_same(other);
        for (std::size_t i = 0; i < words_.size(); ++i) words_[i] |= other.words_[i];
        return *this;
    }
    Bitset& operator&=(const Bitset& other) {
        require_same(other);
        for (std::size_t i = 0; i < words_.size(); ++i) words_[i] &= other.words_[i];
        return *this;
    }

    // Complement, masked to the row count so the padding bits in the last word
    // cannot become set — they would be counted and decoded as phantom rows.
    void flip() {
        for (std::uint64_t& w : words_) w = ~w;
        mask_tail();
    }

    void reset() { std::fill(words_.begin(), words_.end(), std::uint64_t{0}); }

    // Ascending set positions. `fn` takes std::size_t.
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (std::size_t w = 0; w < words_.size(); ++w) {
            std::uint64_t bits = words_[w];
            while (bits) {
                const int offset = bit_detail::trailing_zeros64(bits);
                fn(w * 64 + static_cast<std::size_t>(offset));
                bits &= bits - 1;  // clear the lowest set bit
            }
        }
    }

    std::size_t memory_bytes() const {
        return words_.capacity() * sizeof(std::uint64_t);
    }

private:
    void mask_tail() {
        const std::size_t used = bits_ & 63;
        if (used != 0 && !words_.empty()) {
            words_.back() &= (std::uint64_t{1} << used) - 1;
        }
    }

    void require_same(const Bitset& other) const {
        if (other.bits_ != bits_) {
            throw std::invalid_argument(
                "Bitset: cannot combine sets over " + std::to_string(bits_) +
                " and " + std::to_string(other.bits_) + " rows");
        }
    }

    std::vector<std::uint64_t> words_;
    std::size_t bits_ = 0;
};

// A dictionary-encoded bitmap index over one column.
//
// `Row` is the payload type — a record key here. Built from a column already
// sorted by (value, row), which is what the extractor produces anyway.
template <typename T, typename Row = std::int64_t>
class BitmapIndex {
public:
    BitmapIndex() = default;

    // `row_space`, when given, is every row the table holds rather than only
    // the rows carrying a value for this column.
    //
    // It matters because two bitmaps can be AND-ed only if position i means
    // the same row in both. A column present on every row and one present on
    // half of them would otherwise have positions that silently disagree, and
    // a conjunction over them would return rows that match neither predicate.
    // Rows with no value occupy a position and appear in no bitmap, which is
    // exactly the "absent matches nothing" rule the schema already commits to.
    void build(const std::vector<T>& values, const std::vector<Row>& rows,
               const std::vector<Row>* row_space = nullptr) {
        if (values.size() != rows.size()) {
            throw std::invalid_argument(
                "BitmapIndex::build: " + std::to_string(values.size()) +
                " values but " + std::to_string(rows.size()) + " rows");
        }
        dictionary_.clear();
        bitmaps_.clear();
        rows_.clear();
        identity_rows_ = false;

        // The row table, sorted so decoding yields ascending record keys and
        // key -> position is a binary search.
        std::vector<Row> sorted_rows = row_space ? *row_space : rows;
        std::sort(sorted_rows.begin(), sorted_rows.end());
        sorted_rows.erase(std::unique(sorted_rows.begin(), sorted_rows.end()),
                          sorted_rows.end());
        n_ = sorted_rows.size();
        identity_rows_ = is_identity(sorted_rows);
        if (!identity_rows_) rows_ = std::move(sorted_rows);

        // The dictionary. `values` arrives sorted, so distinct values are
        // adjacent and one pass finds them.
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (dictionary_.empty() || dictionary_.back() < values[i]) {
                dictionary_.push_back(values[i]);
            }
        }
        bitmaps_.assign(dictionary_.size(), Bitset(n_));

        for (std::size_t i = 0; i < values.size(); ++i) {
            bitmaps_[code_of(values[i])].set(position_of(rows[i]));
        }
        // Live means "in some bitmap". Rows in the row space with no value for
        // this column are not live and match no predicate on it.
        live_ = values.size();
    }

    std::size_t size() const { return live_; }
    std::size_t rows() const { return n_; }
    std::size_t distinct() const { return dictionary_.size(); }
    const std::vector<T>& dictionary() const { return dictionary_; }

    // --- queries ------------------------------------------------------------

    Bitset equal(const T& value) const {
        const auto it = std::lower_bound(dictionary_.begin(), dictionary_.end(), value);
        if (it == dictionary_.end() || *it != value) return Bitset(n_);
        return bitmaps_[static_cast<std::size_t>(it - dictionary_.begin())];
    }

    // Codes in [lo, hi), OR-ed. The contiguous run is what a sorted dictionary
    // buys, and it is why the dictionary is sorted rather than hashed.
    Bitset run(std::size_t lo, std::size_t hi) const {
        lo = std::min(lo, dictionary_.size());
        hi = std::min(hi, dictionary_.size());
        if (lo >= hi) return Bitset(n_);

        // When the run covers most of the dictionary it is cheaper to build
        // the complement and invert. Both are exact; this only chooses which
        // costs fewer word operations.
        const std::size_t width = hi - lo;
        if (width * 2 > dictionary_.size() && dictionary_.size() > 1) {
            Bitset out(n_);
            for (std::size_t c = 0; c < lo; ++c) out |= bitmaps_[c];
            for (std::size_t c = hi; c < dictionary_.size(); ++c) out |= bitmaps_[c];
            out.flip();
            // Rows deleted from every bitmap must not reappear through the
            // complement: they belong to no value at all.
            if (has_deletions()) out &= live_mask();
            return out;
        }

        Bitset out(n_);
        for (std::size_t c = lo; c < hi; ++c) out |= bitmaps_[c];
        return out;
    }

    std::size_t lower_code(const T& value) const {
        return static_cast<std::size_t>(
            std::lower_bound(dictionary_.begin(), dictionary_.end(), value) -
            dictionary_.begin());
    }
    std::size_t upper_code(const T& value) const {
        return static_cast<std::size_t>(
            std::upper_bound(dictionary_.begin(), dictionary_.end(), value) -
            dictionary_.begin());
    }

    // Record keys for the set positions, ascending — because the row table is
    // sorted, so decoding is already in order and needs no sort afterwards.
    std::vector<Row> decode(const Bitset& set) const {
        std::vector<Row> out;
        out.reserve(set.count());
        set.for_each([&](std::size_t position) {
            out.push_back(key_at(position));
        });
        return out;
    }

    // --- writes -------------------------------------------------------------

    // Set a row's value, clearing whatever it had. Returns false when the row
    // is not in the table and cannot be appended in place (see append()).
    bool assign(const T& value, const Row& row) {
        std::size_t position = 0;
        if (!find_position(row, &position)) return false;
        const bool was_live = clear_at(position);
        bitmaps_[ensure_code(value)].set(position);
        if (!was_live) ++live_;
        return true;
    }

    // Append a row whose key sorts after every existing one.
    //
    // The one write shape this structure takes in place. A key landing in the
    // middle would have to shift every later position by one, invalidating
    // every bitmap at once — so that case is refused here and the caller
    // rebuilds. Auto-increment ids and timestamps append, which is most of
    // what a low-cardinality column is keyed beside.
    bool append(const T& value, const Row& row) {
        if (n_ != 0 && !(key_at(n_ - 1) < row)) return false;
        if (identity_rows_ && row != static_cast<Row>(n_)) {
            // The identity shortcut only holds while keys are exactly 0..n-1.
            materialise_rows();
        }
        if (!identity_rows_) rows_.push_back(row);
        for (Bitset& b : bitmaps_) b.push_back(false);
        ++n_;
        bitmaps_[ensure_code(value)].set(n_ - 1);
        ++live_;
        return true;
    }

    // Clear a row from every bitmap. Exact, O(d) words touched, no tombstone
    // and no rebuild — the one family here where a delete costs nothing.
    bool erase(const Row& row) {
        std::size_t position = 0;
        if (!find_position(row, &position)) return false;
        if (!clear_at(position)) return false;
        --live_;
        return true;
    }

    bool contains_row(const Row& row) const {
        std::size_t position = 0;
        return find_position(row, &position) && is_live(position);
    }

    // --- accounting ---------------------------------------------------------

    std::size_t memory_bytes() const {
        std::size_t total = dictionary_.capacity() * sizeof(T);
        // Only a string dictionary owns anything beyond its slots, and the
        // `if constexpr` keeps this loop from being instantiated for
        // std::vector<bool>, whose iteration yields a proxy rather than a
        // reference.
        if constexpr (std::is_same_v<T, std::string>) {
            for (const std::string& value : dictionary_) {
                total += KeyHeapBytes<std::string>::of(value);
            }
        }
        for (const Bitset& b : bitmaps_) total += b.memory_bytes();
        total += rows_.capacity() * sizeof(Row);
        return total;
    }

    void validate() const {
        if (bitmaps_.size() != dictionary_.size()) {
            throw std::logic_error("BitmapIndex: dictionary and bitmaps differ");
        }
        for (std::size_t i = 1; i < dictionary_.size(); ++i) {
            if (!(dictionary_[i - 1] < dictionary_[i])) {
                throw std::logic_error(
                    "BitmapIndex: dictionary is not strictly ascending at " +
                    std::to_string(i));
            }
        }
        // Every row belongs to at most one value: the bitmaps partition the
        // live rows. Overlap would make a range query double-count.
        std::size_t total = 0;
        for (const Bitset& b : bitmaps_) {
            if (b.size() != n_) {
                throw std::logic_error("BitmapIndex: a bitmap is the wrong width");
            }
            total += b.count();
        }
        if (total != live_) {
            throw std::logic_error(
                "BitmapIndex: bitmaps hold " + std::to_string(total) +
                " set bits but " + std::to_string(live_) + " rows are live; a "
                "row set in two bitmaps would be counted twice by a range");
        }
        if (!identity_rows_ && rows_.size() != n_) {
            throw std::logic_error("BitmapIndex: row table is the wrong length");
        }
        for (std::size_t i = 1; i < rows_.size(); ++i) {
            if (!(rows_[i - 1] < rows_[i])) {
                throw std::logic_error(
                    "BitmapIndex: row table is not ascending at " +
                    std::to_string(i) + "; decoding would not be in key order "
                    "and key lookup would be wrong");
            }
        }
    }

private:
    static bool is_identity(const std::vector<Row>& sorted) {
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            if (sorted[i] != static_cast<Row>(i)) return false;
        }
        return true;
    }

    void materialise_rows() {
        rows_.resize(n_);
        for (std::size_t i = 0; i < n_; ++i) rows_[i] = static_cast<Row>(i);
        identity_rows_ = false;
    }

    Row key_at(std::size_t position) const {
        return identity_rows_ ? static_cast<Row>(position) : rows_[position];
    }

    bool find_position(const Row& row, std::size_t* out) const {
        if (identity_rows_) {
            if (row < 0 || static_cast<std::size_t>(row) >= n_) return false;
            *out = static_cast<std::size_t>(row);
            return true;
        }
        const auto it = std::lower_bound(rows_.begin(), rows_.end(), row);
        if (it == rows_.end() || *it != row) return false;
        *out = static_cast<std::size_t>(it - rows_.begin());
        return true;
    }

    std::size_t position_of(const Row& row) const {
        std::size_t out = 0;
        if (!find_position(row, &out)) {
            throw std::logic_error("BitmapIndex: row missing from the row table");
        }
        return out;
    }

    std::size_t code_of(const T& value) const {
        const auto it = std::lower_bound(dictionary_.begin(), dictionary_.end(), value);
        return static_cast<std::size_t>(it - dictionary_.begin());
    }

    // The code for a value, adding it to the dictionary if new.
    //
    // Inserting into the middle of the dictionary shifts later codes, which is
    // why the bitmaps move with it. That is O(d) vector moves, not O(n) bit
    // work: the bitmaps themselves are not rewritten, only reordered.
    std::size_t ensure_code(const T& value) {
        const auto it = std::lower_bound(dictionary_.begin(), dictionary_.end(), value);
        const std::size_t at = static_cast<std::size_t>(it - dictionary_.begin());
        if (it != dictionary_.end() && *it == value) return at;
        dictionary_.insert(it, value);
        bitmaps_.insert(bitmaps_.begin() + static_cast<std::ptrdiff_t>(at),
                        Bitset(n_));
        return at;
    }

    bool clear_at(std::size_t position) {
        for (Bitset& b : bitmaps_) {
            if (b.test(position)) {
                b.clear(position);
                return true;
            }
        }
        return false;
    }

    bool is_live(std::size_t position) const {
        for (const Bitset& b : bitmaps_) {
            if (b.test(position)) return true;
        }
        return false;
    }

    bool has_deletions() const { return live_ != n_; }

    // Rows belonging to some value. Only built when deletions have happened,
    // because it is the complement path's correctness guard and nothing else.
    Bitset live_mask() const {
        Bitset out(n_);
        for (const Bitset& b : bitmaps_) out |= b;
        return out;
    }

    std::vector<T> dictionary_;
    std::vector<Bitset> bitmaps_;
    std::vector<Row> rows_;      // empty when identity_rows_
    bool identity_rows_ = false;
    std::size_t n_ = 0;          // positions, including deleted rows
    std::size_t live_ = 0;       // rows currently in some bitmap
};

}  // namespace hylis::index
