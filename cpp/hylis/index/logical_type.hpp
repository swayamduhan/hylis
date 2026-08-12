// index/logical_type.hpp
//
// What a column *is*, as opposed to how it happens to be stored.
//
// Why this header exists
// ----------------------
// Every index in this project was written against `std::int64_t` keys, and
// column_index.hpp says why (lines 31-33): it kept IndexPlan a plain
// serialisable struct and int64 was what the planner and bindings used. That
// was right while every column was a synthetic key array. It stops being right
// the moment records are real, because a real table has prices, categories,
// titles, flags and timestamps, and those want four different structures.
//
// The observation that forces the design is that the two ordered structures
// are not equally general:
//
//   * BPlusTree touches keys only through std::lower_bound / std::upper_bound
//     and ==. No arithmetic anywhere. BPlusTree<std::string, int64_t> compiles
//     and works today; nothing but ColumnIndex was stopping it.
//   * RMIndex does static_cast<double>(key) to fit its models. A learned index
//     approximates a CDF, and a CDF needs a metric on the key space. There is
//     no repair for this — a model fitted to string *ranks* would be fitted to
//     an ordering the model itself imposed, which measures nothing.
//
// So the logical type is a hard filter on which index families are candidates
// at all, and measurement decides among the survivors. Type eliminates, cost
// tiebreaks. That is the whole idea, and this header is the vocabulary for it.
//
// Naming
// ------
// `Datum` for a single typed value, following Postgres. `Value` was taken —
// every index template in this project already uses it for the payload type,
// and ColumnValue means "row id". A third meaning would have been one too many.

#pragma once

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "index/key_bytes.hpp"

namespace hylis::index {

// ---------------------------------------------------------------------------
// Logical types
// ---------------------------------------------------------------------------

enum class LogicalType {
    Int64,      // integers, and the right type for money (cents)
    Double,     // genuinely continuous measures
    String,     // indexed natively by the B+ tree; never encoded to int64
    Bool,       // two distinct values; an ordered index over it is pointless
    Timestamp,  // physically Int64 (epoch ms), logically distinct
    Vector,     // not a scalar index at all; HNSW / routed HNSW / flat
};

inline const char* to_string(LogicalType type) {
    switch (type) {
        case LogicalType::Int64: return "int64";
        case LogicalType::Double: return "double";
        case LogicalType::String: return "string";
        case LogicalType::Bool: return "bool";
        case LogicalType::Timestamp: return "timestamp";
        case LogicalType::Vector: return "vector";
    }
    return "int64";
}

inline LogicalType logical_type_from_string(const std::string& text) {
    if (text == "int64") return LogicalType::Int64;
    if (text == "double") return LogicalType::Double;
    if (text == "string") return LogicalType::String;
    if (text == "bool") return LogicalType::Bool;
    if (text == "timestamp") return LogicalType::Timestamp;
    if (text == "vector") return LogicalType::Vector;
    throw std::invalid_argument("unknown logical type: " + text);
}

// Whether the type has a total order worth indexing with an ordered structure.
// Bool is technically ordered and excluded anyway: with two distinct keys a
// B+ tree degenerates to a sorted list of every row in the table, providing no
// ordering benefit whatever. It gets a bitmap instead.
inline bool type_is_ordered(LogicalType type) {
    return type != LogicalType::Vector && type != LogicalType::Bool;
}

// Whether a key of this type can be cast to double without losing the ordering
// a learned index needs. This is the RMI eligibility test, and the reason
// String has no RMI option anywhere in this codebase.
inline bool type_supports_rmi(LogicalType type) {
    return type == LogicalType::Int64 || type == LogicalType::Double ||
           type == LogicalType::Timestamp;
}

// ---------------------------------------------------------------------------
// How a column's values are turned into index keys
// ---------------------------------------------------------------------------
//
// Native      the value itself. Requires the values be unique, because every
//             ordered structure here maps one key to one row.
// Composite   pair<T, record_key>. Makes duplicated values unique by pairing
//             them with the row they came from — the textbook secondary-index
//             representation, and it needs no change to btree.hpp because
//             std::pair already compares lexicographically-then-tiebreak.
// Position    sorted rank. The only encoding available to an RMI over a
//             duplicated numeric column, since a pair cannot be cast to
//             double. Confined to exactly that case.
// Dictionary  a dense code per distinct value, with one bitmap each.
enum class KeyEncoding { Native, Composite, Position, Dictionary };

inline const char* to_string(KeyEncoding encoding) {
    switch (encoding) {
        case KeyEncoding::Native: return "native";
        case KeyEncoding::Composite: return "composite";
        case KeyEncoding::Position: return "position";
        case KeyEncoding::Dictionary: return "dictionary";
    }
    return "native";
}

inline KeyEncoding key_encoding_from_string(const std::string& text) {
    if (text == "native") return KeyEncoding::Native;
    if (text == "composite") return KeyEncoding::Composite;
    if (text == "position") return KeyEncoding::Position;
    if (text == "dictionary") return KeyEncoding::Dictionary;
    throw std::invalid_argument("unknown key encoding: " + text);
}

// ---------------------------------------------------------------------------
// Datum: one typed column value
// ---------------------------------------------------------------------------
//
// Four alternatives, not six: Timestamp is carried as int64 and Vector never
// appears as a scalar predicate operand. The logical type lives in the schema,
// which is what tells a bare int64 whether it is a count or an instant.
using Datum = std::variant<std::int64_t, double, std::string, bool>;

// The alternative index a value of this logical type occupies. Int64 and
// Timestamp share one, which is exactly why the schema has to be consulted
// rather than the variant interrogated.
inline std::size_t datum_slot(LogicalType type) {
    switch (type) {
        case LogicalType::Int64: return 0;
        case LogicalType::Timestamp: return 0;
        case LogicalType::Double: return 1;
        case LogicalType::String: return 2;
        case LogicalType::Bool: return 3;
        case LogicalType::Vector: break;
    }
    throw std::invalid_argument(
        "Datum cannot hold a vector: vector columns are served by the graph "
        "indexes and never appear as a scalar predicate operand");
}

inline bool datum_matches(const Datum& d, LogicalType type) {
    return type != LogicalType::Vector && d.index() == datum_slot(type);
}

// Comparison, defined only between the same alternative.
//
// Deliberately throwing rather than falling back to variant's own operator<,
// which compares by alternative index first and would silently rank every
// integer below every string. A type mismatch here is a schema bug at the call
// site, and returning a plausible-looking wrong answer is the worst outcome.
inline void require_same(const Datum& a, const Datum& b, const char* what) {
    if (a.index() != b.index()) {
        throw std::invalid_argument(
            std::string(what) + ": cannot compare a value of one type with a "
            "value of another; the predicate does not match the column type");
    }
}

inline bool datum_less(const Datum& a, const Datum& b) {
    require_same(a, b, "datum_less");
    switch (a.index()) {
        case 0: return std::get<std::int64_t>(a) < std::get<std::int64_t>(b);
        case 1: return std::get<double>(a) < std::get<double>(b);
        case 2: return std::get<std::string>(a) < std::get<std::string>(b);
        default: return std::get<bool>(a) < std::get<bool>(b);
    }
}

inline bool datum_equal(const Datum& a, const Datum& b) {
    require_same(a, b, "datum_equal");
    switch (a.index()) {
        case 0: return std::get<std::int64_t>(a) == std::get<std::int64_t>(b);
        case 1: return std::get<double>(a) == std::get<double>(b);
        case 2: return std::get<std::string>(a) == std::get<std::string>(b);
        default: return std::get<bool>(a) == std::get<bool>(b);
    }
}

// Bytes one Datum occupies, for reporting. See index/key_bytes.hpp for the
// trait the indexes themselves use.
inline std::size_t datum_bytes(const Datum& d) {
    return sizeof(Datum) +
           (d.index() == 2 ? KeyHeapBytes<std::string>::of(std::get<std::string>(d))
                           : 0);
}

// ---------------------------------------------------------------------------
// Civil dates, for Timestamp
// ---------------------------------------------------------------------------
//
// Howard Hinnant's days_from_civil / civil_from_days. Written out rather than
// pulled from <chrono>'s C++20 calendar because this project targets C++17,
// and rather than from a dependency because it is fifteen lines and this
// project has no core-logic dependencies by design.

namespace time_detail {

inline std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

struct Civil {
    std::int64_t year;
    unsigned month;
    unsigned day;
};

inline Civil civil_from_days(std::int64_t z) {
    z += 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const std::int64_t y = static_cast<std::int64_t>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp + (mp < 10 ? 3 : -9);
    return Civil{y + (m <= 2), m, d};
}

inline void two_digits(std::string& out, unsigned v) {
    out += static_cast<char>('0' + (v / 10) % 10);
    out += static_cast<char>('0' + v % 10);
}

}  // namespace time_detail

// ---------------------------------------------------------------------------
// Parsing and formatting
// ---------------------------------------------------------------------------

// Whether `text` is entirely consumed by a numeric parse. Trailing junk is a
// parse failure, not a value: "12abc" is a typo, and accepting it as 12 turns
// a data-entry mistake into a silently wrong index.
namespace parse_detail {

inline bool parse_int64(const std::string& text, std::int64_t* out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') return false;
    *out = static_cast<std::int64_t>(v);
    return true;
}

inline bool parse_double(const std::string& text, double* out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') return false;
    // NaN has no ordering, and infinities defeat the mean-centred fit every
    // linear model here depends on. Both are rejected at the boundary so no
    // structure ever has to cope with one.
    if (std::isnan(v) || std::isinf(v)) return false;
    *out = v;
    return true;
}

inline bool parse_bool(const std::string& text, bool* out) {
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    if (lower == "true" || lower == "1") { *out = true; return true; }
    if (lower == "false" || lower == "0") { *out = false; return true; }
    return false;
}

// ISO-8601, or a bare integer already in epoch milliseconds.
//
// Accepted: "2026-08-13", "2026-08-13T14:30:00", "2026-08-13T14:30:00Z",
// "2026-08-13 14:30:00", and any of those with fractional seconds. Offsets
// other than Z are rejected rather than silently treated as UTC — a timestamp
// that is wrong by hours is worse than one that failed to load.
inline bool parse_timestamp(const std::string& text, std::int64_t* out) {
    if (parse_int64(text, out)) return true;
    if (text.size() < 10) return false;

    auto digits = [&](std::size_t at, std::size_t n, std::int64_t* v) {
        if (at + n > text.size()) return false;
        std::int64_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const char c = text[at + i];
            if (c < '0' || c > '9') return false;
            acc = acc * 10 + (c - '0');
        }
        *v = acc;
        return true;
    };

    std::int64_t y = 0, mo = 0, d = 0;
    if (!digits(0, 4, &y) || text[4] != '-' || !digits(5, 2, &mo) ||
        text[7] != '-' || !digits(8, 2, &d)) {
        return false;
    }
    if (mo < 1 || mo > 12 || d < 1 || d > 31) return false;

    std::int64_t h = 0, mi = 0, s = 0, ms = 0;
    std::size_t at = 10;
    if (at < text.size()) {
        if (text[at] != 'T' && text[at] != ' ') return false;
        ++at;
        if (!digits(at, 2, &h) || text[at + 2] != ':' ||
            !digits(at + 3, 2, &mi)) {
            return false;
        }
        at += 5;
        if (at < text.size() && text[at] == ':') {
            if (!digits(at + 1, 2, &s)) return false;
            at += 3;
            if (at < text.size() && text[at] == '.') {
                ++at;
                std::int64_t scale = 100;
                while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
                    ms += (text[at] - '0') * scale;
                    scale /= 10;
                    ++at;
                }
            }
        }
        if (at < text.size()) {
            if (text[at] != 'Z' || at + 1 != text.size()) return false;
        }
        if (h > 23 || mi > 59 || s > 60) return false;
    }

    const std::int64_t days = time_detail::days_from_civil(
        y, static_cast<unsigned>(mo), static_cast<unsigned>(d));
    *out = ((days * 24 + h) * 60 + mi) * 60000 + s * 1000 + ms;
    return true;
}

}  // namespace parse_detail

// Parse text into a Datum of the given logical type. Returns false rather than
// throwing, so a caller extracting a whole column can count rejects instead of
// aborting on the first one — but Schema::validate turns a reject into an
// error at ingest, which is the difference between a schema and a convention.
inline bool try_parse_datum(LogicalType type, const std::string& text,
                            Datum* out) {
    switch (type) {
        case LogicalType::Int64: {
            std::int64_t v = 0;
            if (!parse_detail::parse_int64(text, &v)) return false;
            *out = v;
            return true;
        }
        case LogicalType::Timestamp: {
            std::int64_t v = 0;
            if (!parse_detail::parse_timestamp(text, &v)) return false;
            *out = v;
            return true;
        }
        case LogicalType::Double: {
            double v = 0.0;
            if (!parse_detail::parse_double(text, &v)) return false;
            *out = v;
            return true;
        }
        case LogicalType::Bool: {
            bool v = false;
            if (!parse_detail::parse_bool(text, &v)) return false;
            *out = v;
            return true;
        }
        case LogicalType::String:
            *out = text;
            return true;
        case LogicalType::Vector:
            return false;
    }
    return false;
}

inline Datum parse_datum(LogicalType type, const std::string& text) {
    Datum out;
    if (!try_parse_datum(type, text, &out)) {
        throw std::invalid_argument("cannot parse '" + text + "' as " +
                                    to_string(type));
    }
    return out;
}

// Render a Datum back to the text form the record store holds.
//
// Round-trips: format_datum(parse_datum(t, s)) parses back to the same Datum
// for every t and every s this accepts. Timestamps normalise to ISO-8601 with
// a Z, so the stored form is canonical even when the input was epoch millis —
// which is what makes a checkpoint diffable.
inline std::string format_datum(LogicalType type, const Datum& d) {
    switch (type) {
        case LogicalType::Int64:
            return std::to_string(std::get<std::int64_t>(d));
        case LogicalType::Double: {
            // 17 significant digits round-trips every double exactly; the
            // default six would quietly lose the low bits of a price or a
            // coordinate.
            char buffer[40];
            std::snprintf(buffer, sizeof(buffer), "%.17g", std::get<double>(d));
            return std::string(buffer);
        }
        case LogicalType::String:
            return std::get<std::string>(d);
        case LogicalType::Bool:
            return std::get<bool>(d) ? "true" : "false";
        case LogicalType::Timestamp: {
            std::int64_t ms = std::get<std::int64_t>(d);
            std::int64_t day = ms / 86400000;
            std::int64_t rem = ms % 86400000;
            if (rem < 0) { rem += 86400000; --day; }
            const auto civil = time_detail::civil_from_days(day);

            std::string out = std::to_string(civil.year);
            while (out.size() < 4) out.insert(out.begin(), '0');
            out += '-';
            time_detail::two_digits(out, civil.month);
            out += '-';
            time_detail::two_digits(out, civil.day);
            out += 'T';
            time_detail::two_digits(out, static_cast<unsigned>(rem / 3600000));
            out += ':';
            time_detail::two_digits(out, static_cast<unsigned>(rem / 60000 % 60));
            out += ':';
            time_detail::two_digits(out, static_cast<unsigned>(rem / 1000 % 60));
            const std::int64_t millis = rem % 1000;
            if (millis != 0) {
                out += '.';
                out += static_cast<char>('0' + millis / 100);
                time_detail::two_digits(out, static_cast<unsigned>(millis % 100));
            }
            out += 'Z';
            return out;
        }
        case LogicalType::Vector:
            break;
    }
    throw std::invalid_argument("format_datum: vector columns have no text form");
}

// ---------------------------------------------------------------------------
// Prefix ranges
// ---------------------------------------------------------------------------

// The exclusive upper bound of the prefix range for `p`, or false when the
// prefix has no upper bound because every trailing byte is 0xFF.
//
// This is what makes `title LIKE 'nike%'` a single leaf-chain walk on a B+
// tree: the answer is exactly [p, prefix_upper_bound(p)). It is impossible
// under any int64 encoding of the string, which makes it the clearest single
// argument for indexing strings natively rather than by sorted rank.
inline bool prefix_upper_bound(const std::string& p, std::string* out) {
    *out = p;
    while (!out->empty()) {
        auto& last = out->back();
        if (static_cast<unsigned char>(last) != 0xFFu) {
            last = static_cast<char>(static_cast<unsigned char>(last) + 1);
            return true;
        }
        out->pop_back();
    }
    return false;  // the empty prefix, or all 0xFF: unbounded above
}

}  // namespace hylis::index
