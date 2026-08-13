// query/predicate.hpp
//
// One structured constraint, over a typed column.
//
// Why this is wider than CompareOp
// --------------------------------
// CompareOp is the vocabulary every *ordered index* speaks, and it stops at
// the five comparisons because that is what a B+ tree and an RMI both
// implement. A table has to answer more than an index can:
//
//   Between   one leaf-chain walk rather than two predicates intersected
//   Prefix    the string operator, and the reason string columns are indexed
//             as strings -- `title LIKE 'nike%'` is the half-open range
//             [p, succ(p)), which no integer encoding of the string can express
//   Contains  an infix match, which **no** index here can serve. Supported
//             anyway, executed by a full scan, and labelled as such: silently
//             refusing it and silently scanning are both worse than saying so
//   IsNull    rows absent from the column, which by construction appear in no
//             index at all
//
// So PredOp is the table's vocabulary and CompareOp stays the index's. The
// mapping between them is one switch, in table.hpp, and the two operators with
// no index to push down to are the ones that make the distinction worth
// drawing rather than collapsing.

#pragma once

#include <stdexcept>
#include <string>

#include "index/compare_op.hpp"
#include "index/logical_type.hpp"

namespace hylis::query {

using hylis::index::CompareOp;
using hylis::index::Datum;

enum class PredOp {
    Eq,        // == value
    Lt,        // <  value
    Le,        // <= value
    Gt,        // >  value
    Ge,        // >= value
    Between,   // value <= x <= value2, inclusive both ends
    Prefix,    // string columns only
    Contains,  // string columns only; full scan, never an index
    IsNull,    // the column is absent from the row
};

inline const char* to_string(PredOp op) {
    switch (op) {
        case PredOp::Eq: return "==";
        case PredOp::Lt: return "<";
        case PredOp::Le: return "<=";
        case PredOp::Gt: return ">";
        case PredOp::Ge: return ">=";
        case PredOp::Between: return "between";
        case PredOp::Prefix: return "prefix";
        case PredOp::Contains: return "contains";
        case PredOp::IsNull: return "is null";
    }
    return "?";
}

// Whether an ordered index can answer this operator directly.
//
// The two that cannot are the honest boundary of this design: an infix match
// has no ordering to exploit, and an absent value is absent from the index by
// construction. Both are still answered, by scanning, and the plan says so.
inline bool op_is_indexable(PredOp op) {
    return op != PredOp::Contains && op != PredOp::IsNull;
}

inline bool op_needs_second_value(PredOp op) { return op == PredOp::Between; }

inline bool op_is_string_only(PredOp op) {
    return op == PredOp::Prefix || op == PredOp::Contains;
}

// The index-level operator a table-level one becomes, for the five that map
// straight across. Throws for the rest, which have their own execution paths.
inline CompareOp compare_op_of(PredOp op) {
    switch (op) {
        case PredOp::Eq: return CompareOp::Eq;
        case PredOp::Lt: return CompareOp::Lt;
        case PredOp::Le: return CompareOp::Le;
        case PredOp::Gt: return CompareOp::Gt;
        case PredOp::Ge: return CompareOp::Ge;
        default: break;
    }
    throw std::invalid_argument(
        std::string("compare_op_of: '") + to_string(op) +
        "' is not one of the five index comparisons; it has its own path");
}

struct Predicate {
    std::string column;
    PredOp op = PredOp::Eq;
    Datum value;
    Datum value2;  // Between only

    Predicate() = default;
    Predicate(std::string c, PredOp o, Datum v)
        : column(std::move(c)), op(o), value(std::move(v)) {}
    Predicate(std::string c, PredOp o, Datum v, Datum v2)
        : column(std::move(c)), op(o), value(std::move(v)), value2(std::move(v2)) {}

    // How the predicate would read in the query it came from. For an
    // explanation string or a REPL prompt, not for evaluation.
    std::string describe(hylis::index::LogicalType type) const {
        if (op == PredOp::IsNull) return column + " is null";
        const std::string rendered = index::format_datum(type, value);
        if (op == PredOp::Between) {
            return column + " between " + rendered + " and " +
                   index::format_datum(type, value2);
        }
        if (op == PredOp::Prefix) return column + " like '" + rendered + "%'";
        if (op == PredOp::Contains) return column + " like '%" + rendered + "%'";
        return column + " " + to_string(op) + " " + rendered;
    }
};

}  // namespace hylis::query
