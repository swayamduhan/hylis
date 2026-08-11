// index/compare_op.hpp
//
// The predicate vocabulary every ordered index in hylis speaks.
//
// This lives in its own header because more than one structure implements it.
// The query planner picks an index per column when the data is loaded and
// then issues predicates without knowing what answered them — a B+ tree, a
// learned index, or whatever replaces them later. Keeping the enum separate
// means the learned index does not have to include the whole B+ tree just to
// name an operator.

#pragma once

namespace hylis::index {

enum class CompareOp {
    Eq,  // == value
    Lt,  // <  value
    Le,  // <= value
    Gt,  // >  value
    Ge,  // >= value
};

// The operator as it would appear in the query it came from. Used where a
// predicate has to be shown to a person — a planner's explanation, a REPL
// prompt — rather than evaluated.
inline const char* symbol_of(CompareOp op) {
    switch (op) {
        case CompareOp::Eq: return "==";
        case CompareOp::Lt: return "<";
        case CompareOp::Le: return "<=";
        case CompareOp::Gt: return ">";
        case CompareOp::Ge: return ">=";
    }
    return "?";
}

}  // namespace hylis::index
