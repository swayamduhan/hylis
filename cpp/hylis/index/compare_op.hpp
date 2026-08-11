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

}  // namespace hylis::index
