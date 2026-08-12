// index/key_bytes.hpp
//
// How many bytes a key owns beyond its own object.
//
// Why this is not just sizeof
// ---------------------------
// BPlusTree::memory_bytes charges `capacity() * sizeof(Key)` per node, which
// is right for every key this project had until now. It is wrong for
// std::string, where sizeof is the 32-byte object header and the characters
// live on the heap. A string column would report a fraction of its real size.
//
// That number is not cosmetic. It feeds IndexPlan::index_bytes, choose_index's
// size budget, and the bitmap-versus-tree memory comparison — where
// undercounting the tree would make the tree look like the right answer for
// exactly the categorical columns a bitmap exists to serve.
//
// The trait reports only the *heap* bytes, so the existing
// `capacity() * sizeof(Key)` term stays and the numbers already measured for
// int64 columns are unchanged to the byte. Adding a term is safe in a way that
// replacing one is not.
//
// It lives in its own header, rather than in logical_type.hpp, so btree.hpp
// can use it without pulling in <variant> and the whole type vocabulary. The
// B+ tree should not have to know what a logical type is.

#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace hylis::index {

// Zero for every trivially-stored key: an int64 or a double owns nothing
// beyond itself, and the node already counted the slot it sits in.
template <typename Key>
struct KeyHeapBytes {
    static std::size_t of(const Key&) { return 0; }
};

template <>
struct KeyHeapBytes<std::string> {
    static std::size_t of(const std::string& s) {
        // Only when there is an allocation. Short strings live inside the
        // object on every standard library this builds against, and charging
        // for both the object and a phantom buffer would overcount precisely
        // the case that is cheapest.
        return s.capacity() > sizeof(std::string) ? s.capacity() : 0;
    }
};

// A composite secondary-index key is (value, row id). Only the value half can
// own heap memory; the row id is an integer sitting in the pair.
template <typename A, typename B>
struct KeyHeapBytes<std::pair<A, B>> {
    static std::size_t of(const std::pair<A, B>& p) {
        return KeyHeapBytes<A>::of(p.first) + KeyHeapBytes<B>::of(p.second);
    }
};

}  // namespace hylis::index
