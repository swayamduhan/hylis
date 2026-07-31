// storage/record.hpp
//
// The Record type: the unit of data stored by RecordStore.
//
// A record is intentionally schema-light — a primary key plus an arbitrary
// map of column name -> value (values are strings for now; rich types are an
// index-layer concern, not a storage-layer concern). The B+ tree indexes the
// primary key; vector columns later get indexed by HNSW / the neural router.
//
// Records are value types (copyable, comparable by key). We do NOT make them
// immutable in C++ the way we might in Python, but the storage API treats
// updates as delete+insert at the logical level so the WAL story stays clean.

#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace hylis::storage {

struct Record {
    // Primary key. int64_t so it can also address large datasets without
    // wrapping, and it serves directly as the comparand in the B+ tree.
    std::int64_t key = 0;

    // Arbitrary payload. std::map (not unordered_map) so column iteration is
    // deterministic — handy for tests, dumps, and checkpoint serialization.
    std::map<std::string, std::string> columns;

    Record() = default;
    explicit Record(std::int64_t k) : key(k) {}
    Record(std::int64_t k, std::map<std::string, std::string> cols)
        : key(k), columns(std::move(cols)) {}

    // Ordered by key. Lets a Record be used directly in sorted containers /
    // range algorithms, which several tests rely on.
    bool operator<(const Record& other) const { return key < other.key; }
    bool operator==(const Record& other) const { return key == other.key; }

    std::string get(const std::string& name, const std::string& default_value = "") const {
        auto it = columns.find(name);
        return it == columns.end() ? default_value : it->second;
    }
};

} // namespace hylis::storage
