// storage/store.hpp
//
// The record store: in-memory records made durable via the WAL.
//
// Why in-memory + WAL instead of a page store? This project's contribution
// is the *indexing* layer, not storage. An in-memory hash map with a WAL
// gives real durability semantics (crash recovery) with a tiny amount of
// code, so the complexity budget can go to the B+ tree, learned index, and
// neural router instead. Known tradeoff: fsync-per-write caps throughput;
// group-commit/batched fsync would be a reasonable stretch feature.

#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "storage/json_detail.hpp"
#include "storage/record.hpp"
#include "storage/wal.hpp"

namespace hylis::storage {

namespace fs = std::filesystem;

class RecordStore {
public:
    // Files used under `directory`:
    //   wal.log          — the write-ahead log
    //   checkpoint.json  — full state snapshot written on checkpoint()
    static constexpr const char* WAL_NAME        = "wal.log";
    static constexpr const char* CHECKPOINT_NAME = "checkpoint.json";

    // `recover=true` replays the WAL on open. Tests may pass false for an
    // ephemeral store that starts empty even if a stale WAL is present.
    explicit RecordStore(const fs::path& directory, bool recover = true)
        : directory_(directory) {
        fs::create_directories(directory_);
        wal_ = std::make_unique<WriteAheadLog>((directory_ / WAL_NAME).string());
        if (recover) recover_state();
    }

    ~RecordStore() = default;
    RecordStore(const RecordStore&) = delete;
    RecordStore& operator=(const RecordStore&) = delete;

    // Insert or update. WAL first (append + fsync), then memory: if we crash
    // after the WAL append but before the map update, recovery still applies
    // the put, so disk and memory converge.
    void put(const Record& r) {
        wal_->append_put(r.key, r.columns);
        records_[r.key] = r;
    }

    // Delete by key. Returns true iff a record was removed. Deletes of
    // already-absent keys aren't logged, to keep the log small.
    bool del(std::int64_t key) {
        auto it = records_.find(key);
        if (it == records_.end()) return false;
        wal_->append_delete(key);
        records_.erase(it);
        return true;
    }

    const Record* get(std::int64_t key) const {
        auto it = records_.find(key);
        return it == records_.end() ? nullptr : &it->second;
    }

    bool contains(std::int64_t key) const { return records_.count(key) != 0; }
    std::size_t size() const { return records_.size(); }

    // All live records in unspecified (hash) order. Sorted iteration is the
    // B+ tree's job; the store stays a thin key/value layer.
    std::vector<Record> records() const {
        std::vector<Record> out;
        out.reserve(records_.size());
        for (const auto& [k, v] : records_) out.push_back(v);
        return out;
    }

    std::vector<std::int64_t> keys() const {
        std::vector<std::int64_t> out;
        out.reserve(records_.size());
        for (const auto& [k, v] : records_) out.push_back(k);
        return out;
    }

    // Snapshots state to disk and truncates the WAL:
    //   1. Write the snapshot to a temp file in the same directory, fsync,
    //      then atomically rename over the previous checkpoint (a crash
    //      mid-write leaves the previous checkpoint intact).
    //   2. Truncate the WAL, then log a checkpoint marker. If we crash
    //      between these two steps the worst case is an empty log plus a
    //      valid checkpoint file, which recovery still handles correctly.
    void checkpoint() {
        const fs::path ckpt_path = directory_ / CHECKPOINT_NAME;
        const fs::path tmp_path = directory_ / ".ckpt_tmp.json";

        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("checkpoint: cannot open temp");
            const std::string blob = serialize_snapshot();
            out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
            out.flush();
        }
        fsync_path(tmp_path);
        fs::rename(tmp_path, ckpt_path); // atomic replace of any prior checkpoint
        fsync_path(ckpt_path);

        wal_->truncate();
        wal_->append_checkpoint();
    }

    void close() { if (wal_) wal_->close(); }

    const fs::path& directory() const { return directory_; }

    // Test-only accessor so tests can assert the LSN counter survives a
    // reopen. Not part of the public contract.
    std::int64_t wal_next_lsn_for_test() const { return wal_ ? wal_->next_lsn() : -1; }

private:
    using RecordMap = std::unordered_map<std::int64_t, Record>;

    fs::path directory_;
    std::unique_ptr<WriteAheadLog> wal_;
    RecordMap records_;

    void recover_state() {
        const fs::path ckpt = directory_ / CHECKPOINT_NAME;
        if (fs::exists(ckpt)) {
            std::ifstream in(ckpt, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
            parse_snapshot(s);
        }
        // Replay the WAL on top; put/delete are idempotent under replay.
        for (const auto& e : wal_->iter_entries()) apply(e);
    }

    void apply(const LogEntry& e) {
        if (e.op == OP_PUT) {
            Record r;
            r.key = e.key.value_or(-1);
            if (e.payload) r.columns = *e.payload;
            records_[r.key] = std::move(r);
        } else if (e.op == OP_DELETE) {
            if (e.key) records_.erase(*e.key);
        } else if (e.op == OP_CHECKPOINT) {
            // marker only
        } else {
            throw std::runtime_error("RecordStore: unknown WAL op '" + e.op + "'");
        }
    }

    // Snapshot grammar: {"records":[[key,{"col":"val",...}], ...]}, keys
    // sorted for deterministic, diff-friendly output. Reuses the WAL's JSON
    // primitives rather than a second hand-rolled parser.
    std::string serialize_snapshot() const {
        std::vector<std::int64_t> sorted_keys;
        sorted_keys.reserve(records_.size());
        for (const auto& [k, v] : records_) sorted_keys.push_back(k);
        std::sort(sorted_keys.begin(), sorted_keys.end());

        std::string out = "{\"records\":[";
        bool first = true;
        for (auto k : sorted_keys) {
            const auto& r = records_.at(k);
            if (!first) out += ',';
            first = false;
            out += '[';
            out += std::to_string(k);
            out += ",{";
            bool fc = true;
            for (const auto& [ck, cv] : r.columns) {
                if (!fc) out += ',';
                fc = false;
                out += '"';
                out += json_detail::escape_string(ck);
                out += "\":\"";
                out += json_detail::escape_string(cv);
                out += '"';
            }
            out += "}]";
        }
        out += "]}";
        return out;
    }

    void parse_snapshot(const std::string& s) {
        using namespace json_detail;
        const char* p = s.c_str();

        skip_ws(p);
        expect(p, '{'); skip_ws(p);
        if (read_string(p) != "records") throw std::runtime_error("snapshot: missing records");
        skip_ws(p); expect(p, ':'); skip_ws(p); expect(p, '[');
        if (*p == ']') { ++p; return; }

        while (true) {
            skip_ws(p); expect(p, '[');
            const std::int64_t key = read_int(p);
            skip_ws(p); expect(p, ','); skip_ws(p);
            std::map<std::string, std::string> cols = read_string_object(p);
            skip_ws(p); expect(p, ']');

            Record r; r.key = key; r.columns = std::move(cols);
            records_[r.key] = std::move(r);

            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == ']') { ++p; break; }
            throw std::runtime_error("snapshot: expected , or ]");
        }
    }

    // fsync a path just written via ofstream.
    //
    // On POSIX, opening "rb" is fine — fsync() works on any fd with a
    // backing file regardless of open mode. On Windows, FlushFileBuffers
    // requires a handle with write access, so we open "r+b" (read+write, no
    // truncation) instead. Directory fsync is best-effort: not every
    // filesystem supports fsyncing a directory handle.
    static void fsync_path(const fs::path& p) {
#ifdef _WIN32
        std::FILE* f = std::fopen(p.string().c_str(), "r+b");
#else
        std::FILE* f = std::fopen(p.string().c_str(), "rb");
#endif
        if (f) { detail::fsync_file(f); std::fclose(f); }

        const auto dir = p.parent_path().string();
#ifdef _WIN32
        std::FILE* d = std::fopen(dir.c_str(), "r+b");
#else
        std::FILE* d = std::fopen(dir.c_str(), "rb");
#endif
        if (d) { try { detail::fsync_file(d); } catch (...) {} std::fclose(d); }
    }
};

} // namespace hylis::storage
