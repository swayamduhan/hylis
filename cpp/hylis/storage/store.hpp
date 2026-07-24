// storage/store.hpp
//
// The record store: in-memory records made durable via the WAL.
//
// Responsibilities
// ----------------
// * Hold the authoritative in-memory state: key -> Record.
// * Route every mutation through the WAL **first** (write-ahead rule).
// * Recover state from the WAL on open (replay).
// * Periodically *checkpoint*: snapshot state to a file, then truncate the WAL
//   so recovery is fast.
//
// Why in-memory + WAL instead of a page store?
// --------------------------------------------
// This project's contribution is the *indexing* layer, not storage. An
// in-memory hash map with a WAL gives real durability semantics (crash
// recovery) with a tiny amount of code, so we can spend our complexity budget
// on the B+ tree, learned index, and neural router. The WAL is the honest,
// defensible way to add durability to an in-memory store.

#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    // Accept fs::path directly (tests create dirs via fs, and -Wpedantic
    // disables the implicit path→string conversion). Store as path internally.
    explicit RecordStore(const fs::path& directory, bool recover = true)
        : directory_(directory) {
        fs::create_directories(directory_);
        wal_ = std::make_unique<WriteAheadLog>((directory_ / WAL_NAME).string());
        if (recover) recover_state();
    }

    ~RecordStore() = default;
    RecordStore(const RecordStore&) = delete;
    RecordStore& operator=(const RecordStore&) = delete;

    // Insert or update. Ordering: WAL first (append + fsync), then memory.
    // If we crashed after the WAL append but before the map update, recovery
    // would still apply the put, so disk and memory converge.
    void put(const Record& r) {
        wal_->append_put(r.key, r.columns);
        records_[r.key] = r;
    }

    // Delete by key. Returns true iff a record was removed. We log deletes
    // only for keys that existed, to keep the log small.
    bool del(std::int64_t key) {
        auto it = records_.find(key);
        if (it == records_.end()) return false;
        wal_->append_delete(key);
        records_.erase(it);
        return true;
    }

    // Point lookup. O(1) hash-map access.
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

    // Snapshot state to disk and truncate the WAL.
    //
    // 1. Write the snapshot to a temp file in the same dir, then std::filesystem::rename
    //    (atomic on POSIX; on Windows it's atomic-replace when the target exists).
    //    A crash mid-snapshot leaves the previous checkpoint intact.
    // 2. fsync the snapshot.
    // 3. Append a checkpoint marker, then truncate the WAL.
    //
    // There's a small window between writing the checkpoint and truncating the
    // WAL where both contain overlapping data; replay is idempotent, so safe.
    void checkpoint() {
        const fs::path ckpt_path = directory_ / CHECKPOINT_NAME;

        // Serialize: {"records":[[key,{"col":"v",...}], ...]}.
        std::string blob = serialize_snapshot();

        // Temp file in the SAME directory so rename stays intra-filesystem.
        const fs::path tmp_path = directory_ / ".ckpt_tmp.json";
        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("checkpoint: cannot open temp");
            out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
            out.flush();
            out.close();
        }
        fsync_path(tmp_path);
        fs::rename(tmp_path, ckpt_path);   // atomic replace of any prior checkpoint
        fsync_path(ckpt_path);

        // Truncate + marker. We truncate first so the marker (written second)
        // survives in the fresh log. If a crash happens between them, the worst
        // case is an empty log with a valid checkpoint file on disk, which
        // recovery handles correctly (it loads the checkpoint, then replays
        // nothing, and ends up in the right state).
        wal_->truncate();
        wal_->append_checkpoint();
    }

    void close() { if (wal_) wal_->close(); }

    const fs::path& directory() const { return directory_; }

    // Test-only accessor exposing the WAL's next LSN, so tests can assert the
    // counter survives a reopen. Not part of the public contract.
    std::int64_t wal_next_lsn_for_test() const { return wal_ ? wal_->next_lsn() : -1; }

private:
    using RecordMap = std::unordered_map<std::int64_t, Record>;

    fs::path directory_;
    std::unique_ptr<WriteAheadLog> wal_;
    RecordMap records_;

    // ---- recovery -------------------------------------------------------
    void recover_state() {
        // 1) Load checkpoint if present.
        const fs::path ckpt = directory_ / CHECKPOINT_NAME;
        if (fs::exists(ckpt)) {
            std::ifstream in(ckpt, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
            parse_snapshot(s);   // fills records_
        }
        // 2) Replay WAL on top. Operations are idempotent under replay.
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

    // ---- snapshot serialization (same JSON dialect as the WAL) ----------
    static void esc(std::string& out, const std::string& s) {
        // Reuse WAL escaping logic by constructing a LogEntry line fragment.
        // Simplest: inline the same minimal escaper (kept in sync with wal.hpp).
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
    }

    std::string serialize_snapshot() const {
        // Emit keys in sorted order for deterministic, diff-friendly snapshots.
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
                esc(out, ck);
                out += "\":\"";
                esc(out, cv);
                out += '"';
            }
            out += "}]";
        }
        out += "]}";
        return out;
    }

    void parse_snapshot(const std::string& s) {
        // Tiny hand-rolled parser for our exact snapshot grammar:
        // {"records":[[k,{"c":"v",...}], ...]}
        const char* p = s.c_str();
        auto skip_ws = [](const char*& q){ while(*q==' '||*q=='\t'||*q=='\n'||*q=='\r') ++q; };
        auto expect = [](const char*& q, char c){ if(*q!=c) throw std::runtime_error("snapshot: parse"); ++q; };
        auto read_string = [&](const char*& q) -> std::string {
            expect(q, '"');
            std::string o;
            while (*q && *q != '"') {
                if (*q == '\\') {
                    ++q;
                    switch (*q) {
                        case '"': o += '"'; break;
                        case '\\': o += '\\'; break;
                        case '/': o += '/'; break;
                        case 'b': o += '\b'; break;
                        case 'f': o += '\f'; break;
                        case 'n': o += '\n'; break;
                        case 'r': o += '\r'; break;
                        case 't': o += '\t'; break;
                        case 'u': {
                            ++q; char hex[5]={0};
                            for(int i=0;i<4;++i){ if(!*q) throw std::runtime_error("bad \\u"); hex[i]=*q++; }
                            int code = (int)std::strtol(hex,nullptr,16);
                            if(code>0x7F) throw std::runtime_error("non-ASCII \\u unsupported");
                            o += (char)code; --q; break;
                        }
                        default: throw std::runtime_error("bad esc");
                    }
                    ++q;
                } else { o += *q++; }
            }
            expect(q, '"');
            return o;
        };

        skip_ws(p);
        expect(p, '{'); skip_ws(p);
        // Expect a single key "records"
        std::string key = read_string(p); skip_ws(p);
        if (key != "records") throw std::runtime_error("snapshot: missing records");
        expect(p, ':'); skip_ws(p); expect(p, '[');
        if (*p == ']') { ++p; return; }
        while (true) {
            skip_ws(p); expect(p, '[');
            char* end = nullptr;
            errno = 0;
            long long k = std::strtoll(p, &end, 10);
            if (end == p) throw std::runtime_error("snapshot: bad key");
            p = end; skip_ws(p); expect(p, ','); skip_ws(p);
            expect(p, '{');
            std::map<std::string,std::string> cols;
            skip_ws(p);
            if (*p != '}') {
                while (true) {
                    skip_ws(p);
                    std::string ck = read_string(p); skip_ws(p);
                    expect(p, ':'); skip_ws(p);
                    std::string cv = read_string(p);
                    cols.emplace(std::move(ck), std::move(cv));
                    skip_ws(p);
                    if (*p == ',') { ++p; continue; }
                    if (*p == '}') break;
                    throw std::runtime_error("snapshot: expected , or }");
                }
            }
            expect(p, '}'); skip_ws(p); expect(p, ']');
            Record r; r.key = (std::int64_t)k; r.columns = std::move(cols);
            records_[r.key] = std::move(r);
            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == ']') { ++p; break; }
            throw std::runtime_error("snapshot: expected , or ]");
        }
    }

    // fsync a path that we've just written via ofstream.
    //
    // On POSIX, opening "rb" is fine because fsync() works on any fd with a
    // backing file regardless of the open mode.
    //
    // On Windows, FlushFileBuffers *requires* a handle opened with write access
    // (else ERROR_ACCESS_DENIED / err=5). We open "r+b" (read+write, no
    // truncation) so the handle has write access without disturbing the file.
    // The directory fsync is best-effort: swallow errors since not all
    // filesystems support fsyncing a directory handle.
    static void fsync_path(const fs::path& p) {
#ifdef _WIN32
        std::FILE* f = std::fopen(p.string().c_str(), "r+b");
#else
        std::FILE* f = std::fopen(p.string().c_str(), "rb");
#endif
        if (f) { detail::fsync_file(f); std::fclose(f); }
        // Directory sync (best-effort, swallow errors).
        const auto dir = p.parent_path().string();
        std::FILE* d = std::fopen(dir.c_str(),
#ifdef _WIN32
            "r+b");
#else
            "rb");
#endif
        if (d) { try { detail::fsync_file(d); } catch(...) {} std::fclose(d); }
    }
};

} // namespace hylis::storage
