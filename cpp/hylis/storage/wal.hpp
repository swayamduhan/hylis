// storage/wal.hpp
//
// Write-Ahead Log (WAL).
//
// Before any mutation is applied to memory, RecordStore appends a record
// describing it here and fsyncs the file — the "write-ahead" rule. On
// recovery we replay the log in order; `put`/`delete` are naturally
// idempotent, so replaying them reconstructs the exact final state even if
// we crashed mid-sequence.
//
// Format: append-only NDJSON (one JSON object per line), hand-rolled rather
// than via a JSON library — the payload is always string->string, so a full
// parser would be overkill. NDJSON specifically (over a binary format)
// because a torn final line (crash mid-write) is safely recoverable: we
// just skip unparseable lines on replay instead of aborting, and the file
// stays greppable for demos/debugging.
//
// After a checkpoint (see RecordStore) the log is truncated to bound replay
// time; the truncation itself is logged as a sentinel entry.

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/detail.hpp"
#include "storage/json_detail.hpp"

namespace hylis::storage {

inline constexpr const char* OP_PUT        = "put";        // upsert (full payload)
inline constexpr const char* OP_DELETE     = "delete";     // remove by key
inline constexpr const char* OP_CHECKPOINT = "checkpoint"; // snapshot marker

// A single WAL entry. `lsn` is the monotonic log sequence number assigned at
// append time, giving a total order on operations.
struct LogEntry {
    std::int64_t lsn = -1;
    std::string op;
    std::optional<std::int64_t> key;
    std::optional<std::map<std::string, std::string>> payload;
};

namespace detail_wal {

// One NDJSON line for a LogEntry. Fixed key order (lsn, op, key, payload)
// for stable diffs and readable logs.
inline std::string to_json_line(const LogEntry& e) {
    using json_detail::escape_string;
    std::string out;
    out += "{\"lsn\":";
    out += std::to_string(e.lsn);
    out += ",\"op\":\"";
    out += escape_string(e.op);
    out += "\",\"key\":";
    if (e.key) out += std::to_string(*e.key); else out += "null";
    out += ",\"payload\":";
    if (e.payload) {
        out += '{';
        bool first = true;
        for (const auto& [k, v] : *e.payload) {
            if (!first) out += ',';
            first = false;
            out += '"';
            out += escape_string(k);
            out += "\":\"";
            out += escape_string(v);
            out += '"';
        }
        out += '}';
    } else {
        out += "null";
    }
    out += "}\n";
    return out;
}

// Parses one NDJSON line into a LogEntry. Throws std::runtime_error on any
// malformation; the caller treats that as "torn line, skip".
inline LogEntry parse_line(const std::string& line) {
    using namespace json_detail;
    LogEntry e;
    const char* p = line.c_str();
    expect(p, '{');

    while (*p && *p != '}') {
        skip_ws(p);
        if (*p == ',') { ++p; skip_ws(p); }
        if (*p == '}') break;
        std::string key = read_string(p);
        skip_ws(p);
        expect(p, ':');
        skip_ws(p);
        if (key == "lsn") {
            e.lsn = read_int(p);
        } else if (key == "op") {
            e.op = read_string(p);
        } else if (key == "key") {
            e.key = consume_null(p) ? std::optional<std::int64_t>{} : read_int(p);
        } else if (key == "payload") {
            e.payload = consume_null(p)
                ? std::optional<std::map<std::string, std::string>>{}
                : read_string_object(p);
        } else {
            skip_value(p); // unknown field: tolerate and skip its value
        }
    }
    if (e.lsn < 0) throw std::runtime_error("missing lsn");
    return e;
}

} // namespace detail_wal

class WriteAheadLog {
public:
    explicit WriteAheadLog(std::string path)
        : path_(std::move(path)) {
        // "ab": append + binary, created if absent. We never seek backwards —
        // durability comes from append + truncate, never in-place edits.
        file_ = std::fopen(path_.c_str(), "ab");
        if (!file_) {
            throw std::runtime_error("WriteAheadLog: cannot open " + path_);
        }
        seed_lsn_from_existing_entries();
    }

    ~WriteAheadLog() { close(); }

    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

    // The only way records hit the log. Callers must call this and wait for
    // it to return *before* mutating in-memory state — that ordering is the
    // entire correctness guarantee of the WAL.
    LogEntry append(const std::string& op,
                    const std::optional<std::int64_t>& key,
                    const std::optional<std::map<std::string, std::string>>& payload) {
        if (op != OP_PUT && op != OP_DELETE && op != OP_CHECKPOINT) {
            throw std::runtime_error("WriteAheadLog: unknown op '" + op + "'");
        }
        LogEntry e{next_lsn_++, op, key, payload};
        const std::string line = detail_wal::to_json_line(e);
        if (std::fwrite(line.data(), 1, line.size(), file_) != line.size()) {
            throw std::runtime_error("WriteAheadLog: write failed");
        }
        detail::fsync_file(file_); // force onto disk before we tell the caller we're done
        return e;
    }

    LogEntry append_put(std::int64_t key, std::map<std::string, std::string> payload) {
        return append(OP_PUT, key, std::move(payload));
    }
    LogEntry append_delete(std::int64_t key) {
        return append(OP_DELETE, key, std::nullopt);
    }
    LogEntry append_checkpoint() {
        return append(OP_CHECKPOINT, std::nullopt, std::nullopt);
    }

    // Reads every *parseable* entry in order. A torn tail line (crash
    // mid-write) is necessarily the last line, so skipping it loses nothing.
    std::vector<LogEntry> iter_entries() const {
        std::vector<LogEntry> out;
        std::ifstream in(path_, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            try {
                out.push_back(detail_wal::parse_line(line));
            } catch (const std::runtime_error&) {
                // Torn line — the WAL's contract is "recover everything that
                // fully landed on disk", so we swallow this deliberately.
            }
        }
        return out;
    }

    // Empties the log. Called by RecordStore after a checkpoint has
    // snapshotted state elsewhere. LSNs keep growing across truncations —
    // they're global, not reset per-file.
    void truncate() {
        if (file_) { std::fclose(file_); file_ = nullptr; }
        std::FILE* tf = std::fopen(path_.c_str(), "wb"); // reopen in truncate mode
        if (tf) std::fclose(tf);
        file_ = std::fopen(path_.c_str(), "ab");         // then flip back to append
        if (!file_) throw std::runtime_error("WriteAheadLog: reopen failed post-truncate");
    }

    void close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    std::int64_t next_lsn() const { return next_lsn_; }
    const std::string& path() const { return path_; }

private:
    // Walk the file to find the highest LSN so a reopened log keeps growing
    // LSNs instead of restarting at 0. Torn lines are tolerated as usual.
    void seed_lsn_from_existing_entries() {
        for (const auto& e : iter_entries()) {
            if (e.lsn >= next_lsn_) next_lsn_ = e.lsn + 1;
        }
    }

    std::string path_;
    std::FILE* file_ = nullptr;
    std::int64_t next_lsn_ = 0;
};

} // namespace hylis::storage
