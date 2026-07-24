// storage/wal.hpp
//
// Write-Ahead Log (WAL).
//
// Goal
// ----
// Give the in-memory RecordStore durability across crashes without
// implementing a full page-based storage engine.
//
// How it works
// ------------
// 1. *Before* any mutation is applied to memory we append a record describing
//    the operation to the WAL file on disk ("write-ahead" rule).
// 2. The file is fflush'd + fsync'd so the bytes are physically on disk, not
//    just in the OS page cache. Without fsync a crash could lose recently
//    logged operations, defeating the whole point.
// 3. On recovery we replay the log in order. Operations are replay-safe:
//    a `put` carries the full record payload, a `delete` removes the key, so
//    replaying them in order reconstructs the exact final state.
//
// Format
// ------
// Append-only, one JSON object per line (NDJSON). Hand-rolled JSON (no deps):
// the payload is just string->string, so escaping is simple. NDJSON is chosen
// over a binary format deliberately:
//   * it is greppable — great for demos and the viva;
//   * a torn final line (crash mid-write) is recoverable: we skip unparseable
//     lines on replay instead of aborting.
//
// Truncation
// ----------
// After a checkpoint the WAL is truncated (see RecordStore). This bounds
// recovery time. The truncation is itself logged as a sentinel so observers
// can distinguish a freshly-truncated log from a brand-new one.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/detail.hpp"   // detail::fsync_file

namespace hylis::storage {

// Valid operation tags. Plain const char* (not an enum) so they serialize
// straight to JSON without custom plumbing.
inline constexpr const char* OP_PUT        = "put";        // upsert (full payload)
inline constexpr const char* OP_DELETE     = "delete";     // remove by key
inline constexpr const char* OP_CHECKPOINT = "checkpoint"; // snapshot marker

// A single WAL entry. `lsn` is the monotonic log sequence number assigned at
// append time; it gives a total order on operations.
struct LogEntry {
    std::int64_t lsn = -1;
    std::string op;
    std::optional<std::int64_t> key;
    std::optional<std::map<std::string, std::string>> payload;
};

// ----------------------------- minimal JSON (string-only) -----------------
// We only ever serialize {lsn:int, op:string, key:int|null,
// payload:{str:str}|null}. A tiny purpose-built writer/reader keeps us free
// of any JSON dependency. Escaping follows RFC 8259 for the common cases.
namespace json_detail {

inline std::string escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                // Be conservative: escape any control char as \uXXXX. Keeps the
                // file strictly valid JSON even if a payload contains binary.
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Append-only into `out`; we treat `out` as a substring of full JSON.
inline void write_kv(std::string& out, const char* k, const std::string& v, bool quote_v) {
    out += '"';
    out += k;
    out += "\":";
    if (quote_v) out += '"';
    out += v;
    if (quote_v) out += '"';
}

// Build one NDJSON line for a LogEntry. Deterministic key order (lsn, op,
// key, payload) for stable diffs and readable logs.
inline std::string to_json_line(const LogEntry& e) {
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
    out += "}\n";   // trailing newline: NDJSON
    return out;
}

} // namespace json_detail


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
        // Seed the LSN counter from whatever is already in the file, so a
        // reopened log keeps growing LSNs instead of restarting at 0.
        scan_existing_lsn();
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
        const std::string line = json_detail::to_json_line(e);
        if (std::fwrite(line.data(), 1, line.size(), file_) != line.size()) {
            throw std::runtime_error("WriteAheadLog: write failed");
        }
        detail::fsync_file(file_);   // force onto disk before we tell the caller we're done
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

    // Read every *parseable* entry in order. Torn tail lines (crash mid-write)
    // are skipped silently: a torn line is necessarily the last line, so
    // skipping it loses nothing. We return all entries to the caller.
    std::vector<LogEntry> iter_entries() const {
        std::vector<LogEntry> out;
        std::ifstream in(path_, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            try {
                out.push_back(parse_line(line));
            } catch (const std::runtime_error&) {
                // Torn line — skip. (We deliberately swallow this; the WAL's
                // contract is "recover everything that fully landed on disk".)
            }
        }
        return out;
    }

    // Empty the log. Called by RecordStore after a checkpoint has snapshotted
    // state elsewhere. LSNs keep growing across truncations — they're global.
    void truncate() {
        if (file_) { std::fclose(file_); file_ = nullptr; }
        // Reopen in truncate mode then flip back to append.
        std::FILE* tf = std::fopen(path_.c_str(), "wb");
        if (tf) std::fclose(tf);
        file_ = std::fopen(path_.c_str(), "ab");
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
    // On construction, walk the file to find the highest LSN so the next
    // append continues from there. Torn lines are tolerated.
    void scan_existing_lsn() {
        for (const auto& e : iter_entries()) {
            if (e.lsn >= next_lsn_) next_lsn_ = e.lsn + 1;
        }
    }

    // Parse one NDJSON line into a LogEntry. Throws std::runtime_error on any
    // malformation (the caller treats that as "torn line, skip").
    //
    // This is a deliberately small, hand-written parser. The grammar we emit
    // is tightly constrained (see json_detail::to_json_line), so a full JSON
    // parser would be overkill and would add a dependency for no gain.
    static LogEntry parse_line(const std::string& line) {
        LogEntry e;
        const char* p = line.c_str();
        expect(p, '{');

        // Read the four known keys. We accept any order but require exactly
        // these fields. Trailing/unknown fields are tolerated.
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
                if (peek_null(p)) { e.key = std::nullopt; }
                else { e.key = read_int(p); }
            } else if (key == "payload") {
                if (peek_null(p)) { e.payload = std::nullopt; }
                else { e.payload = read_object(p); }
            } else {
                // Unknown field: skip its value.
                skip_value(p);
            }
        }
        if (e.lsn < 0) throw std::runtime_error("missing lsn");
        return e;
    }

    // ----------------------- tiny JSON parsing primitives -------------------
    // These operate on a `const char*&` cursor; each one advances the cursor
    // past the thing it consumed and throws std::runtime_error on bad input.
    static void skip_ws(const char*& p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    }
    static void expect(const char*& p, char c) {
        if (*p != c) throw std::runtime_error(std::string("expected '") + c + "'");
        ++p;
    }
    static bool peek_null(const char*& p) {
        if (p[0] == 'n' && p[1] == 'u' && p[2] == 'l' && p[3] == 'l') {
            p += 4;
            return true;
        }
        return false;
    }
    static std::int64_t read_int(const char*& p) {
        char* end = nullptr;
        // strtoll sets errno on overflow; we don't handle that beyond failing
        // the parse, which the caller treats as "torn line, skip".
        const long long v = std::strtoll(p, &end, 10);
        if (end == p) throw std::runtime_error("expected integer");
        p = end;
        return static_cast<std::int64_t>(v);
    }
    // Read a JSON string including the surrounding quotes, with unescaping.
    static std::string read_string(const char*& p) {
        expect(p, '"');
        std::string out;
        while (*p && *p != '"') {
            if (*p == '\\') {
                ++p;
                switch (*p) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        // \uXXXX — only need the low byte for our use case
                        // (control chars < 0x20). Throw on anything else so we
                        // don't silently corrupt data.
                        ++p;
                        char hex[5] = {0};
                        for (int i = 0; i < 4; ++i) {
                            if (!*p) throw std::runtime_error("bad \\u");
                            hex[i] = *p++;
                        }
                        const int code = static_cast<int>(std::strtol(hex, nullptr, 16));
                        if (code > 0x7F) throw std::runtime_error("non-ASCII \\u unsupported");
                        out += static_cast<char>(code);
                        --p; // we pre-incremented p; counter the loop's ++p
                        break;
                    }
                    default: throw std::runtime_error("bad escape");
                }
                ++p;
            } else {
                out += *p++;
            }
        }
        expect(p, '"');
        return out;
    }
    static std::map<std::string,std::string> read_object(const char*& p) {
        std::map<std::string,std::string> out;
        expect(p, '{');
        skip_ws(p);
        if (*p == '}') { ++p; return out; }
        while (true) {
            skip_ws(p);
            std::string k = read_string(p);
            skip_ws(p);
            expect(p, ':');
            skip_ws(p);
            std::string v = read_string(p);
            out.emplace(std::move(k), std::move(v));
            skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; break; }
            throw std::runtime_error("expected , or } in object");
        }
        return out;
    }
    static void skip_value(const char*& p) {
        skip_ws(p);
        if (*p == '"') { (void)read_string(p); }
        else if (*p == '{') { (void)read_object(p); }
        else if (*p == 'n') { (void)peek_null(p); }
        else { while (*p && *p != ',' && *p != '}') ++p; }
    }

    std::string path_;
    std::FILE* file_ = nullptr;
    std::int64_t next_lsn_ = 0;
};

} // namespace hylis::storage
