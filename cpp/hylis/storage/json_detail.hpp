// storage/json_detail.hpp
//
// Minimal hand-rolled JSON primitives shared by the WAL line format and the
// checkpoint snapshot format. Both only ever serialize plain
// string/int/object-of-strings values, so one small parser/writer covers
// both instead of maintaining two copies of the same escaping and parsing
// logic (as the code used to).

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace hylis::storage::json_detail {

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
                // Escape other control chars as \uXXXX so the file stays
                // valid JSON even if a payload contains binary.
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

// ---- tiny cursor-based reader ------------------------------------------
// Each function advances the `const char*&` cursor past what it consumed
// and throws std::runtime_error on malformed input. Callers treat a throw
// as "this line/entry is corrupt" (e.g. a torn write) and skip it.

inline void skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
}

inline void expect(const char*& p, char c) {
    if (*p != c) throw std::runtime_error(std::string("expected '") + c + "'");
    ++p;
}

inline bool consume_null(const char*& p) {
    if (p[0] == 'n' && p[1] == 'u' && p[2] == 'l' && p[3] == 'l') {
        p += 4;
        return true;
    }
    return false;
}

inline std::int64_t read_int(const char*& p) {
    char* end = nullptr;
    const long long v = std::strtoll(p, &end, 10);
    if (end == p) throw std::runtime_error("expected integer");
    p = end;
    return static_cast<std::int64_t>(v);
}

// Reads a JSON string including the surrounding quotes, unescaping as it
// goes. Only supports the escapes we ourselves emit, plus \u00XX for
// control chars (anything above 0x7F throws rather than silently mangling).
inline std::string read_string(const char*& p) {
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
                    ++p;
                    char hex[5] = {0};
                    for (int i = 0; i < 4; ++i) {
                        if (!*p) throw std::runtime_error("bad \\u");
                        hex[i] = *p++;
                    }
                    const int code = static_cast<int>(std::strtol(hex, nullptr, 16));
                    if (code > 0x7F) throw std::runtime_error("non-ASCII \\u unsupported");
                    out += static_cast<char>(code);
                    --p; // pre-incremented above; the loop's trailing ++p accounts for it
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

inline double read_double(const char*& p) {
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p) throw std::runtime_error("expected number");
    p = end;
    return v;
}

// Reads a flat [n, n, ...] array of numbers. Used for neural-router weights,
// which are the only float payload in the project — everything else the
// engine persists is strings and integers.
inline std::vector<double> read_number_array(const char*& p) {
    std::vector<double> out;
    skip_ws(p);
    expect(p, '[');
    skip_ws(p);
    if (*p == ']') { ++p; return out; }
    while (true) {
        skip_ws(p);
        out.push_back(read_double(p));
        skip_ws(p);
        if (*p == ',') { ++p; continue; }
        if (*p == ']') { ++p; break; }
        throw std::runtime_error("expected , or ] in number array");
    }
    return out;
}

// Reads a flat [n, n, ...] array of integers, exactly.
//
// Not read_number_array with a cast: record keys are int64 and doubles hold
// only 53 bits of mantissa, so a key above 2^53 would come back as a
// *different* key with no error reported anywhere. The one place this is used
// — the row -> record key map of a vector column — is precisely where that
// would silently associate an embedding with the wrong row.
inline std::vector<std::int64_t> read_int_array(const char*& p) {
    std::vector<std::int64_t> out;
    skip_ws(p);
    expect(p, '[');
    skip_ws(p);
    if (*p == ']') { ++p; return out; }
    while (true) {
        skip_ws(p);
        out.push_back(read_int(p));
        skip_ws(p);
        if (*p == ',') { ++p; continue; }
        if (*p == ']') { ++p; break; }
        throw std::runtime_error("expected , or ] in integer array");
    }
    return out;
}

// Reads a flat {"str":"str", ...} object into a map.
inline std::map<std::string, std::string> read_string_object(const char*& p) {
    std::map<std::string, std::string> out;
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

// Skips one value of unknown shape (string/object/null/bare token) — used
// to tolerate unrecognized fields without needing a full JSON grammar.
inline void skip_value(const char*& p) {
    skip_ws(p);
    if (*p == '"') { (void)read_string(p); }
    else if (*p == '{') { (void)read_string_object(p); }
    else if (*p == 'n') { (void)consume_null(p); }
    else if (*p == '[') {
        // Arrays need real bracket matching: the bare-token fallback below
        // stops at the first comma, which is *inside* an array rather than
        // after it, and would leave the cursor mid-value.
        int depth = 0;
        do {
            if (*p == '[') ++depth;
            else if (*p == ']') --depth;
            else if (*p == '"') { (void)read_string(p); continue; }
            ++p;
        } while (*p && depth > 0);
    }
    else { while (*p && *p != ',' && *p != '}') ++p; }
}

} // namespace hylis::storage::json_detail
