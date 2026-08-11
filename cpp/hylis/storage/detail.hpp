// storage/detail.hpp
//
// Tiny platform-abstraction helpers for the storage layer.
//
// This is the *only* place in the codebase that branches on the OS. Keeping
// the platform fork in one file means the rest of the engine stays portable
// and clean, and means a reviewer can audit "what's Windows-specific?" by
// reading exactly one header.
//
// Why we need it: a durable write-ahead log must force bytes onto the disk
// (past the OS page cache), and the syscall to do that is OS-specific:
//   * POSIX:  fsync(fd)
//   * Windows: FlushFileBuffers(handle)
// Everything else in storage/ is pure standard C++17.

#pragma once

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  include <windows.h>            // FlushFileBuffers, GetOsErrorMessage via FORMAT_MESSAGE
#  include <io.h>                 // _get_osfhandle
#else
#  include <unistd.h>             // fsync
#endif

namespace hylis::storage::detail {

// Force the given C FILE* to be physically on disk. Throws std::runtime_error
// on failure (durability failure is fatal for a WAL — better to surface it).
inline void fsync_file(std::FILE* f) {
    if (f == nullptr) return;

#if defined(_WIN32)
    // On Windows, FlushFileBuffers works on the underlying OS file handle.
    // std::fflush empties the userspace CRT buffer first, then we tell the
    // kernel to flush its cache too.
    if (std::fflush(f) != 0) {
        throw std::runtime_error("fsync_file: fflush failed");
    }
    // _get_osfhandle converts the CRT fd into a Windows HANDLE.
    const int fd = ::_fileno(f);
    if (fd < 0) {
        throw std::runtime_error("fsync_file: _fileno failed");
    }
    const intptr_t handle = ::_get_osfhandle(fd);
    if (handle == (intptr_t)-1) {
        throw std::runtime_error("fsync_file: _get_osfhandle failed");
    }
    if (::FlushFileBuffers(reinterpret_cast<HANDLE>(handle)) == 0) {
        throw std::runtime_error("fsync_file: FlushFileBuffers failed (err="
                                 + std::to_string(::GetLastError()) + ")");
    }
#else
    if (std::fflush(f) != 0) {
        throw std::runtime_error("fsync_file: fflush failed");
    }
    const int fd = ::fileno(f);
    if (fd < 0) {
        throw std::runtime_error("fsync_file: fileno failed");
    }
    if (::fsync(fd) != 0) {
        throw std::runtime_error("fsync_file: fsync failed");
    }
#endif
}

// Force a path that was just written (typically via ofstream) onto the disk,
// along with its containing directory so the entry itself is durable.
//
// On POSIX, opening "rb" is fine — fsync() works on any fd with a backing
// file regardless of open mode. On Windows, FlushFileBuffers requires a
// handle with write access, so we open "r+b" (read+write, no truncation)
// instead. Directory fsync is best-effort: not every filesystem supports
// fsyncing a directory handle, and failing to do so is not worth aborting a
// write that already succeeded.
inline void fsync_path(const std::string& path) {
#ifdef _WIN32
    std::FILE* f = std::fopen(path.c_str(), "r+b");
#else
    std::FILE* f = std::fopen(path.c_str(), "rb");
#endif
    if (f) { fsync_file(f); std::fclose(f); }

    const std::string dir = path.substr(0, path.find_last_of("/\\") + 1);
    if (dir.empty()) return;
#ifdef _WIN32
    std::FILE* d = std::fopen(dir.c_str(), "r+b");
#else
    std::FILE* d = std::fopen(dir.c_str(), "rb");
#endif
    if (d) { try { fsync_file(d); } catch (...) {} std::fclose(d); }
}

// Write `blob` to `path` so that a crash can never leave a partial file:
// write a sibling temp file, force it to disk, then rename over the target.
// rename() is atomic within a directory on every filesystem hylis targets, so
// a reader sees either the whole old file or the whole new one.
//
// Shared by the record store's checkpoint and the index catalog — both need
// exactly this guarantee, and there should only be one implementation of it.
inline void atomic_write(const std::string& path, const std::string& temp_path,
                         const std::string& blob) {
    {
        std::FILE* out = std::fopen(temp_path.c_str(), "wb");
        if (out == nullptr) {
            throw std::runtime_error("atomic_write: cannot open " + temp_path);
        }
        const std::size_t written = std::fwrite(blob.data(), 1, blob.size(), out);
        const bool ok = written == blob.size();
        if (ok) fsync_file(out);
        std::fclose(out);
        if (!ok) throw std::runtime_error("atomic_write: short write to " + temp_path);
    }
    // std::filesystem::rename, not std::rename: C's rename fails on Windows
    // when the target exists, and deleting it first would open a window where
    // the file is simply absent — which is exactly the state this function
    // exists to make impossible. std::filesystem::rename maps to MoveFileEx
    // with MOVEFILE_REPLACE_EXISTING, an atomic replace.
    std::filesystem::rename(temp_path, path);
    fsync_path(path);
}

} // namespace hylis::storage::detail
