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

} // namespace hylis::storage::detail
