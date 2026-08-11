# hylis — Hybrid Learned Index System
# Root build file. See README.md for the architecture overview.
#
# Design notes (defendable in the viva):
#
# * C++17, no vendor extensions. `-Wpedantic` on GCC/Clang and `/permissive-`
#   on MSVC enforce standards conformance, so the code is portable across
#   GCC, Clang and MSVC out of the box.
# * The core library (storage + B+ tree) is built as a static lib `hylis_core`
#   so the GoogleTest binaries and the pybind11 extension can both link it
#   without recompiling.
# * pybind11 and GoogleTest are pulled in via FetchContent — no system-level
#   install required, fully reproducible from source. (Needs git on PATH, which
#   the project setup guarantees.)
# * HYLIS_BUILD_TESTS (default ON) and HYLIS_BUILD_PYTHON (default ON) let a
#   reviewer build just the C++ core if they don't have Python handy.

cmake_minimum_required(VERSION 3.18)

# Project-level policies: prefer the modern module path behaviour.
project(hylis
    VERSION 0.1.0
    DESCRIPTION "Hybrid Learned Index System: B+ tree + learned index + neural-routed HNSW"
    LANGUAGES CXX)

# ----------------------------------------------------------------- options
option(HYLIS_BUILD_TESTS  "Build the GoogleTest unit tests"         ON)
option(HYLIS_BUILD_PYTHON "Build the pybind11 Python bindings"      ON)

# hnswlib is a *benchmark baseline only* — nothing in the engine may depend on
# it, per the README's from-scratch constraint. It is here to answer one
# question honestly: is our HNSW competitive, or a strawman that flatters the
# numbers?
#
# Fetching it needs network, exactly as GoogleTest and pybind11 already do, so
# defaulting it ON adds no new requirement to a fresh clone. Configure with
# -DHYLIS_WITH_HNSWLIB=OFF to build without it; the `hylis._hnswlib` module is
# still produced, reports available=False, and every test and benchmark that
# uses it skips itself.
option(HYLIS_WITH_HNSWLIB "Fetch nmslib/hnswlib as a benchmark baseline" ON)

# --------------------------------------------------------- build type
# A bare `cmake -B build` leaves CMAKE_BUILD_TYPE empty on single-config
# generators, which means no optimisation flag at all — not even -O1. That is
# a quiet trap for a project whose deliverable is a set of benchmark numbers:
# an unoptimised scan measures roughly 4x slower than a real one, so every
# comparison drawn from it is wrong while looking perfectly plausible.
# Default to Release; `--preset default` still asks for Debug explicitly.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug/Release/RelWithDebInfo/MinSizeRel)" FORCE)
    message(STATUS "hylis: CMAKE_BUILD_TYPE was unset; defaulting to Release")
endif()

# ----------------------------------------------------------- C++ standard
# We require C++17. `req_on_std` semantics: CMake sets the standard as a hard
# requirement, so consumers can't accidentally downgrade it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # -std=c++17, not -std=gnu++17 → portable

# ------------------------------------------------------- portable warnings
if(MSVC)
    # /permissive- = strict standards conformance. /W4 = high warning level.
    # /WX would treat warnings as errors; we keep them as warnings so the build
    # doesn't break on a noisy platform, but we keep the code warning-clean.
    add_compile_options(/W4 /permissive- /constexpr:steps2000000)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Export compile commands so editors / clangd can introspect the build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------- MinGW: static C++ runtime
# Without this, binaries depend on libstdc++-6.dll / libgcc_s_seh-1.dll, which
# breaks in two ways on Windows:
#   * Git for Windows ships its own copy of both in a directory that usually
#     precedes the real toolchain on PATH, so a binary loads a mismatched
#     libstdc++ and dies with STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139).
#   * Python 3.8+ ignores PATH when resolving an extension module's dependent
#     DLLs, so the .pyd cannot find them at all.
# Linking them statically makes both the test binaries and the .pyd
# self-contained, which is also what we want for a reviewer's machine.
# `-static` also covers libwinpthread-1.dll, which this toolchain's posix
# threading model pulls in and which would otherwise reintroduce the same
# problem. python3xx.dll stays dynamic (it is linked via an import library),
# which is correct — the interpreter always provides it.
if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

# ------------------------------------------------------------- hnswlib
# Declared here rather than in a subdirectory so both the tests and the
# bindings can link it. HYLIS_HAS_HNSWLIB is the single flag everything else
# keys off.
set(HYLIS_HAS_HNSWLIB OFF)
if(HYLIS_WITH_HNSWLIB)
    include(FetchContent)
    FetchContent_Declare(
        hnswlib
        GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
        GIT_TAG v0.8.0
        EXCLUDE_FROM_ALL   # we want the headers, not its examples and tests
    )
    FetchContent_MakeAvailable(hnswlib)

    if(EXISTS "${hnswlib_SOURCE_DIR}/hnswlib/hnswlib.h")
        # Our own interface target rather than hnswlib's, so the include can be
        # marked SYSTEM — third-party headers should not have to satisfy this
        # project's -Wall -Wextra -Wpedantic.
        add_library(hnswlib_headers INTERFACE)
        target_include_directories(hnswlib_headers SYSTEM INTERFACE
            "${hnswlib_SOURCE_DIR}")
        set(HYLIS_HAS_HNSWLIB ON)
        message(STATUS "hylis: hnswlib available as a benchmark baseline")
    else()
        message(WARNING "hylis: hnswlib headers not found after fetch; building "
                        "without the external baseline.")
    endif()
endif()

# -------------------------------------------------------------- subdirs
add_subdirectory(cpp/hylis)          # core static library target: hylis_core

if(HYLIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

if(HYLIS_BUILD_PYTHON)
    add_subdirectory(cpp/bindings)
endif()
