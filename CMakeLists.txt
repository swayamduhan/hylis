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

# -------------------------------------------------------------- subdirs
add_subdirectory(cpp/hylis)          # core static library target: hylis_core

if(HYLIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

if(HYLIS_BUILD_PYTHON)
    add_subdirectory(cpp/bindings)
endif()
