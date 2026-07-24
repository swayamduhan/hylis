# hylis_core — the C++ core library (storage + B+ tree).
#
# We make the core a header-only INTERFACE library *now* (simple, fast
# incremental builds) but define it so that .cpp files can be added later
# without changing downstreams. For a viva-size codebase this is the right
# tradeoff: no link step, simpler CMake, and the optimizer still inlines
# across the headers.

add_library(hylis_core INTERFACE)
target_include_directories(hylis_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(hylis_core INTERFACE cxx_std_17)
