# pybind11 bindings. Each .cpp file becomes one importable Python module.
#
# pybind11 is fetched via FetchContent (header-only, no pip install required).
# We discover the target Python via find_package(Python), preferring the one
# on PATH.

include(FetchContent)

set(PYBIND11_TAG v2.13.6)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG ${PYBIND11_TAG}
)
FetchContent_MakeAvailable(pybind11)

# Built extensions land directly inside the Python package so `import
# hylis._storage` works straight from a build tree, with no install step and
# no PYTHONPATH juggling beyond putting python/ on sys.path (see conftest.py).
set(HYLIS_PY_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/python/hylis")

# Declare one pybind11 module per binding source, output into the package dir.
# `name` is the importable module name (leading underscore), `source` the .cpp.
function(hylis_add_binding name source)
    pybind11_add_module(${name} ${source})
    target_link_libraries(${name} PRIVATE hylis_core)
    # Lets Python see how the extension was compiled, so a benchmark can
    # refuse to report timings from an unoptimised build.
    target_compile_definitions(${name} PRIVATE
        HYLIS_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
    # Multi-config generators (MSVC) append a per-config subdir unless the
    # per-config property is set too, so set both.
    set_target_properties(${name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${HYLIS_PY_PACKAGE_DIR}")
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${cfg} cfg_upper)
        set_target_properties(${name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} "${HYLIS_PY_PACKAGE_DIR}")
    endforeach()
endfunction()

# Each binding module is declared here once its .cpp exists.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/storage.cpp")
    hylis_add_binding(_storage storage.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/btree.cpp")
    hylis_add_binding(_btree btree.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flat.cpp")
    hylis_add_binding(_flat flat.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rmi.cpp")
    hylis_add_binding(_rmi rmi.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hnsw.cpp")
    hylis_add_binding(_hnsw hnsw.cpp)
endif()
