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

# Each binding module is declared here once its .cpp exists.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/storage.cpp")
    pybind11_add_module(_storage storage.cpp)
    target_link_libraries(_storage PRIVATE hylis_core)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/btree.cpp")
    pybind11_add_module(_btree btree.cpp)
    target_link_libraries(_btree PRIVATE hylis_core)
endif()
