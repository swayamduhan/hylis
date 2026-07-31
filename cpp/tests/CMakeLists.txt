# GoogleTest unit tests for the C++ core.
#
# GoogleTest is fetched via FetchContent so the project is self-contained —
# a fresh clone + `cmake -B build` is all that's needed (plus git on PATH,
# which the toolchain setup guarantees). CTest picks up each test binary.

include(FetchContent)

# Pin GoogleTest to a release tag for reproducibility. Bumping is a deliberate
# decision, not something that happens implicitly.
set(GTEST_TAG v1.15.2)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${GTEST_TAG}
)
# On Windows / MinGW we want shared runtime consistency; this guard prevents
# the "dllinterface" warning noise when GTest is built against a different
# runtime than the test binary.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Helper: declare a C++ test binary that links hylis_core + GTest and registers
# it with CTest.
function(hylis_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE hylis_core GTest::gtest_main)
    # Tests need a writable temp dir; we pass the source dir so they can make
    # their own per-test subdirs under build/test-tmp.
    target_compile_definitions(${name} PRIVATE
        HYLIS_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp")
    gtest_discover_tests(${name})
endfunction()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test-tmp)

hylis_add_test(test_storage)
hylis_add_test(test_btree)
