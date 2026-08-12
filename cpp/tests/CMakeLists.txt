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
    # Benchmark baseline only; tests guard on HYLIS_HAS_HNSWLIB and skip
    # themselves when it was not fetched.
    if(HYLIS_HAS_HNSWLIB)
        target_link_libraries(${name} PRIVATE hnswlib_headers)
        target_compile_definitions(${name} PRIVATE HYLIS_HAS_HNSWLIB)
    endif()
    # Tests need a writable temp dir; we pass the source dir so they can make
    # their own per-test subdirs under build/test-tmp.
    target_compile_definitions(${name} PRIVATE
        HYLIS_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp")
    # DISCOVERY_MODE PRE_TEST, not the POST_BUILD default.
    #
    # In POST_BUILD mode each target's test list is enumerated as a build
    # step, into a file named with a hash of the target's properties. Those
    # hashes all came out as e3b0c442 here — the SHA1 of the empty string —
    # so with Ninja building the test binaries in parallel the discovery
    # steps raced and clobbered one another. The observed result was
    # test_flat's registration being filled with test_storage's test names
    # pointed at test_flat.exe: 28 real tests silently not running, and 12
    # phantom ones "passing" because --gtest_filter matched nothing and
    # gtest exits 0. A green suite that tests nothing is the worst possible
    # failure mode, so discovery is deferred to ctest run time instead.
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
endfunction()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test-tmp)

hylis_add_test(test_storage)
hylis_add_test(test_btree)
hylis_add_test(test_flat)
hylis_add_test(test_rmi)
hylis_add_test(test_dynamic_rmi)
hylis_add_test(test_column_index)
hylis_add_test(test_schema)
hylis_add_test(test_typed_column)
hylis_add_test(test_hnsw)
hylis_add_test(test_router)
hylis_add_test(test_three_way)
hylis_add_test(test_planner)
