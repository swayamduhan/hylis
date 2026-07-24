## Revised scope — C++ core + pybind11 bridge

**You decided:** portable C++17 + CMake (MinGW/Clang day-to-day on Windows, also builds clean on Linux/macOS GCC/Clang for evaluation), and a **pybind11** bridge so the ML/router/planner modules stay Python while the perf-critical storage + B+ tree core is C++. This is a genuinely good architecture for a viva — *exactly* how real hybrid systems (e.g. FAISS, DuckDB, ClickHouse) are layered.

### Housekeeping first
- Delete the stale Python-only scaffolding I wrote earlier (`hylis/__init__.py`, `hylis/storage/*`). The `README.md` will be rewritten to reflect the C++/Python split.

---

## Part A — Build system & layout

```
hylis/
  CMakeLists.txt                 # root: C++17, portable flags, options
  cmake/                         # FetchContent recipes for pybind11 + GoogleTest
  cpp/
    hylis/                       # header-only core library (consumed by tests + bindings)
      storage/   {record.hpp, wal.hpp, store.hpp}
      index/     {btree.hpp}
      CMakeLists.txt              # builds the core as a static lib for tests
    bindings/                    # pybind11 modules
      storage.cpp                # -> hylis._storage (Record, RecordStore)
      btree.cpp                  # -> hylis._btree  (BPlusTree)
      CMakeLists.txt
    tests/                       # GoogleTest
      test_storage.cpp
      test_btree.cpp
      CMakeLists.txt
  python/
    hylis/                       # pure-Python user-facing API (thin wrappers)
      __init__.py
      storage.py                 # re-exports _storage + py Record
      index.py                   # re-exports _btree
  tests/                         # pytest, end-to-end (cross-language)
    test_storage.py
    test_btree.py
    test_integration.py
  README.md                      # rewritten: C++ core + Python ML layer
```

**CMake design (all defensible in the viva):**
- `CMAKE_CXX_STANDARD 17`, `-Wall -Wextra -Wpedantic` (GCC/Clang), `/W4 /permissive-` (MSVC). **No vendor extensions** — `-Wpedantic` enforces this on GCC/Clang.
- Portable fsync: `fopen`/`fsync`/`FlushFileBuffers` behind a small `detail::fsync_file()` helper in `storage/detail.hpp` (only Win32/POSIX *fork* in the whole codebase, and it's one function).
- **pybind11 via FetchContent** (no `pip install pybind11` assumption; reproducible from source, header-only). Same for **GoogleTest**.
- Two CMake targets: `hylis_core` (static lib, linked by tests) and the pybind11 extension modules. A `HYLIS_BUILD_TESTS` option toggles GoogleTest.
- Build outputs land in `build/`; `.gitignore`-style hygiene.

---

## Part B — Storage core (C++)

Same semantics as the earlier Python design, ported to C++ — and now genuinely fast.

- **`Record`** (in `record.hpp`): `int64_t key` + `std::unordered_map<std::string,std::string> columns` (string values for now — type-rich columns are an index-layer concern). Trivially copyable value type.
- **`WriteAheadLog`** (`wal.hpp`): append-only file handle, `append(op, key, payload)` writes one NDJSON line + `fflush` + `fsync`/`FlushFileBuffers`. `LogEntry{lsn, op, key, payload}`. `iter_entries()` skips torn tail lines. Monotonic LSN seeded from existing file at construction.
- **`RecordStore`** (`store.hpp`): owns the WAL + a `std::unordered_map<int64_t,Record>`. `put`/`delete`/`get` apply the **write-ahead rule** (log → fsync → memory). `checkpoint()` atomic-writes `checkpoint.json` via temp-file + `std::filesystem::rename`, appends checkpoint marker, truncates WAL. `_recover()` = load checkpoint + replay WAL (idempotent).

---

## Part C — Storage tests (GoogleTest, C++ side)
`cpp/tests/test_storage.cpp`: put/get/delete, iteration, WAL replay across "crash" (drop object, reopen), LSN monotonic across reopen, checkpoint truncation + restore, torn-line tolerance, and a direct invariant check that every live key has a logged `put`.

---

## Part D — pybind11 storage binding
`cpp/bindings/storage.cpp` → importable as `hylis._storage`. Thin Python wrappers in `python/hylis/storage.py` give a clean dict-like API. Verified by `tests/test_storage.py` (pytest) driving the C++ engine through the binding.

---

## Part E — B+ tree (C++)
The main event.

- **Generic `BPlusTree<Key, Value>`** (`index/btree.hpp`), `Key = int64_t`, `Value = int64_t` (record id) by default but templated for reuse.
- **Distinct node classes** — `LeafNode` and `InternalNode` both deriving a tiny `Node` base (cleaner to explain than a flag, since the shapes differ: leaves hold values + a `next` pointer; internals hold child pointers).
- **Convention (commented in-file):** `order = m`. Max `m−1` keys / `m` children per node; non-root minimum `⌈m/2⌉` keys. Default `m = 32`.
- **Insert + split:** descend → insert in sorted position → leaf overflow splits at midpoint and *copies* the middle key up (leaf copy-up); internal overflow *pushes* the middle key up (internal push-up) — this distinction is the viva gold. Splits propagate to the root; root split grows the tree by one level.
- **Point search:** descend by key comparison to a leaf, binary-search the leaf's sorted key array.
- **Range search `[lo, hi]`:** descend to `lo`'s leaf, then walk the leaf `next` chain collecting keys `≤ hi`.
- **Delete + rebalance:** remove → on underflow try **borrow from a sibling** (rotate through the parent's separator), else **merge** (pull the separator down); underflow propagates up; a root left with one child is replaced by that child (tree shrinks). `min_keys` enforced for non-root only.
- **`validate()`**: invariant checker (sorted keys, child-count bounds, separator correctness, leaf chain integrity) for tests/debug.

---

## Part F — B+ tree tests (GoogleTest, C++ side)
`cpp/tests/test_btree.cpp`: hand-crafted cases for split (copy-up vs push-up), borrow-left/right, merge, and root collapse at `m = 3` (small order so every structural case fires). Plus a **randomized differential test** vs `std::map` over thousands of mixed insert/delete/point/range ops with seeded RNG — this is what gives real confidence the tree matches a known-good reference.

---

## Part G — pybind11 B+ tree binding + integration test
`cpp/bindings/btree.cpp` → `hylis._btree`. `tests/test_btree.py` (pytest) exercises it through Python. `tests/test_integration.py` builds a `RecordStore`, mirrors inserts into a `BPlusTree` over PKs, and verifies point + range lookups return the right records — proving the two layers compose end-to-end through the bridge.

---

## Verification bar (before I report back)
1. `cmake -B build` + `cmake --build build` compiles **clean with `-Wall -Wextra -Wpedantic`** (no warnings on GCC/Clang, `/W4` clean on MSVC).
2. `ctest --test-dir build` — all GoogleTest cases green.
3. `pytest` — all Python tests green, proving the bindings actually work.
4. I'll show you the build + test output, not just claim it.

## Explicitly out of scope for this step
Vector search, learned index, HNSW, neural router, planner, benchmarks, CLI. No FAISS/hnswlib/etc. anywhere in the core. The C++17 + CMake + pybind11 foundation built here is what all later modules plug into.

**Approve and I'll execute in order: layout → CMake → storage core → storage tests → binding → B+ tree → tree tests → binding → integration → show green build+test output.**