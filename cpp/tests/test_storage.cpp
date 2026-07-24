// tests/test_storage.cpp
//
// Unit tests for the storage layer: Record + WAL + RecordStore.
//
// We verify the guarantees we actually make:
//   * basic put/get/delete + iteration
//   * durability: state survives a "crash" (destroy the object, reopen) via
//     WAL replay
//   * checkpoint: snapshot is loaded and WAL is truncated
//   * torn-line tolerance: a corrupted final WAL line doesn't break recovery
//   * the write-ahead rule: nothing is in memory that isn't in the log first
//
// Each test gets its own subdirectory under build/test-tmp (HYLIS_TEST_TMP)
// so they don't tangle.

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>

#include "storage/store.hpp"
#include "storage/wal.hpp"

namespace fs = std::filesystem;
using hylis::storage::Record;
using hylis::storage::RecordStore;
using hylis::storage::WriteAheadLog;

namespace {
std::string tmp_root() {
    const char* env = std::getenv("HYLIS_TEST_TMP");
    return env ? std::string(env) : std::string("test-tmp");
}

// A fresh per-test directory.
fs::path make_test_dir(const std::string& name) {
    static std::atomic<unsigned> counter{0};
    const auto seed = std::random_device{}();
    fs::path p = fs::path(tmp_root()) / (name + "_" + std::to_string(seed) + "_" +
                                         std::to_string(counter.fetch_add(1)));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

Record r(std::int64_t key, std::map<std::string,std::string> cols = {}) {
    return Record{key, std::move(cols)};
}
} // namespace


// --------------------------------------------------------------- basic ops
TEST(Storage, PutGetRoundtrip) {
    RecordStore s(make_test_dir("putget"));
    s.put(r(1, {{"name","alice"}, {"age","30"}}));
    const Record* got = s.get(1);
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(got->key, 1);
    EXPECT_EQ(got->columns.at("name"), "alice");
    EXPECT_EQ(got->columns.at("age"), "30");
    EXPECT_EQ(s.get(999), nullptr);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_TRUE(s.contains(1));
    EXPECT_FALSE(s.contains(999));
}

TEST(Storage, PutUpdatesSameKey) {
    RecordStore s(make_test_dir("upsert"));
    s.put(r(1, {{"v","a"}}));
    s.put(r(1, {{"v","b"}}));   // upsert
    EXPECT_EQ(s.get(1)->columns.at("v"), "b");
    EXPECT_EQ(s.size(), 1u);
}

TEST(Storage, DeleteExistingAndMissing) {
    RecordStore s(make_test_dir("del"));
    s.put(r(1, {{"v","a"}}));
    EXPECT_TRUE(s.del(1));
    EXPECT_EQ(s.get(1), nullptr);
    EXPECT_FALSE(s.del(1));      // already gone; no log pollution
    EXPECT_EQ(s.size(), 0u);
}

TEST(Storage, IterationReturnsAllRecords) {
    RecordStore s(make_test_dir("iter"));
    for (std::int64_t i = 0; i < 5; ++i) s.put(r(i, {{"i", std::to_string(i)}}));
    std::unordered_set<std::int64_t> seen;
    for (const auto& rec : s.records()) seen.insert(rec.key);
    EXPECT_EQ(seen, (std::unordered_set<std::int64_t>{0,1,2,3,4}));
}

// ------------------------------------------------------------- durability
TEST(Storage, StateSurvivesReopenViaWalReplay) {
    const auto d = make_test_dir("walreplay");
    {
        RecordStore s(d);
        for (std::int64_t i = 0; i < 10; ++i) s.put(r(i, {{"label", "row" + std::to_string(i)}}));
        ASSERT_TRUE(s.del(3));
        s.put(r(7, {{"label","updated"}}));     // update on a different key
        s.close();
    }
    RecordStore s2(d);                            // "crash" + recover
    EXPECT_EQ(s2.size(), 9u);                     // 10 inserts - 1 delete
    EXPECT_EQ(s2.get(0)->columns.at("label"), "row0");
    EXPECT_EQ(s2.get(3), nullptr);                // deleted stays gone
    EXPECT_EQ(s2.get(7)->columns.at("label"), "updated");   // latest write wins
}

TEST(Storage, EmptyStoreReopensEmpty) {
    const auto d = make_test_dir("empty");
    { RecordStore s(d); s.close(); }
    RecordStore s2(d);
    EXPECT_EQ(s2.size(), 0u);
    EXPECT_TRUE(s2.records().empty());
}

TEST(Storage, WalLsnMonotonicAndContinuesAcrossReopen) {
    const auto d = make_test_dir("lsn");
    std::int64_t first;
    {
        RecordStore s(d);
        s.put(r(1, {}));
        first = s.wal_next_lsn_for_test();        // next LSN to be assigned
        s.close();
    }
    RecordStore s2(d);
    EXPECT_EQ(s2.wal_next_lsn_for_test(), first); // counter preserved
}

// ------------------------------------------------------------ checkpoint
TEST(Storage, CheckpointTruncatesWalAndRestoresState) {
    const auto d = make_test_dir("ckpt");
    {
        RecordStore s(d);
        for (std::int64_t i = 0; i < 20; ++i) s.put(r(i, {{"i", std::to_string(i)}}));
        s.checkpoint();

        // WAL should be tiny (just the checkpoint marker line).
        const fs::path wal_path = d / RecordStore::WAL_NAME;
        std::error_code ec;
        const auto sz = fs::file_size(wal_path, ec);
        ASSERT_FALSE(ec) << "wal size lookup failed";
        EXPECT_LT(sz, 256u);                      // just the marker line
        // Read it back and confirm it's the checkpoint marker.
        std::ifstream in(wal_path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(in)), {});
        EXPECT_NE(content.find("checkpoint"), std::string::npos);

        EXPECT_TRUE(fs::exists(d / RecordStore::CHECKPOINT_NAME));

        // More ops after checkpoint, then reopen.
        s.put(r(100, {{"i","100"}}));
        ASSERT_TRUE(s.del(5));
        s.close();
    }
    RecordStore s2(d);
    std::unordered_set<std::int64_t> keys;
    for (auto k : s2.keys()) keys.insert(k);
    // 0..19 minus 5 plus 100.
    std::unordered_set<std::int64_t> expected;
    for (std::int64_t i = 0; i < 20; ++i) if (i != 5) expected.insert(i);
    expected.insert(100);
    EXPECT_EQ(keys, expected);
    EXPECT_EQ(s2.get(100)->columns.at("i"), "100");
}

// ------------------------------------------------------- torn-line safety
TEST(Storage, TornFinalWalLineIsSkippedNotFatal) {
    const auto d = make_test_dir("torn");
    {
        RecordStore s(d);
        s.put(r(1, {{"v","a"}}));
        s.close();
    }
    // Append a partial (torn) line to the file — simulates a crash mid-write.
    const fs::path wal_path = d / RecordStore::WAL_NAME;
    {
        std::ofstream out(wal_path, std::ios::app | std::ios::binary);
        out << R"({"lsn":99,"op":"pu)";   // torn JSON
    }
    // Recovery must succeed and retain the good record.
    RecordStore s2(d);
    EXPECT_NE(s2.get(1), nullptr);
    EXPECT_EQ(s2.get(1)->columns.at("v"), "a");
}

// ----------------------------------------------- write-ahead invariant
TEST(Storage, WriteAheadOrderingMemoryNeverAheadOfLog) {
    const auto d = make_test_dir("wa");
    {
        RecordStore s(d);
        s.put(r(1, {}));
        s.put(r(2, {}));
        s.put(r(3, {}));
        ASSERT_TRUE(s.del(2));
        s.close();
    }
    // Re-open the log read-only and tally.
    WriteAheadLog wal((d / RecordStore::WAL_NAME).string());
    std::unordered_set<std::int64_t> put_keys, del_keys;
    for (const auto& e : wal.iter_entries()) {
        if (e.op == hylis::storage::OP_PUT) put_keys.insert(*e.key);
        else if (e.op == hylis::storage::OP_DELETE) del_keys.insert(*e.key);
    }
    // Final live keys = unique puts - deletes.
    std::unordered_set<std::int64_t> live = put_keys;
    for (auto k : del_keys) live.erase(k);
    EXPECT_EQ(live, (std::unordered_set<std::int64_t>{1, 3}));
}

TEST(Storage, UnknownWalOpRejected) {
    WriteAheadLog wal((make_test_dir("badop") / "wal.log").string());
    EXPECT_THROW(wal.append("frobnicate", 1, std::nullopt), std::runtime_error);
}

// ---------------------------------------------- special chars in payload
TEST(Storage, PayloadWithSpecialCharsSurvivesRoundtrip) {
    const auto d = make_test_dir("special");
    {
        RecordStore s(d);
        s.put(r(1, {{"q","he said \"hi\""}, {"nl","line1\nline2"}, {"tab","a\tb"}, {"bs","a\\b"}}));
        s.close();
    }
    RecordStore s2(d);
    const Record* got = s2.get(1);
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(got->columns.at("q"), "he said \"hi\"");
    EXPECT_EQ(got->columns.at("nl"), "line1\nline2");
    EXPECT_EQ(got->columns.at("tab"), "a\tb");
    EXPECT_EQ(got->columns.at("bs"), "a\\b");
}
