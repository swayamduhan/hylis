// tests/test_btree.cpp
//
// Unit tests for the B+ tree.
//
// Two complementary layers:
//   * Hand-built cases at order 3, small enough that each split, borrow,
//     merge and root collapse can be forced deliberately and reasoned about
//     by hand.
//   * A randomized differential test against std::map: the same operation
//     sequence is applied to both and the results compared, with validate()
//     run throughout. This is what actually gives confidence that the
//     rebalancing logic is correct in cases nobody thought to write down.

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include "index/btree.hpp"

using hylis::index::BPlusTree;
using hylis::index::CompareOp;

namespace {

// Order 3 is the smallest legal order, so nodes overflow after 2 keys and
// underflow below 1. Structural events happen almost immediately, which is
// exactly what we want when testing them individually.
constexpr std::size_t kTiny = 3;

std::vector<std::int64_t> keys_of(const BPlusTree<>& t) { return t.keys(); }

} // namespace


// ------------------------------------------------------------ basic usage
TEST(BTree, EmptyTree) {
    BPlusTree<> t;
    EXPECT_EQ(t.size(), 0u);
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.height(), 1u);
    EXPECT_EQ(t.find(42), nullptr);
    EXPECT_FALSE(t.contains(42));
    EXPECT_FALSE(t.erase(42));
    EXPECT_TRUE(t.keys().empty());
    t.validate();
}

TEST(BTree, RejectsOrderBelowThree) {
    EXPECT_THROW(BPlusTree<>(2), std::invalid_argument);
    EXPECT_THROW(BPlusTree<>(0), std::invalid_argument);
    EXPECT_NO_THROW(BPlusTree<>(3));
}

TEST(BTree, InsertAndFind) {
    BPlusTree<> t(kTiny);
    EXPECT_TRUE(t.insert(10, 100));
    EXPECT_TRUE(t.insert(20, 200));
    EXPECT_TRUE(t.insert(5, 50));

    ASSERT_NE(t.find(10), nullptr);
    EXPECT_EQ(*t.find(10), 100);
    EXPECT_EQ(*t.find(20), 200);
    EXPECT_EQ(*t.find(5), 50);
    EXPECT_EQ(t.find(999), nullptr);
    EXPECT_EQ(t.size(), 3u);
    t.validate();
}

TEST(BTree, InsertExistingKeyOverwritesAndReportsFalse) {
    BPlusTree<> t(kTiny);
    EXPECT_TRUE(t.insert(1, 10));
    EXPECT_FALSE(t.insert(1, 99));    // overwrite, not a new key
    EXPECT_EQ(*t.find(1), 99);
    EXPECT_EQ(t.size(), 1u);
    t.validate();
}

TEST(BTree, KeysComeBackSortedRegardlessOfInsertOrder) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k : {50, 10, 90, 30, 70, 20, 80, 40, 60, 100}) t.insert(k, k * 2);
    EXPECT_EQ(keys_of(t), (std::vector<std::int64_t>{10,20,30,40,50,60,70,80,90,100}));
    t.validate();
}

TEST(BTree, ItemsPairsKeysWithValues) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 1; k <= 5; ++k) t.insert(k, k * 11);
    const auto items = t.items();
    ASSERT_EQ(items.size(), 5u);
    for (std::size_t i = 0; i < items.size(); ++i) {
        EXPECT_EQ(items[i].first, static_cast<std::int64_t>(i + 1));
        EXPECT_EQ(items[i].second, static_cast<std::int64_t>((i + 1) * 11));
    }
}

TEST(BTree, ClearResetsToEmpty) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 0; k < 50; ++k) t.insert(k, k);
    t.clear();
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.height(), 1u);
    EXPECT_EQ(t.find(10), nullptr);
    t.validate();
}


// ---------------------------------------------------------------- splits
TEST(BTree, LeafSplitGrowsHeightToTwo) {
    BPlusTree<> t(kTiny);
    t.insert(1, 1);
    t.insert(2, 2);
    EXPECT_EQ(t.height(), 1u);        // still a single leaf root

    t.insert(3, 3);                   // overflows the leaf -> copy-up split
    EXPECT_EQ(t.height(), 2u);
    EXPECT_EQ(keys_of(t), (std::vector<std::int64_t>{1, 2, 3}));
    t.validate();
}

TEST(BTree, InternalSplitGrowsHeightToThree) {
    BPlusTree<> t(kTiny);
    // Enough keys at order 3 to overflow the root's internal node and force
    // a push-up split.
    for (std::int64_t k = 1; k <= 12; ++k) {
        t.insert(k, k);
        t.validate();
    }
    EXPECT_GE(t.height(), 3u);
    EXPECT_EQ(t.size(), 12u);

    std::vector<std::int64_t> expected(12);
    for (std::int64_t i = 0; i < 12; ++i) expected[static_cast<std::size_t>(i)] = i + 1;
    EXPECT_EQ(keys_of(t), expected);
}

TEST(BTree, DescendingInsertsAlsoSplitCorrectly) {
    // Reverse order stresses the "new smallest key" separator-update path.
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 30; k >= 1; --k) {
        t.insert(k, k * 3);
        t.validate();
    }
    ASSERT_EQ(t.size(), 30u);
    for (std::int64_t k = 1; k <= 30; ++k) {
        ASSERT_NE(t.find(k), nullptr) << "missing key " << k;
        EXPECT_EQ(*t.find(k), k * 3);
    }
}

TEST(BTree, ValidateHoldsAcrossManyInsertsAtSeveralOrders) {
    for (std::size_t order : {3u, 4u, 5u, 8u, 32u}) {
        BPlusTree<> t(order);
        for (std::int64_t k = 0; k < 200; ++k) {
            t.insert(k, k);
            t.validate();
        }
        EXPECT_EQ(t.size(), 200u) << "order " << order;
    }
}


// --------------------------------------------------------------- deletes
TEST(BTree, EraseFromSingleLeafRoot) {
    BPlusTree<> t(kTiny);
    t.insert(1, 1);
    t.insert(2, 2);
    EXPECT_TRUE(t.erase(1));
    EXPECT_EQ(t.find(1), nullptr);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_FALSE(t.erase(1));         // already gone
    t.validate();

    EXPECT_TRUE(t.erase(2));
    EXPECT_TRUE(t.empty());
    t.validate();
}

TEST(BTree, EraseMissingKeyIsANoOp) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 0; k < 10; ++k) t.insert(k, k);
    EXPECT_FALSE(t.erase(999));
    EXPECT_EQ(t.size(), 10u);
    t.validate();
}

TEST(BTree, EraseTriggersBorrowFromSibling) {
    // Build a tree, then delete just enough to make one leaf underflow while
    // a sibling still has a key to spare.
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 1; k <= 6; ++k) t.insert(k, k);
    t.validate();

    ASSERT_TRUE(t.erase(1));
    t.validate();
    EXPECT_EQ(keys_of(t), (std::vector<std::int64_t>{2, 3, 4, 5, 6}));
    for (std::int64_t k = 2; k <= 6; ++k) ASSERT_NE(t.find(k), nullptr);
}

TEST(BTree, EraseTriggersMergeAndShrinksHeight) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 1; k <= 5; ++k) t.insert(k, k);
    const std::size_t tall = t.height();
    ASSERT_GE(tall, 2u);

    // Delete almost everything: merges must cascade and collapse the root.
    for (std::int64_t k = 1; k <= 4; ++k) {
        ASSERT_TRUE(t.erase(k)) << "erasing " << k;
        t.validate();
    }
    EXPECT_EQ(t.size(), 1u);
    EXPECT_LT(t.height(), tall);      // root collapsed at least one level
    EXPECT_NE(t.find(5), nullptr);
}

TEST(BTree, EraseEverythingLeavesValidEmptyTree) {
    for (std::size_t order : {3u, 4u, 7u}) {
        BPlusTree<> t(order);
        for (std::int64_t k = 0; k < 100; ++k) t.insert(k, k);
        for (std::int64_t k = 0; k < 100; ++k) {
            ASSERT_TRUE(t.erase(k)) << "order " << order << " key " << k;
            t.validate();
        }
        EXPECT_TRUE(t.empty());
        EXPECT_EQ(t.height(), 1u) << "order " << order;
    }
}

TEST(BTree, EraseInReverseOrder) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 0; k < 60; ++k) t.insert(k, k);
    for (std::int64_t k = 59; k >= 0; --k) {
        ASSERT_TRUE(t.erase(k)) << "key " << k;
        t.validate();
    }
    EXPECT_TRUE(t.empty());
}

TEST(BTree, ReinsertAfterEraseWorks) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 0; k < 40; ++k) t.insert(k, k);
    for (std::int64_t k = 0; k < 40; k += 2) ASSERT_TRUE(t.erase(k));
    t.validate();
    for (std::int64_t k = 0; k < 40; k += 2) ASSERT_TRUE(t.insert(k, k * 10));
    t.validate();

    EXPECT_EQ(t.size(), 40u);
    for (std::int64_t k = 0; k < 40; ++k) {
        ASSERT_NE(t.find(k), nullptr) << "key " << k;
        EXPECT_EQ(*t.find(k), (k % 2 == 0) ? k * 10 : k);
    }
}


// ---------------------------------------------------------- range queries
TEST(BTree, RangeInclusiveBounds) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 0; k < 20; ++k) t.insert(k, k * 100);

    EXPECT_EQ(t.range(5, 8), (std::vector<std::int64_t>{500, 600, 700, 800}));
    EXPECT_EQ(t.range(0, 0), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(t.range(19, 19), (std::vector<std::int64_t>{1900}));
}

TEST(BTree, RangeHandlesEmptyAndOutOfBoundsWindows) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 10; k < 20; ++k) t.insert(k, k);

    EXPECT_TRUE(t.range(8, 5).empty());       // inverted window
    EXPECT_TRUE(t.range(0, 5).empty());       // entirely below
    EXPECT_TRUE(t.range(50, 90).empty());     // entirely above
    EXPECT_EQ(t.range(0, 100).size(), 10u);   // covers everything
}

TEST(BTree, RangeSpansManyLeaves) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 0; k < 500; ++k) t.insert(k, k);
    const auto got = t.range(100, 399);
    ASSERT_EQ(got.size(), 300u);
    EXPECT_EQ(got.front(), 100);
    EXPECT_EQ(got.back(), 399);
}

TEST(BTree, RangeQueryOperators) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k = 1; k <= 5; ++k) t.insert(k, k * 10);

    EXPECT_EQ(t.range_query(CompareOp::Eq, 3), (std::vector<std::int64_t>{30}));
    EXPECT_TRUE(t.range_query(CompareOp::Eq, 99).empty());
    EXPECT_EQ(t.range_query(CompareOp::Lt, 3), (std::vector<std::int64_t>{10, 20}));
    EXPECT_EQ(t.range_query(CompareOp::Le, 3), (std::vector<std::int64_t>{10, 20, 30}));
    EXPECT_EQ(t.range_query(CompareOp::Gt, 3), (std::vector<std::int64_t>{40, 50}));
    EXPECT_EQ(t.range_query(CompareOp::Ge, 3), (std::vector<std::int64_t>{30, 40, 50}));
}

TEST(BTree, RangeQueryOnAbsentPivotStillPartitionsCorrectly) {
    BPlusTree<> t(kTiny);
    for (std::int64_t k : {10, 20, 30, 40}) t.insert(k, k);

    // 25 is not in the tree; the partition should still be clean.
    EXPECT_EQ(t.range_query(CompareOp::Lt, 25), (std::vector<std::int64_t>{10, 20}));
    EXPECT_EQ(t.range_query(CompareOp::Le, 25), (std::vector<std::int64_t>{10, 20}));
    EXPECT_EQ(t.range_query(CompareOp::Gt, 25), (std::vector<std::int64_t>{30, 40}));
    EXPECT_EQ(t.range_query(CompareOp::Ge, 25), (std::vector<std::int64_t>{30, 40}));
}

TEST(BTree, RangeQueryOnEmptyTreeReturnsNothing) {
    BPlusTree<> t(kTiny);
    for (auto op : {CompareOp::Eq, CompareOp::Lt, CompareOp::Le,
                    CompareOp::Gt, CompareOp::Ge}) {
        EXPECT_TRUE(t.range_query(op, 5).empty());
    }
}


// ------------------------------------------------- differential fuzz test
//
// The single strongest correctness check here: drive a BPlusTree and a
// std::map through an identical random operation sequence and require they
// agree at every step, with validate() confirming the tree's internal
// invariants after each mutation.
TEST(BTree, DifferentialFuzzAgainstStdMap) {
    for (unsigned seed : {1u, 7u, 42u, 1337u, 90210u}) {
        for (std::size_t order : {3u, 4u, 6u, 32u}) {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<std::int64_t> key_dist(0, 300);
            std::uniform_int_distribution<int> op_dist(0, 99);

            BPlusTree<> tree(order);
            std::map<std::int64_t, std::int64_t> ref;

            for (int step = 0; step < 3000; ++step) {
                const std::int64_t k = key_dist(rng);
                const int roll = op_dist(rng);

                if (roll < 60) {                      // 60% insert
                    const std::int64_t v = k * 7 + step;
                    const bool tree_new = tree.insert(k, v);
                    const bool ref_new = ref.insert_or_assign(k, v).second;
                    ASSERT_EQ(tree_new, ref_new)
                        << "seed " << seed << " order " << order
                        << " step " << step << " key " << k;
                } else if (roll < 90) {               // 30% erase
                    const bool tree_hit = tree.erase(k);
                    const bool ref_hit = ref.erase(k) != 0;
                    ASSERT_EQ(tree_hit, ref_hit)
                        << "seed " << seed << " order " << order
                        << " step " << step << " key " << k;
                } else {                              // 10% lookup
                    const std::int64_t* got = tree.find(k);
                    const auto it = ref.find(k);
                    if (it == ref.end()) {
                        ASSERT_EQ(got, nullptr) << "step " << step << " key " << k;
                    } else {
                        ASSERT_NE(got, nullptr) << "step " << step << " key " << k;
                        ASSERT_EQ(*got, it->second) << "step " << step << " key " << k;
                    }
                }

                ASSERT_EQ(tree.size(), ref.size())
                    << "seed " << seed << " order " << order << " step " << step;
                ASSERT_NO_THROW(tree.validate())
                    << "seed " << seed << " order " << order << " step " << step;
            }

            // Full contents must match, in order.
            std::vector<std::pair<std::int64_t, std::int64_t>> expected(ref.begin(), ref.end());
            ASSERT_EQ(tree.items(), expected)
                << "seed " << seed << " order " << order;
        }
    }
}

// Ranges are the query shape the planner actually depends on, so fuzz them
// against std::map too rather than trusting the hand-written cases alone.
TEST(BTree, DifferentialFuzzRangeQueries) {
    std::mt19937 rng(2026);
    std::uniform_int_distribution<std::int64_t> key_dist(0, 500);

    BPlusTree<> tree(6);
    std::map<std::int64_t, std::int64_t> ref;
    for (int i = 0; i < 400; ++i) {
        const std::int64_t k = key_dist(rng);
        tree.insert(k, k * 2);
        ref[k] = k * 2;
    }
    tree.validate();

    for (int trial = 0; trial < 300; ++trial) {
        std::int64_t lo = key_dist(rng);
        std::int64_t hi = key_dist(rng);
        if (hi < lo) std::swap(lo, hi);

        std::vector<std::int64_t> expected;
        for (auto it = ref.lower_bound(lo); it != ref.end() && it->first <= hi; ++it) {
            expected.push_back(it->second);
        }
        ASSERT_EQ(tree.range(lo, hi), expected)
            << "trial " << trial << " window [" << lo << ", " << hi << "]";

        // And the uniform predicate interface, against the same reference.
        const std::int64_t pivot = key_dist(rng);
        std::vector<std::int64_t> lt, ge;
        for (const auto& [k, v] : ref) {
            if (k < pivot) lt.push_back(v);
            else ge.push_back(v);
        }
        ASSERT_EQ(tree.range_query(CompareOp::Lt, pivot), lt) << "pivot " << pivot;
        ASSERT_EQ(tree.range_query(CompareOp::Ge, pivot), ge) << "pivot " << pivot;
    }
}
