"""Python-side tests for the pybind11 B+ tree binding.

These test the *bridge*: that the C++ tree is usable from Python and that
types cross the boundary correctly. The invariant-level coverage (splits,
borrows, merges, the differential fuzz test against std::map) lives in the
C++ GoogleTest suite, which is the right place for it.
"""

import pytest

from hylis import BPlusTree, CompareOp


@pytest.fixture
def tree():
    t = BPlusTree(order=4)
    for k in range(1, 11):
        t.insert(k, k * 10)
    return t


def test_insert_and_find(tree):
    assert tree.find(3) == 30
    assert tree.find(999) is None
    assert len(tree) == 10


def test_insert_reports_new_vs_overwrite():
    t = BPlusTree(order=4)
    assert t.insert(1, 100) is True
    assert t.insert(1, 200) is False
    assert t.find(1) == 200
    assert len(t) == 1


def test_contains(tree):
    assert tree.contains(5)
    assert 5 in tree
    assert 999 not in tree


def test_erase(tree):
    assert tree.erase(5) is True
    assert tree.find(5) is None
    assert tree.erase(5) is False
    assert len(tree) == 9


def test_keys_and_items_are_sorted(tree):
    assert tree.keys() == list(range(1, 11))
    assert tree.items() == [(k, k * 10) for k in range(1, 11)]


def test_range(tree):
    assert tree.range(3, 6) == [30, 40, 50, 60]
    assert tree.range(0, 100) == [k * 10 for k in range(1, 11)]
    assert tree.range(50, 60) == []


def test_range_query_operators(tree):
    assert tree.range_query(CompareOp.Eq, 4) == [40]
    assert tree.range_query(CompareOp.Lt, 4) == [10, 20, 30]
    assert tree.range_query(CompareOp.Le, 4) == [10, 20, 30, 40]
    assert tree.range_query(CompareOp.Gt, 8) == [90, 100]
    assert tree.range_query(CompareOp.Ge, 8) == [80, 90, 100]


def test_order_below_three_rejected():
    with pytest.raises(ValueError):
        BPlusTree(order=2)


def test_clear_and_validate(tree):
    tree.validate()
    tree.clear()
    assert len(tree) == 0
    assert tree.height() == 1
    tree.validate()


def test_matches_dict_over_many_operations():
    """Light differential check that the binding preserves semantics."""
    t = BPlusTree(order=8)
    ref = {}
    for k in range(0, 500, 3):
        t.insert(k, k * 2)
        ref[k] = k * 2
    for k in range(0, 500, 9):
        if t.erase(k):
            del ref[k]

    t.validate()
    assert len(t) == len(ref)
    assert t.items() == sorted(ref.items())
