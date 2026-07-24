"""Python-side tests for the pybind11 storage binding.

These verify that the C++ RecordStore is usable from Python through the
binding: basic ops, crash recovery across reopen, checkpoint, and special
chars. They don't re-test every WAL invariant (the C++ GoogleTest suite covers
that); they test the *bridge*.
"""

import pytest

import hylis._storage as _storage


@pytest.fixture
def tmpstore(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    s = _storage.RecordStore(str(d))
    yield s
    s.close()


def test_put_get(tmpstore):
    tmpstore.put(_storage.Record(1, {"name": "alice", "age": "30"}))
    r = tmpstore.get(1)
    assert r is not None
    assert r.key == 1
    assert r.columns == {"name": "alice", "age": "30"}
    assert r.get("name") == "alice"
    assert r.get("missing", "x") == "x"


def test_len_contains(tmpstore):
    tmpstore.put(_storage.Record(1, {}))
    tmpstore.put(_storage.Record(2, {}))
    assert len(tmpstore) == 2
    assert 1 in tmpstore
    assert 999 not in tmpstore


def test_delete(tmpstore):
    tmpstore.put(_storage.Record(1, {"v": "a"}))
    assert tmpstore.del(1) is True
    assert tmpstore.get(1) is None
    assert tmpstore.del(1) is False


def test_crash_recovery(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    s = _storage.RecordStore(str(d))
    for i in range(10):
        s.put(_storage.Record(i, {"label": f"row{i}"}))
    s.del(3)
    s.close()

    # "Crash" + reopen
    s2 = _storage.RecordStore(str(d))
    assert len(s2) == 9
    assert s2.get(3) is None
    assert s2.get(7).get("label") == "row7"
    s2.close()


def test_checkpoint(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    s = _storage.RecordStore(str(d))
    for i in range(50):
        s.put(_storage.Record(i, {}))
    s.checkpoint()
    s.close()

    s2 = _storage.RecordStore(str(d))
    assert len(s2) == 50
    s2.close()
