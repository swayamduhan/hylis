"""Python-side convenience wrappers for the C++ index structures.

The C++ extension is importable as ``hylis._btree`` and provides:
  - BPlusTree: int64 -> int64 ordered index with range scans
  - CompareOp: the predicate enum used by ``range_query``

``CompareOp`` and ``range_query`` are the interface the query planner codes
against. The learned index will expose the same pair, so the planner can use
either without knowing which is underneath.
"""

from hylis._btree import BPlusTree, CompareOp  # noqa: F401
