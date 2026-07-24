"""hylis -- Hybrid Learned Index System.

C++ core (storage + B+ tree) compiled as pybind11 extensions, exposed here
as the user-facing Python API. The ML layers (learned index, neural router,
planner) live in pure Python and call into this C++ core in-process.

Architecture:
  hylis._storage  (C++ pybind11) -> WAL-durable record store
  hylis._btree    (C++ pybind11) -> B+ tree index
  hylis.storage   (this file)     -> thin Python convenience wrappers
  hylis.index     (this file)     -> thin Python convenience wrappers
"""

from hylis.storage import Record, RecordStore  # noqa: F401

__version__ = "0.1.0"
