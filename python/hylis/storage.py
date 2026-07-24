"""Python-side convenience wrappers for the C++ storage engine.

The C++ extension is importable as ``hylis._storage`` and provides:
  - Record:    key (int) + columns (dict[str,str])
  - RecordStore: put/get/del/checkpoint/close with WAL durability

This module re-exports them directly so users write ``from hylis import Record``
rather than ``from hylis._storage import Record``. If later we want Python-only
convenience methods (e.g. context managers, dict-like sugar), they go here.
"""

from hylis._storage import Record, RecordStore  # noqa: F401
