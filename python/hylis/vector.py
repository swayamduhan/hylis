"""Python-side wrappers for the C++ vector indexes.

The C++ extension is importable as ``hylis._flat`` and provides:
  - FlatIndex: exhaustive (brute-force) k-nearest-neighbour search
  - Metric: L2 / InnerProduct / Cosine
  - Neighbor: one result, with ``.id`` and ``.score``

``FlatIndex`` is exact, and stays in the final build alongside the
approximate HNSW index rather than being replaced by it. It is the oracle
HNSW's recall is measured against, the baseline its speedups are quoted
against, and -- via ``search_filtered`` -- genuinely the cheaper plan once a
selective predicate has cut the candidate set down. HNSW will expose the same
``search`` / ``search_filtered`` pair so the query planner can choose between
them without caring which is underneath, the same way ``range_query`` is
shared between the B+ tree and the learned index.
"""

from hylis._flat import FlatIndex, Metric, Neighbor  # noqa: F401
from hylis._hnsw import HnswIndex, HnswStats, RouterHealth  # noqa: F401
