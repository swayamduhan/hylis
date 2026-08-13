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

from hylis.index import (  # noqa: F401
    BPlusTree,
    ColumnIndex,
    CompareOp,
    DynamicConfig,
    DynamicRMIndex,
    DynamicStats,
    IndexCatalog,
    IndexKind,
    IndexPlan,
    RMIndex,
    choose_index,
    measure_plan,
)
from hylis.query import (  # noqa: F401
    HybridPlanner,
    PlanKind,
    Predicate,
    QueryPlan,
)
from hylis.schema import (  # noqa: F401
    ColumnDef,
    ColumnShape,
    KeyEncoding,
    LogicalType,
    Schema,
    candidates_for,
    format_value,
    measure_shape,
    parse_value,
    prefix_upper_bound,
    type_supports_rmi,
)
from hylis.storage import Record, RecordStore  # noqa: F401
from hylis.table import (  # noqa: F401
    ColumnInfo,
    PredOp,
    QueryTrace,
    Table,
    WriteResult,
    op_is_indexable,
)
from hylis.vector import (  # noqa: F401
    FlatIndex,
    HnswIndex,
    HnswStats,
    Metric,
    Neighbor,
    RouterHealth,
)

__version__ = "0.1.0"
