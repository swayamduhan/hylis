"""The table layer: records in a store, indexes over their columns.

This is the join that had been missing. ``RecordStore`` was complete, durable
and tested, and no index had ever read from it -- every index in the project
was built by handing ``ColumnIndex`` two lists that came from a test fixture or
a benchmark script. So there was no path from a stored record to an index over
it, and no ``SELECT ... WHERE`` that returned rows.

A ``Table`` owns nothing but the connection::

    store  = RecordStore("data/shop")
    schema = Schema([
        ColumnDef("price",    LogicalType.Int64),
        ColumnDef("category", LogicalType.String),
        ColumnDef("title",    LogicalType.String),
    ])
    table = Table(store, schema)

    table.put_batch(rows)
    table.create_index("category", write_fraction=0.1)

    keys, trace = table.select_keys("category", PredOp.Eq, "shoes")
    rows, trace = table.select("title", PredOp.Prefix, "nike")

Every query returns ``(result, trace)``. The trace says whether an index
answered it and how many rows were scanned if not -- so a predicate no index
can serve (``Contains``, ``IsNull``) is answered correctly *and* visibly,
rather than being silently slow or silently refused.
"""

from hylis._table import (  # noqa: F401
    ColumnInfo,
    PredOp,
    QueryTrace,
    Table,
    WriteResult,
    op_is_indexable,
)
