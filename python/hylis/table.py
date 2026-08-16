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

Vector columns are declared in the schema and stored *outside* the record::

    schema.add(ColumnDef("image", LogicalType.Vector, dim=128))
    table.create_vector_index("image", VectorPlan(structure=VectorStructure.Graph))
    table.put_vectors("image", keys, embeddings)          # (n, 128) float32

    hits           = table.knn("image", query, k=10)      # record keys
    similar        = table.knn_by_key("image", 42, k=10)  # more-like-this
    hits, trace    = table.hybrid([("price", PredOp.Lt, 5000)], "image", query, k=10)

A 128-float embedding base64s to ~700 bytes per row, so putting it in the
record payload would make the JSON write-ahead log the dominant cost of the
whole system. The floats live in the vector index instead, and reach disk at
``save_vectors()`` -- which ``checkpoint()`` calls. **They are not
write-ahead logged**, so a crash between two saves loses every embedding
attached since the last one while the records themselves survive.
"""

from hylis._planner import PredOp, op_is_indexable  # noqa: F401
from hylis._table import (  # noqa: F401
    ColumnInfo,
    HybridTrace,
    QueryTrace,
    Table,
    VectorInfo,
    VectorMatch,
    VectorPlan,
    VectorStructure,
    WriteResult,
)
