"""The typed column layer: logical types, column definitions, and schemas.

``RecordStore`` holds ``dict[str, str]``. Every index below it is keyed on
something narrower, and which structures can serve a column is decided by what
the column *is* rather than by what happens to be fastest:

* ``BPlusTree`` touches keys only through ``lower_bound`` and ``==``, so it
  serves every ordered type natively. A ``String`` column is indexed as a
  string and is never encoded to an integer.
* ``RMIndex`` fits models to ``float(key)`` -- a learned index approximates a
  CDF, and a CDF needs a metric on the key space. A model over string *ranks*
  would be fitted to an ordering the model itself imposed, which measures
  nothing. So ``type_supports_rmi(LogicalType.String)`` is False, and that is
  a fact about learned indexes rather than a limitation here.

The schema also enforces types at ingest, which is the practical difference
between a schema and a convention: ``price = "abc"`` and a misspelled column
name are both refused at ``put`` with the column named, rather than showing up
later as an index quietly missing rows.
"""

from hylis._rmi import (  # noqa: F401
    ColumnShape,
    KeyEncoding,
    LogicalType,
    candidates_for,
    measure_shape,
    type_supports_rmi,
)
from hylis._schema import (  # noqa: F401
    ColumnDef,
    Schema,
    format_value,
    parse_value,
    prefix_upper_bound,
)
