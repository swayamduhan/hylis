"""The hybrid query planner: one query, two indexes.

A hybrid query is a structured predicate and a vector similarity search
together::

    SELECT id FROM t
    WHERE  category < 40
    ORDER BY distance(embedding, $q)
    LIMIT  10

Everything needed to serve it existed before this module. ``ColumnIndex``
answers a predicate with row ids; ``FlatIndex`` and ``HnswIndex`` both take row
ids as a filter. Nothing joined them, so every caller built its filter by hand.
``HybridPlanner`` is that join, plus the decision of how to execute it.

The decision rule is measured rather than assumed -- a filtered exact scan is
``O(|allowed|)`` and gets cheaper as a predicate tightens, while a filtered
graph search must step through non-matching nodes and gets more expensive.
They cross at ~50% selectivity, and below it the scan wins by up to 50x.

Every plan returns the same rows: the planner chooses between *costs*, never
between *answers*.
"""

# ``Predicate`` is here too, not in ``hylis._table``. ``query/predicate.hpp``
# is the query layer's shared type -- the planner and the table used to define
# one each, which meant they could never appear in the same program, and a demo
# of the whole system has to do exactly that.
#
# It is bound by ``_planner`` rather than ``_table`` because pybind11 allows one
# owner per C++ type and ``table.hpp`` includes ``planner.hpp``: downstream
# imports upstream. The other direction makes the two extension modules a cycle.
from hylis._planner import (  # noqa: F401
    HybridPlanner,
    PlanKind,
    PredOp,
    Predicate,
    QueryPlan,
    op_is_indexable,
)


import numpy as np


class ScalarColumn:
    """A scalar attribute, encoded so an int64 index can answer predicates on it.

    ``ColumnIndex`` is keyed on int64 and requires **strictly ascending unique**
    keys, but real attributes are floats (``price``, ``timestamp``) as often as
    integers, and they repeat. Rather than cast and lose ordering, this stores
    an **order-preserving encoding**: each row's key is its position in the
    attribute's sorted order, and a threshold is translated into that space
    before the predicate runs.

    Order-preserving encoding is what a column store does for exactly this
    reason, so it is the standard move rather than a workaround. It is exact:
    ``value < t`` holds for precisely the rows whose ``key < encode(t)``,
    because sorted position is monotone in value and the sort is stable, so
    every row below the threshold sorts before every row above it.

    **The one thing it cannot express directly is ``Eq`` on a repeated value.**
    Equality becomes a contiguous *range* of keys rather than one key, so it
    needs ``encode_range`` and two comparisons. Single-predicate queries are
    all this planner takes, so that case is served by ``rows_equal`` instead.
    A multi-predicate planner would express it as a conjunction; that is out of
    scope here, and this is the honest limitation of the encoding rather than a
    bug in it.
    """

    def __init__(self, values: np.ndarray):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"expected a 1-D attribute, got shape {values.shape}")
        self._values = values
        # Rows in ascending attribute order. Stable, so ties keep row order and
        # the encoding is reproducible rather than dependent on sort internals.
        self._order = np.argsort(values, kind="stable").astype(np.int64)
        self._sorted = values[self._order]

        # key = position in sorted order (unique, ascending by construction);
        # value = the row id that position belongs to. So a predicate over this
        # column returns row ids, which is exactly what the vector index takes
        # as its filter.
        self.keys = np.arange(len(self._order), dtype=np.int64)
        self.values = self._order

    def __len__(self) -> int:
        return int(self._values.shape[0])

    @property
    def sorted_values(self) -> np.ndarray:
        return self._sorted

    def encode(self, threshold) -> int:
        """The key a ``< threshold`` predicate becomes.

        Equals the number of rows whose value is strictly below the threshold.
        """
        return int(np.searchsorted(self._sorted, threshold, side="left"))

    def encode_range(self, value) -> tuple[int, int]:
        """``[lo, hi)`` in key space covering every row equal to ``value``."""
        lo = int(np.searchsorted(self._sorted, value, side="left"))
        hi = int(np.searchsorted(self._sorted, value, side="right"))
        return lo, hi

    def rows_equal(self, value) -> list[int]:
        """Row ids whose attribute equals ``value``, for the Eq case above."""
        lo, hi = self.encode_range(value)
        return self._order[lo:hi].tolist()

    def key_cut_for_selectivity(self, selectivity: float) -> int:
        """The key cut whose ``<`` predicate keeps ``selectivity`` of the rows.

        Exact by construction: keys are sorted positions, so keeping the first
        m of them keeps exactly m rows. That exactness is what makes the
        plan-choice tests statements about the planner rather than about how
        the attribute happened to be distributed.
        """
        if not 0.0 <= selectivity <= 1.0:
            raise ValueError(f"selectivity must be in [0, 1], got {selectivity}")
        return int(round(selectivity * len(self)))

    def rows_below(self, cut: int) -> np.ndarray:
        """Row ids a ``< cut`` predicate selects, computed without the planner."""
        return self._order[:max(0, min(cut, len(self)))]

    def attach_to(self, planner, name: str) -> None:
        """Register this column with a planner."""
        planner.set_column(name, self.keys.tolist(), self.values.tolist())
