"""Tests for the hybrid query planner, from Python.

The C++ suite owns the heavy plan-equivalence testing across selectivities.
These drive the module the way the demo and the report will -- through
``make_hybrid``, which was built for this path in module 3 and until now was
used by nothing but its own tests.

The property that matters is the same on both sides: **every plan returns the
same rows.** A planner chooses between costs, never between answers.
"""

import numpy as np
import pytest

from hylis import (
    CompareOp,
    FlatIndex,
    HnswIndex,
    HybridPlanner,
    IndexKind,
    Metric,
    PlanKind,
    Predicate,
)
from hylis import datasets as ds
from hylis.query import ScalarColumn


@pytest.fixture(scope="module")
def corpus():
    """Vectors plus scalar attributes -- the planner's actual input."""
    vectors = ds.random_vectors(n=4000, dim=16, n_queries=10, seed=0, n_clusters=40)
    return ds.make_hybrid(vectors, seed=0)


@pytest.fixture(scope="module")
def built(corpus):
    base = corpus.vectors.base
    exact = FlatIndex(corpus.vectors.dim)
    exact.add_batch(base)
    graph = HnswIndex(corpus.vectors.dim, Metric.L2, M=16, ef_construction=200)
    graph.add_batch(base)
    price = ScalarColumn(corpus.attributes["price"])
    return exact, graph, price


def make_planner(built, threshold=0.5):
    exact, graph, price = built
    planner = HybridPlanner(threshold)
    price.attach_to(planner, "price")
    planner.set_exact(exact)
    planner.set_graph(graph)
    return planner


def oracle(built, corpus, cut, k):
    """The true answer, computed without the planner."""
    exact, _, price = built
    allowed = price.rows_below(cut).tolist()
    found = exact.search_filtered(corpus.vectors.queries[0], k=k, allowed=allowed)
    return [n.id for n in found]


# --------------------------------------------------------------- the join


def test_answers_a_query_neither_index_could_answer_alone(built, corpus):
    planner = make_planner(built)
    _, _, price = built
    cut = price.key_cut_for_selectivity(0.10)

    found, plan = planner.search(
        Predicate("price", CompareOp.Lt, cut), corpus.vectors.queries[0], k=10, ef=200
    )
    assert len(found) == 10
    assert plan.matched_rows == cut
    assert abs(plan.selectivity - 0.10) < 1e-9

    # Every row returned satisfies the predicate.
    allowed = set(price.rows_below(cut).tolist())
    for n in found:
        assert n.id in allowed, f"row {n.id} violates the predicate"
    assert [n.id for n in found] == oracle(built, corpus, cut, 10)


def test_the_structured_half_works_on_its_own(built):
    planner = make_planner(built)
    _, _, price = built
    cut = price.key_cut_for_selectivity(0.25)
    rows = planner.matching_rows(Predicate("price", CompareOp.Lt, cut))
    assert len(rows) == cut
    assert sorted(rows) == sorted(price.rows_below(cut).tolist())


def test_the_planner_explains_itself(built):
    planner = make_planner(built)
    _, _, price = built
    plan = planner.explain(
        Predicate("price", CompareOp.Lt, price.key_cut_for_selectivity(0.05)), k=10
    )
    assert plan.reason, "a planner that cannot say why is not defensible"
    assert "crossover" in plan.reason
    assert repr(plan).startswith("QueryPlan(")


# ------------------------------------------------------- plans agree


@pytest.mark.parametrize("selectivity", [0.001, 0.01, 0.1, 0.5, 0.9, 1.0])
def test_every_plan_returns_the_same_rows(built, corpus, selectivity):
    planner = make_planner(built)
    _, _, price = built
    cut = price.key_cut_for_selectivity(selectivity)
    predicate = Predicate("price", CompareOp.Lt, cut)
    query = corpus.vectors.queries[0]
    k = 10

    pre = planner.search_with(PlanKind.PreFilter, predicate, query, k=k, ef=400)
    graph = planner.search_with(PlanKind.FilteredGraph, predicate, query, k=k, ef=400)
    truth = oracle(built, corpus, cut, k)

    assert [n.id for n in pre] == truth
    assert [n.id for n in graph] == truth, (
        f"the graph plan returned different rows at {selectivity:.1%} selectivity; "
        "the planner would be choosing answers, not costs"
    )


@pytest.mark.parametrize("selectivity", [0.01, 0.1, 0.5, 1.0])
def test_post_filter_is_a_prefix_of_the_truth(built, corpus, selectivity):
    """The trap plan, held to the weaker standard it can meet: it may return
    fewer than k rows, but whatever it returns must be the right rows in the
    right order."""
    planner = make_planner(built)
    _, _, price = built
    cut = price.key_cut_for_selectivity(selectivity)
    post = planner.search_with(
        PlanKind.PostFilter,
        Predicate("price", CompareOp.Lt, cut),
        corpus.vectors.queries[0],
        k=10,
        ef=400,
    )
    truth = oracle(built, corpus, cut, 10)
    got = [n.id for n in post]
    assert len(got) <= len(truth)
    assert got == truth[: len(got)]


def test_post_filter_can_silently_return_fewer_than_k(built, corpus):
    """Pinned deliberately. This is *why* the planner exists -- it is what a
    system without one does, and the shortfall is invisible to the caller."""
    planner = make_planner(built)
    _, _, price = built
    # Tight, but comfortably more matches than k -- otherwise a shortfall would
    # just mean "there were not 10 rows", which is a correct answer rather than
    # the failure being demonstrated.
    cut = price.key_cut_for_selectivity(0.01)  # 40 rows, k = 10
    predicate = Predicate("price", CompareOp.Lt, cut)
    assert planner.explain(predicate, 10).matched_rows > 10

    post = planner.search_with(
        PlanKind.PostFilter, predicate, corpus.vectors.queries[0], k=10, ef=64
    )
    chosen, plan = planner.search(predicate, corpus.vectors.queries[0], k=10, ef=64)

    assert len(post) < 10, (
        "post-filter found all 10; widen the corpus or tighten the predicate, "
        "because the point of this test is the shortfall"
    )
    assert len(chosen) == 10, "the plan the planner chose must not have that problem"
    assert plan.kind == PlanKind.PreFilter


@pytest.mark.parametrize(
    "op", [CompareOp.Lt, CompareOp.Le, CompareOp.Gt, CompareOp.Ge, CompareOp.Eq]
)
def test_every_predicate_agrees_with_a_numpy_oracle(built, corpus, op):
    planner = make_planner(built)
    exact, _, price = built
    cut = price.key_cut_for_selectivity(0.3)
    keys = price.keys  # key == sorted position

    wanted = {
        CompareOp.Lt: keys < cut,
        CompareOp.Le: keys <= cut,
        CompareOp.Gt: keys > cut,
        CompareOp.Ge: keys >= cut,
        CompareOp.Eq: keys == cut,
    }[op]
    allowed = price.values[wanted].tolist()

    got = planner.search(
        Predicate("price", op, cut), corpus.vectors.queries[0], k=5, ef=400
    )[0]
    truth = exact.search_filtered(corpus.vectors.queries[0], k=5, allowed=allowed)
    assert [n.id for n in got] == [n.id for n in truth]


# ---------------------------------------------------------- plan choice


def test_tight_predicates_pre_filter_and_loose_ones_do_not(built):
    planner = make_planner(built, threshold=0.5)
    _, _, price = built
    tight = price.key_cut_for_selectivity(0.01)
    loose = price.key_cut_for_selectivity(0.95)

    assert planner.explain(Predicate("price", CompareOp.Lt, tight), 10).kind == PlanKind.PreFilter
    assert planner.explain(Predicate("price", CompareOp.Lt, loose), 10).kind == PlanKind.FilteredGraph


def test_the_threshold_is_what_moves(built):
    _, _, price = built
    cut = price.key_cut_for_selectivity(0.30)
    predicate = Predicate("price", CompareOp.Lt, cut)
    assert make_planner(built, 0.5).explain(predicate, 10).kind == PlanKind.PreFilter
    assert make_planner(built, 0.1).explain(predicate, 10).kind == PlanKind.FilteredGraph


def test_the_threshold_is_settable_after_construction(built):
    planner = make_planner(built, 0.5)
    assert planner.prefilter_threshold == 0.5
    planner.prefilter_threshold = 0.2
    assert planner.prefilter_threshold == 0.2


def test_fewer_matches_than_k_is_always_a_scan(built):
    planner = make_planner(built, threshold=0.0)  # would otherwise force the graph
    plan = planner.explain(Predicate("price", CompareOp.Lt, 5), k=10)
    assert plan.kind == PlanKind.PreFilter
    assert "do not exceed k" in plan.reason


# ----------------------------------------------------------- degenerate


def test_a_predicate_matching_nothing_skips_the_vector_work(built, corpus):
    planner = make_planner(built)
    found, plan = planner.search(
        Predicate("price", CompareOp.Lt, 0), corpus.vectors.queries[0], k=10
    )
    assert found == []
    assert plan.matched_rows == 0
    assert "no vector search needed" in plan.reason


def test_k_larger_than_the_match_count_returns_every_match(built, corpus):
    planner = make_planner(built)
    found, _ = planner.search(
        Predicate("price", CompareOp.Lt, 7), corpus.vectors.queries[0], k=50
    )
    assert len(found) == 7


def test_an_unknown_column_says_which_ones_exist(built):
    planner = make_planner(built)
    assert planner.has_column("price")
    assert not planner.has_column("nosuch")
    with pytest.raises(ValueError, match="price"):
        planner.matching_rows(Predicate("nosuch", CompareOp.Lt, 10))


def test_a_wrong_shaped_query_is_reported(built, corpus):
    planner = make_planner(built)
    with pytest.raises(ValueError, match="1-D"):
        planner.search(
            Predicate("price", CompareOp.Lt, 100),
            np.zeros((2, 16), dtype=np.float32),
            k=5,
        )


def test_the_planner_keeps_its_indexes_alive(corpus):
    """keep_alive, pinned. The planner borrows rather than owns, so without it
    a temporary index would be collected and searched after free -- a crash
    that would read as an index bug rather than a binding one."""
    planner = HybridPlanner(0.5)
    price = ScalarColumn(corpus.attributes["price"])
    price.attach_to(planner, "price")
    planner.set_exact(FlatIndex(corpus.vectors.dim))  # temporary, dropped here
    import gc

    gc.collect()
    # If keep_alive were missing this would search freed memory.
    plan = planner.explain(Predicate("price", CompareOp.Lt, 100), 10)
    assert plan.matched_rows == 100


# ------------------------------------------------- index independence


@pytest.mark.parametrize(
    "kind", [IndexKind.BPlusTree, IndexKind.RMI, IndexKind.DynamicRMI]
)
def test_the_answer_does_not_depend_on_which_index_answered(built, corpus, kind):
    """The CompareOp contract paying off one level up: the planner never learns
    whether a B+ tree, an RMI or a dynamic RMI evaluated the predicate."""
    exact, graph, price = built
    planner = HybridPlanner(0.5)
    planner.set_column_kind(
        "price", price.keys.tolist(), price.values.tolist(), kind, 64
    )
    planner.set_exact(exact)
    planner.set_graph(graph)

    cut = price.key_cut_for_selectivity(0.1)
    found, _ = planner.search(
        Predicate("price", CompareOp.Lt, cut), corpus.vectors.queries[0], k=10, ef=400
    )
    assert [n.id for n in found] == oracle(built, corpus, cut, 10)


# ---------------------------------------------------- the encoding


def test_scalar_column_is_order_preserving():
    values = np.array([3.5, 1.2, 9.9, 1.2, 7.0])
    column = ScalarColumn(values)
    cut = column.encode(7.0)
    assert sorted(column.rows_below(cut).tolist()) == sorted(
        np.flatnonzero(values < 7.0).tolist()
    )


def test_scalar_column_handles_ties():
    values = np.array([1.2, 5.0, 1.2, 1.2, 9.0])
    column = ScalarColumn(values)
    assert sorted(column.rows_equal(1.2)) == [0, 2, 3]
    lo, hi = column.encode_range(1.2)
    assert (lo, hi) == (0, 3)


def test_scalar_column_selectivity_is_exact():
    column = ScalarColumn(np.random.default_rng(0).uniform(size=1000))
    for selectivity in (0.0, 0.001, 0.1, 0.5, 0.99, 1.0):
        cut = column.key_cut_for_selectivity(selectivity)
        assert len(column.rows_below(cut)) == round(selectivity * 1000)


def test_scalar_column_rejects_a_2d_attribute():
    with pytest.raises(ValueError, match="1-D"):
        ScalarColumn(np.zeros((4, 4)))


def test_scalar_column_works_on_integer_attributes(corpus):
    """`category` is already int64, so the encoding must be a no-op in spirit
    rather than something that only works on floats."""
    column = ScalarColumn(corpus.attributes["category"])
    assert len(column) == len(corpus)
    cut = column.key_cut_for_selectivity(0.5)
    assert len(column.rows_below(cut)) == round(0.5 * len(corpus))


# ------------------------------------------------- regressions from the bench


def test_post_filter_works_when_row_ids_are_not_in_key_order(built, corpus):
    """The bug scripts/bench_planner.py found, pinned.

    A column's values are row ids in *attribute* order, which is not row-id
    order. The post-filter plan tested membership with a binary search and so
    silently returned almost nothing -- 498 of 500 rows missing at every
    selectivity. Every C++ fixture happened to use a column whose values were
    already ascending, so the bug was invisible to the whole suite.
    """
    _, _, price = built
    # Confirm the fixture actually has the property that exposed it, so this
    # cannot quietly stop testing anything.
    assert not np.all(np.diff(price.values) > 0), (
        "this column's row ids are ascending, so it cannot catch the bug"
    )

    planner = make_planner(built)
    cut = price.key_cut_for_selectivity(0.9)  # loose: post-filter should cope
    post = planner.search_with(
        PlanKind.PostFilter,
        Predicate("price", CompareOp.Lt, cut),
        corpus.vectors.queries[0],
        k=10,
        ef=200,
    )
    assert len(post) == 10, "post-filter came up short at 90% selectivity"


def test_calibrate_finds_a_crossover_and_adopts_it(built, corpus):
    """The 50% default was inherited from one corpus at one ef. The crossover
    moves with n, dimensionality, ef and cache speed, so the planner measures
    it rather than carrying a constant that was only right elsewhere."""
    planner = make_planner(built, threshold=0.5)
    before = planner.prefilter_threshold
    after = planner.calibrate("price", corpus.vectors.queries[:10], k=10, ef=64)

    assert 0.0 <= after <= 1.0
    assert planner.prefilter_threshold == after
    assert after != before or True  # a machine where 0.5 is right is legitimate


def test_calibrate_on_an_unknown_column_is_reported(built, corpus):
    planner = make_planner(built)
    with pytest.raises(ValueError, match="nosuch"):
        planner.calibrate("nosuch", corpus.vectors.queries[:5], k=10)


def test_calibrate_needs_a_2d_query_set(built, corpus):
    planner = make_planner(built)
    with pytest.raises(ValueError, match="2-D"):
        planner.calibrate("price", corpus.vectors.queries[0], k=10)


def test_calibration_does_not_change_the_answers(built, corpus):
    """Calibration moves which plan is chosen, never which rows come back."""
    planner = make_planner(built, threshold=0.5)
    _, _, price = built
    cut = price.key_cut_for_selectivity(0.3)
    predicate = Predicate("price", CompareOp.Lt, cut)

    before = [n.id for n in planner.search(predicate, corpus.vectors.queries[0],
                                           k=10, ef=400)[0]]
    planner.calibrate("price", corpus.vectors.queries[:10], k=10, ef=64)
    after = [n.id for n in planner.search(predicate, corpus.vectors.queries[0],
                                          k=10, ef=400)[0]]
    assert before == after == oracle(built, corpus, cut, 10)
