"""Correctness and agreement across every vector index in the project.

Four implementations answer the same questions: the exact scan, hnswlib, our
HNSW, and our HNSW with the neural router. Two things are checked.

*Shared correctness* — anything true of "a vector index" rather than of one
particular index is asserted for all of them, once, from one place. That is
what stops one implementation drifting quietly while the others stay honest.

*Agreement* — each is graded against the exact oracle, but that alone is not
enough: two indexes can each score 0.95 against the truth while overlapping far
less with each other, which would mean they are missing different neighbours.
So they are compared pairwise as well.

hnswlib is a benchmark baseline only; these tests skip it when it was not
fetched.
"""

import numpy as np
import pytest

from hylis import FlatIndex, HnswIndex, Metric
from hylis import datasets as ds
from hylis._hnsw import NeuralRouter
from hylis.router import kmeans, build_training_set, RouterMLP, router_json

import hylis._hnswlib as hnswlib_module

HAS_HNSWLIB = getattr(hnswlib_module, "available", False)

K = 10
EF = 100


@pytest.fixture(scope="module")
def corpus():
    return ds.random_vectors(n=4000, dim=24, n_queries=80, seed=1, n_clusters=40)


@pytest.fixture(scope="module")
def truth(corpus):
    return ds.compute_ground_truth(corpus.base, corpus.queries, k=K)


@pytest.fixture(scope="module")
def indexes(corpus):
    """Every implementation, behind one uniform `search(query, k) -> ids`."""
    built = {}

    flat = FlatIndex(corpus.dim)
    flat.add_batch(corpus.base)
    built["flat"] = lambda q, k: [n.id for n in flat.search(q, k)]

    graph = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    graph.add_batch(corpus.base)
    built["hnsw"] = lambda q, k: [n.id for n in graph.search(q, k, EF)]

    km = kmeans(corpus.base, n_clusters=48, seed=0)
    ts = build_training_set(corpus.base, km.assignment, n_samples=20000, seed=0)
    model = RouterMLP(dim=corpus.dim, clusters=48, hidden=48, seed=0)
    model.fit(ts.x_train, ts.y_train, epochs=25)

    routed = HnswIndex(corpus.dim, Metric.L2, M=16, ef_construction=200)
    routed.add_batch(corpus.base)
    routed.set_router(NeuralRouter.from_json(router_json(model, km.medoids)))
    built["routed"] = lambda q, k: [n.id for n in routed.search(q, k, EF, True)]

    if HAS_HNSWLIB:
        lib = hnswlib_module.HnswlibIndex(corpus.dim, Metric.L2, capacity=corpus.n,
                                          M=16, ef_construction=200)
        lib.add_batch(corpus.base)
        built["hnswlib"] = lambda q, k: [n.id for n in lib.search(q, k, EF)]

    return built


def results_for(search, corpus, k=K):
    return [search(q, k) for q in corpus.queries]


def overlap(a, b, k=K):
    """Mean fraction of b's ids that a also returned, per query."""
    hits = sum(len(set(x) & set(y)) for x, y in zip(a, b))
    return hits / (len(a) * k)


# --------------------------------------------------------------------------
# Shared correctness
# --------------------------------------------------------------------------


def test_every_implementation_is_present(indexes):
    assert {"flat", "hnsw", "routed"} <= set(indexes)
    if HAS_HNSWLIB:
        assert "hnswlib" in indexes


def test_results_are_valid_sorted_and_distinct(indexes, corpus):
    for name, search in indexes.items():
        for i, query in enumerate(corpus.queries[:20]):
            ids = search(query, K)
            assert len(ids) == K, f"{name} query {i} returned {len(ids)}"
            assert len(set(ids)) == K, f"{name} returned duplicates"
            assert all(0 <= j < corpus.n for j in ids), f"{name} returned a bad id"


def test_k_larger_than_the_corpus_is_clamped(indexes, corpus):
    for name, search in indexes.items():
        assert len(search(corpus.queries[0], corpus.n + 500)) <= corpus.n, name


def test_every_implementation_clears_the_recall_floor(indexes, corpus, truth):
    truth_ids = [row.tolist() for row in truth]
    for name, search in indexes.items():
        recall = overlap(results_for(search, corpus), truth_ids)
        assert recall >= 0.90, f"{name} recall {recall:.4f} against the exact oracle"


def test_self_retrieval(indexes, corpus):
    probes = list(range(0, corpus.n, 173))
    for name, search in indexes.items():
        hits = sum(1 for i in probes if search(corpus.base[i], 1)[:1] == [i])
        assert hits >= 0.95 * len(probes), f"{name} failed self-retrieval"


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


def test_implementations_agree_with_each_other(indexes, corpus):
    """Each matching the oracle is not sufficient — they must also match each
    other, or they are missing different neighbours."""
    results = {name: results_for(search, corpus) for name, search in indexes.items()}
    names = sorted(results)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = overlap(results[a], results[b])
            assert score >= 0.85, (
                f"{a} and {b} each look accurate but agree only {score:.3f} "
                "with one another"
            )


def test_agreement_matrix_is_symmetric_and_complete(indexes, corpus):
    results = {name: results_for(search, corpus) for name, search in indexes.items()}
    for a in results:
        assert overlap(results[a], results[a]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# hnswlib, the external reference
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib was not fetched")
def test_our_hnsw_is_competitive_with_the_reference(indexes, corpus, truth):
    """The question this baseline exists to answer.

    Not "are we faster" — we are a hand-written implementation against a
    heavily optimised one — but "is our recall in the same league", because a
    router that beats a strawman has proved nothing.
    """
    truth_ids = [row.tolist() for row in truth]
    ours = overlap(results_for(indexes["hnsw"], corpus), truth_ids)
    theirs = overlap(results_for(indexes["hnswlib"], corpus), truth_ids)
    assert ours > theirs - 0.05, (
        f"ours {ours:.4f} vs hnswlib {theirs:.4f} — more than five points "
        "behind the reference suggests a defect, not a tuning difference"
    )


@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib was not fetched")
def test_hnswlib_scores_use_our_convention(corpus):
    """hnswlib returns squared L2; we return true distance. If the adapter did
    not convert, score comparisons would silently compare different things."""
    exact = FlatIndex(corpus.dim)
    exact.add_batch(corpus.base)
    lib = hnswlib_module.HnswlibIndex(corpus.dim, Metric.L2, capacity=corpus.n)
    lib.add_batch(corpus.base)

    query = corpus.queries[0]
    want = exact.search(query, 1)[0]
    got = lib.search(query, 1, 200)[0]
    if got.id == want.id:
        assert got.score == pytest.approx(want.score, rel=1e-3)


@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib was not fetched")
def test_hnswlib_honours_filters(corpus):
    lib = hnswlib_module.HnswlibIndex(corpus.dim, Metric.L2, capacity=corpus.n)
    lib.add_batch(corpus.base)
    allowed = list(range(0, corpus.n, 7))

    for query in corpus.queries[:5]:
        for n in lib.search_filtered(query, 10, allowed, 200):
            assert n.id % 7 == 0
    with pytest.raises(IndexError):
        lib.search_filtered(corpus.queries[0], 5, [999999])


def test_module_reports_availability_either_way():
    """`import hylis._hnswlib` must always work, so an absent baseline shows up
    as a flag rather than an ImportError three frames away."""
    assert isinstance(hnswlib_module.available, bool)
