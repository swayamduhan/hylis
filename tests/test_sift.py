"""Tests against the real SIFT10K corpus.

Skipped unless the data has been downloaded:

    python scripts/fetch_data.py siftsmall

These are the tests that cannot be faked. Everywhere else the ground truth is
computed by hylis' own code, so a systematic error in it would go unnoticed --
the oracle and the thing under test would be wrong together. SIFT ships
neighbour lists computed by its authors in 2010, years before this project
existed, so agreeing with them is real external evidence.
"""

import numpy as np
import pytest

from hylis import datasets as ds

pytestmark = pytest.mark.sift


@pytest.fixture(scope="module")
def siftsmall():
    try:
        return ds.load_sift("siftsmall")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_shapes_match_the_published_description(siftsmall):
    assert siftsmall.n == 10_000
    assert siftsmall.dim == 128
    assert siftsmall.n_queries == 100
    assert siftsmall.base.dtype == np.float32
    assert siftsmall.ground_truth is not None
    assert siftsmall.ground_truth.shape == (100, 100)


def test_our_oracle_reproduces_the_published_ground_truth(siftsmall):
    """The load-bearing test of this whole module.

    If compute_ground_truth agrees with TEXMEX's own neighbour lists on all
    10k x 100 pairs, then every recall number later measured with it is
    trustworthy. If it disagreed, every benchmark in the project would be
    quietly wrong.

    Compared as sets per query: SIFT vectors are quantised, so exact distance
    ties exist and the tie-break order is arbitrary.
    """
    computed = ds.compute_ground_truth(
        siftsmall.base, siftsmall.queries, k=100, metric="euclidean"
    )
    published = siftsmall.ground_truth

    assert ds.recall_at_k(computed, published, k=100) == 1.0


def test_top_1_neighbour_agrees_exactly(siftsmall):
    """Ties are vanishingly unlikely at rank 1, so this can be exact."""
    computed = ds.compute_ground_truth(siftsmall.base, siftsmall.queries, k=1)
    assert np.array_equal(computed[:, 0], siftsmall.ground_truth[:, 0])


def test_recall_degrades_as_expected_on_a_truncated_result(siftsmall):
    """Sanity-check the recall metric itself against a known-bad result set.

    Replacing half of each row with invalid ids must give exactly 0.5; if the
    metric cannot detect deliberately broken output it cannot be trusted to
    grade HNSW either.
    """
    truth = siftsmall.ground_truth[:, :10]
    broken = truth.copy()
    broken[:, 5:] = -1
    assert ds.recall_at_k(broken, truth, k=10) == pytest.approx(0.5)


def test_hybrid_attributes_over_real_vectors(siftsmall):
    """The planner's actual input: real SIFT vectors, synthetic predicates."""
    h = ds.make_hybrid(siftsmall, seed=0)
    assert len(h) == 10_000

    price = h.attributes["price"]
    for target in (0.001, 0.01, 0.1, 0.5):
        t = ds.threshold_for_selectivity(price, target)
        assert h.selectivity(price < t) == pytest.approx(target, abs=1e-3)
