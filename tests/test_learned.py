"""Tests for the stage-1 model families used in the RMI ablation.

The MLP is written from scratch, so its backward pass is checked against
finite differences rather than trusted. A hand-derived gradient that is
quietly wrong still trains -- just badly -- which would make the whole
linear-vs-neural comparison meaningless.
"""

import numpy as np
import pytest

from hylis import datasets as ds
from hylis.learned import (
    LinearStage1,
    MLPStage1,
    build_rmi_errors,
    normalise_keys,
)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_normalise_maps_onto_the_unit_interval():
    keys = np.array([100, 200, 500, 900], dtype=np.int64)
    x = normalise_keys(keys)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x) > 0)


def test_normalise_handles_degenerate_inputs():
    assert normalise_keys(np.array([], dtype=np.int64)).size == 0
    assert np.all(normalise_keys(np.array([7, 7, 7], dtype=np.int64)) == 0.0)


def test_normalise_survives_huge_keys():
    keys = np.array([2**40, 2**41, 2**42], dtype=np.int64)
    x = normalise_keys(keys)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# LinearStage1
# --------------------------------------------------------------------------


def test_linear_matches_numpy_polyfit():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, size=500))
    y = 3.5 * x - 2.0 + rng.normal(0, 0.01, size=500)

    model = LinearStage1().fit(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    assert model.slope == pytest.approx(slope, rel=1e-9)
    assert model.intercept == pytest.approx(intercept, rel=1e-9)


def test_linear_fits_an_exact_line_exactly():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    model = LinearStage1().fit(x, 2.0 * x + 1.0)
    assert np.allclose(model.predict(x), 2.0 * x + 1.0)


def test_linear_handles_degenerate_inputs():
    assert LinearStage1().fit(np.array([]), np.array([])).slope == 0.0

    single = LinearStage1().fit(np.array([5.0]), np.array([9.0]))
    assert single.predict(np.array([123.0]))[0] == pytest.approx(9.0)

    flat = LinearStage1().fit(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert flat.slope == 0.0
    assert flat.predict(np.array([2.0]))[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# MLPStage1 -- correctness of the hand-written training
# --------------------------------------------------------------------------


def test_mlp_backward_matches_finite_differences():
    """The load-bearing test for the from-scratch network.

    A wrong analytic gradient still produces a network that trains, just
    poorly -- so the linear-vs-neural comparison would be measuring a bug
    rather than a model family.
    """
    rng = np.random.default_rng(0)
    net = MLPStage1(hidden=5, seed=1)
    x = rng.normal(size=(12, 1))
    target = rng.normal(size=(12, 1))
    n = float(x.shape[0])

    def loss():
        out, _ = net._forward(x)
        return float(np.mean((out - target) ** 2))

    out, cache = net._forward(x)
    grads = dict(zip(["w1", "b1", "w2", "b2"],
                     net._backward(cache, 2.0 * (out - target) / n)))

    eps = 1e-6
    for name in ["w1", "b1", "w2", "b2"]:
        param = getattr(net, name)
        analytic = grads[name].reshape(param.shape)
        flat = param.ravel()
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            up = loss()
            flat[i] = original - eps
            down = loss()
            flat[i] = original
            numeric = (up - down) / (2 * eps)
            assert numeric == pytest.approx(analytic.ravel()[i], abs=1e-6), (
                f"{name}[{i}]: analytic {analytic.ravel()[i]} vs numeric {numeric}"
            )


def test_mlp_reduces_loss_on_a_fittable_target():
    x = np.linspace(0, 1, 400)
    y = x ** 2  # curved, and well within a small ReLU net's reach

    net = MLPStage1(hidden=16, epochs=400, lr=0.05, seed=0).fit(x, y)
    assert net.losses[-1] < net.losses[0] / 10, "training should actually train"
    assert net.losses[-1] < 1e-3


def test_mlp_approximates_a_curve_better_than_a_straight_line():
    """The premise of the whole ablation: a network *can* fit curvature that a
    single line cannot. Whether that is worth its cost is the experiment."""
    x = np.linspace(0, 1, 500)
    y = x ** 3

    linear_error = np.abs(LinearStage1().fit(x, y).predict(x) - y).max()
    mlp_error = np.abs(MLPStage1(hidden=16, epochs=600, seed=0).fit(x, y).predict(x) - y).max()
    assert mlp_error < linear_error


def test_mlp_is_deterministic_for_a_seed():
    x = np.linspace(0, 1, 200)
    y = np.sqrt(x)
    a = MLPStage1(epochs=50, seed=3).fit(x, y).predict(x)
    b = MLPStage1(epochs=50, seed=3).fit(x, y).predict(x)
    c = MLPStage1(epochs=50, seed=4).fit(x, y).predict(x)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mlp_rejects_a_zero_width_hidden_layer():
    with pytest.raises(ValueError):
        MLPStage1(hidden=0)


def test_mlp_handles_empty_input():
    net = MLPStage1(epochs=5).fit(np.array([]), np.array([]))
    assert net.predict(np.array([])).size == 0


# --------------------------------------------------------------------------
# The RMI harness
# --------------------------------------------------------------------------


def test_harness_reproduces_the_engine_shape():
    keys = ds.synthetic_keys("uniform", n=5000, seed=0).keys
    fit = build_rmi_errors(keys, LinearStage1(), models=256)

    assert fit.stage1 == "linear"
    assert fit.models == 256
    assert fit.n_keys == 5000
    assert fit.max_error >= fit.mean_error
    assert 0 <= fit.empty_models <= 256
    assert fit.largest_model >= 1
    assert fit.fit_seconds >= 0.0


def test_harness_error_falls_with_more_models_on_a_curve():
    keys = ds.synthetic_keys("lognormal", n=20000, seed=0).keys
    coarse = build_rmi_errors(keys, LinearStage1(), models=16)
    fine = build_rmi_errors(keys, LinearStage1(), models=4096)
    assert fine.mean_error < coarse.mean_error


def test_harness_agrees_with_the_cpp_engine_on_the_easy_case():
    """A sanity cross-check that the numpy harness models the same structure
    the C++ index does, so conclusions drawn here transfer."""
    from hylis import RMIndex

    keys = ds.synthetic_keys("sequential_gaps", n=20000, seed=0).keys
    fit = build_rmi_errors(keys, LinearStage1(), models=1024)

    engine = RMIndex(models=1024)
    engine.build(keys.tolist(), list(range(len(keys))))
    stats = engine.stats()

    # Not identical -- the harness fits on normalised keys and the engine on
    # offsets from the first key -- but they must land in the same league.
    assert fit.mean_error == pytest.approx(stats.mean_error, abs=2.0)


def test_harness_rejects_zero_models():
    with pytest.raises(ValueError):
        build_rmi_errors(np.array([1, 2, 3]), LinearStage1(), models=0)


def test_harness_handles_an_empty_column():
    fit = build_rmi_errors(np.array([], dtype=np.int64), LinearStage1(), models=8)
    assert fit.n_keys == 0
    assert fit.max_error == 0.0


def test_both_stage1_families_run_end_to_end():
    keys = ds.synthetic_keys("lognormal", n=5000, seed=0).keys
    for stage1 in (LinearStage1(), MLPStage1(hidden=8, epochs=50, seed=0)):
        fit = build_rmi_errors(keys, stage1, models=256)
        assert fit.n_keys == 5000
        assert np.isfinite(fit.max_error)
        assert np.isfinite(fit.mean_error)
