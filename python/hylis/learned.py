"""Stage-1 model families for the learned index, and a harness to compare them.

Why this exists
---------------
The C++ RMI uses a linear stage 1, which is what production learned indexes
do. Kraska et al.'s original paper used a small neural network there instead,
and the field moved away from it. This module reproduces that comparison
rather than asserting it, so the choice in the engine is a measured one.

It is deliberately *not* part of the engine. Stage-1 choice is a question
about models, and answering it does not require the answer to live in the
query path.

Why error and not latency
-------------------------
Stage 1 only decides which second-stage model handles a key. Its quality
therefore shows up entirely as the width of the search window the second
stage is left needing, and lookup cost is proportional to that. Timing a
numpy MLP against a C++ linear fit would measure the language; comparing the
windows they produce measures the models.

Everything here is written out by hand on top of numpy -- including the MLP's
forward pass, backward pass and Adam optimiser -- to stay consistent with the
project's from-scratch constraint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LinearStage1",
    "MLPStage1",
    "RMIFit",
    "build_rmi_errors",
    "normalise_keys",
]


def normalise_keys(keys: np.ndarray) -> np.ndarray:
    """Map keys onto [0, 1].

    Every model here consumes normalised keys, which is what keeps the
    closed-form least-squares solution in build_rmi_errors numerically safe:
    on raw int64 keys the sums of squares would reach ~1e31 and lose most of
    their significant digits. It is also what lets the MLP train at all --
    feeding it 1e12-scale inputs would saturate the first layer immediately.
    """
    x = np.asarray(keys, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = float(x[0]), float(x[-1])
    span = hi - lo
    if span <= 0.0:
        return np.zeros_like(x)
    return (x - lo) / span


class LinearStage1:
    """Ordinary least squares. The closed-form baseline.

    There is no iteration and no hyperparameter: slope and intercept fall out
    of five sums in one pass. This is the entire "training" an RMI needs.
    """

    name = "linear"

    def __init__(self) -> None:
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearStage1":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size == 0:
            return self
        if x.size == 1:
            self.slope, self.intercept = 0.0, float(y[0])
            return self

        mean_x, mean_y = x.mean(), y.mean()
        dx = x - mean_x
        sxx = float(dx @ dx)
        if sxx <= 0.0:
            self.slope, self.intercept = 0.0, float(mean_y)
            return self
        self.slope = float(dx @ (y - mean_y) / sxx)
        self.intercept = float(mean_y - self.slope * mean_x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(x, dtype=np.float64) + self.intercept


class MLPStage1:
    """A 1 -> hidden -> 1 network with ReLU, trained by Adam. Written out by hand.

    The point of comparison, not of the engine. A network can in principle
    approximate a curved CDF better than a straight line, which is the claim
    the original paper made; whether that survives contact with build time and
    inference cost is what the experiment measures.
    """

    name = "mlp"

    def __init__(self, hidden: int = 16, epochs: int = 300, lr: float = 0.05,
                 seed: int = 0) -> None:
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._init_parameters()
        # Targets are normalised to [0, 1] during training and scaled back on
        # predict, so the loss surface does not depend on how many keys there
        # happen to be.
        self._y_scale = 1.0
        self.losses: list[float] = []

    def _init_parameters(self) -> None:
        rng = np.random.default_rng(self.seed)
        # He initialisation: variance 2/fan_in, which is what keeps ReLU
        # activations from collapsing toward zero through the layer.
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / 1), size=(1, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden), size=(self.hidden, 1))
        self.b2 = np.zeros(1)

    # -- forward / backward, kept separable so gradients can be checked -----

    def _forward(self, x: np.ndarray):
        """Returns (output, cache). x has shape (n, 1)."""
        pre = x @ self.w1 + self.b1          # (n, hidden)
        act = np.maximum(pre, 0.0)           # ReLU
        out = act @ self.w2 + self.b2        # (n, 1)
        return out, (x, pre, act)

    def _backward(self, cache, dout: np.ndarray):
        """Gradients of the loss w.r.t. every parameter, given dL/dout."""
        x, pre, act = cache
        gw2 = act.T @ dout
        gb2 = dout.sum(axis=0)
        dact = dout @ self.w2.T
        dpre = dact * (pre > 0.0)            # ReLU passes gradient where active
        gw1 = x.T @ dpre
        gb1 = dpre.sum(axis=0)
        return gw1, gb1, gw2, gb2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPStage1":
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if x.size == 0:
            return self

        self._init_parameters()
        self._y_scale = float(y.max()) if float(y.max()) > 0 else 1.0
        target = y / self._y_scale
        n = float(x.shape[0])

        # Adam moments, one pair per parameter tensor.
        params = ["w1", "b1", "w2", "b2"]
        moment1 = {p: np.zeros_like(getattr(self, p)) for p in params}
        moment2 = {p: np.zeros_like(getattr(self, p)) for p in params}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.losses = []
        for step in range(1, self.epochs + 1):
            out, cache = self._forward(x)
            residual = out - target
            self.losses.append(float(np.mean(residual ** 2)))

            # d/dout of mean((out - target)^2)
            dout = 2.0 * residual / n
            grads = dict(zip(params, self._backward(cache, dout)))

            for p in params:
                g = grads[p].reshape(getattr(self, p).shape)
                moment1[p] = beta1 * moment1[p] + (1 - beta1) * g
                moment2[p] = beta2 * moment2[p] + (1 - beta2) * (g * g)
                m_hat = moment1[p] / (1 - beta1 ** step)
                v_hat = moment2[p] / (1 - beta2 ** step)
                setattr(self, p, getattr(self, p) - self.lr * m_hat /
                        (np.sqrt(v_hat) + eps))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        out, _ = self._forward(x)
        return (out * self._y_scale).ravel()


@dataclass(frozen=True)
class RMIFit:
    """What a stage-1 choice cost, and what it bought."""

    stage1: str
    models: int
    n_keys: int
    fit_seconds: float
    max_error: float
    mean_error: float
    empty_models: int
    largest_model: int  # keys routed to the busiest second-stage model


def build_rmi_errors(keys: np.ndarray, stage1, models: int = 1024) -> RMIFit:
    """Build a two-stage RMI with the given stage 1 and report its error.

    Mirrors what cpp/hylis/index/rmi.hpp does, at the level of detail the
    comparison needs: stage 1 routes, each second-stage model is fitted by
    least squares to whatever it was routed, and the residual is the search
    window a lookup would face.

    Second-stage fitting is done with grouped sums via ``np.bincount`` rather
    than a Python loop over models, so ``models`` can reach 65536 without the
    experiment taking minutes. Keys are normalised first, which is what makes
    the uncentred closed form safe here.
    """
    keys = np.asarray(keys)
    n = keys.size
    if n == 0:
        return RMIFit(stage1.name, models, 0, 0.0, 0.0, 0.0, models, 0)
    if models < 1:
        raise ValueError(f"models must be >= 1, got {models}")

    x = normalise_keys(keys)
    y = np.arange(n, dtype=np.float64)

    start = time.perf_counter()
    stage1.fit(x, y)
    predicted = np.asarray(stage1.predict(x), dtype=np.float64)
    fit_seconds = time.perf_counter() - start

    # Route to a second-stage model exactly as the C++ index does.
    scaled = predicted * (models / n)
    bucket = np.clip(np.floor(scaled), 0, models - 1).astype(np.int64)

    counts = np.bincount(bucket, minlength=models).astype(np.float64)
    sum_x = np.bincount(bucket, weights=x, minlength=models)
    sum_y = np.bincount(bucket, weights=y, minlength=models)
    sum_xx = np.bincount(bucket, weights=x * x, minlength=models)
    sum_xy = np.bincount(bucket, weights=x * y, minlength=models)

    # Least squares per group. Guard the degenerate cases -- an empty group,
    # or one whose keys are all identical -- by falling back to predicting the
    # group's mean position, which is what a model carrying no information
    # should do.
    denominator = counts * sum_xx - sum_x * sum_x
    safe = denominator != 0.0
    slope = np.zeros(models)
    np.divide(counts * sum_xy - sum_x * sum_y, denominator,
              out=slope, where=safe)
    intercept = np.zeros(models)
    np.divide(sum_y - slope * sum_x, counts, out=intercept, where=counts > 0)

    final = slope[bucket] * x + intercept[bucket]
    final = np.clip(final, 0.0, n - 1)
    errors = np.abs(final - y)

    return RMIFit(
        stage1=stage1.name,
        models=models,
        n_keys=n,
        fit_seconds=fit_seconds,
        max_error=float(errors.max()),
        mean_error=float(errors.mean()),
        empty_models=int(np.count_nonzero(counts == 0)),
        largest_model=int(counts.max()),
    )
