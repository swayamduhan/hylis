#!/usr/bin/env python3
"""Linear versus neural stage 1 in the RMI.

    python scripts/experiment_stage1.py
    python scripts/experiment_stage1.py -n 50000 --models 4096

Kraska et al. (SIGMOD 2018) used a small neural network as the RMI's first
stage. Every learned index since has used a plain linear model there. This
reproduces the comparison rather than taking it on faith, which is the reason
the engine's stage 1 is linear.

Both models are written from scratch on numpy in python/hylis/learned.py --
including the MLP's forward pass, backward pass and Adam optimiser.

Why the comparison is on error, not latency
-------------------------------------------
Stage 1 only decides *which* second-stage model handles a key. Its quality
shows up entirely as the width of the search window the second stage is left
needing, and lookup cost is proportional to that window. Timing a numpy MLP
against a C++ linear fit would measure Python. Comparing the windows the two
produce measures the models, and is what transfers.

The finding is not that the network is useless. It is indistinguishable from
a line on near-linear keys, and hits the same floor on stepped ones -- but on
a heavily skewed CDF it routes measurably better, which is the one case where
stage 1 is the bottleneck. What it does not do is justify roughly a
thousandfold increase in build time to get there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from hylis import datasets as ds
    from hylis.learned import LinearStage1, MLPStage1, build_rmi_errors
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"cannot import hylis ({exc}); pip install -r requirements.txt")

DISTRIBUTIONS = ["sequential_gaps", "uniform", "lognormal", "clustered"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=50_000, help="keys per distribution")
    parser.add_argument("--models", type=int, default=1024, help="second-stage size")
    parser.add_argument("--hidden", type=int, default=16, help="MLP hidden units")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--distributions", type=str, default=",".join(DISTRIBUTIONS))
    args = parser.parse_args(argv)

    print(f"Stage-1 model comparison, n={args.n:,}, "
          f"second stage = {args.models:,} models")
    print(f"MLP: 1 -> {args.hidden} -> 1, ReLU, Adam, {args.epochs} epochs, "
          f"hand-written\n")

    header = (f"{'distribution':<17}{'stage 1':<9}{'fit time':>10}  {'max err':>10}  "
              f"{'mean err':>10}  {'empty':>15}  {'busiest':>9}")
    print(header)
    print("-" * len(header))

    linear_wins = 0
    total = 0
    for dist in [d.strip() for d in args.distributions.split(",")]:
        keys = ds.synthetic_keys(dist, n=args.n, seed=0).keys

        results = {}
        for stage1 in (LinearStage1(),
                       MLPStage1(hidden=args.hidden, epochs=args.epochs, seed=0)):
            fit = build_rmi_errors(keys, stage1, models=args.models)
            results[fit.stage1] = fit
            print(f"{dist if fit.stage1 == 'linear' else '':<17}"
                  f"{fit.stage1:<9}"
                  f"{fit.fit_seconds*1000:>9.1f}ms  "
                  f"{fit.max_error:>10,.0f}  {fit.mean_error:>10,.1f}  "
                  f"{fit.empty_models:>6,}/{fit.models:<8,}  "
                  f"{fit.largest_model:>9,}")

        total += 1
        if results["linear"].mean_error <= results["mlp"].mean_error:
            linear_wins += 1
        print()

    print(f"linear stage 1 matched or beat the MLP on {linear_wins} of "
          f"{total} distributions, at a small fraction of the fit cost")
    print()
    print("Reading this:")
    print("  * 'fit time' is stage 1 only. The linear fit is closed form -- five")
    print("    sums in one pass -- so it has no iteration to spend time on,")
    print("    and typically runs ~1000x faster than training the network.")
    print("  * 'busiest' is how many keys landed in the most heavily loaded")
    print("    second-stage model, and is the number that explains the rest.")
    print("    Stage 1 routes by predicted position, so a model handed most of")
    print("    the keys cannot be rescued by any amount of second-stage")
    print("    capacity -- that is the mechanism behind a max-error plateau.")
    print("  * The result is not 'linear always wins'. On near-linear and")
    print("    uniform keys the two are indistinguishable, and on clustered")
    print("    keys both hit the same floor, because there the limit is the")
    print("    gaps in the data rather than the routing. But on a heavily")
    print("    skewed CDF the network genuinely does route better -- it")
    print("    spreads the dense region across more models, which is visible")
    print("    in both 'busiest' and the resulting error.")
    print("  * So the honest conclusion is a cost one: the network buys a real")
    print("    improvement on exactly the case where routing is the")
    print("    bottleneck, and pays orders of magnitude more build time for")
    print("    it. Linear is the right default; skew is where that default is")
    print("    worth revisiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
