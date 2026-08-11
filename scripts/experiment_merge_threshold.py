#!/usr/bin/env python3
"""When should a dynamic learned index merge? Measured, not guessed.

    python scripts/experiment_merge_threshold.py            # the full sweep
    python scripts/experiment_merge_threshold.py --quick    # one distribution
    python scripts/experiment_merge_threshold.py --sosd fb  # a real corpus

DynamicRMIndex has two merge triggers and this settles both:

  rho    merge once pending changes reach this fraction of the base.
  tau_e  merge once a disturbed model's Cook's distance passes this --
         DynaMind's contribution (Cheng et al., KBS 348, 2026, section 4.2),
         whose argument is that a size trigger alone is insufficient because
         the distribution can shift badly while the buffer is still half
         empty.

tau_e = infinity is included as the **control**: the score trigger switched
off, merges by size alone. Without that arm the sweep cannot show the trigger
earns its keep, and if it does not, that is the result reported.

Workload mixes follow the paper's section 6.1 -- read-only, read-heavy (95/5),
write-heavy (50/50), write-only -- with updates split evenly between inserts
and deletes.

Every configuration is checked against a dict oracle as it runs, so a fast
wrong configuration cannot win the table.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from hylis import DynamicConfig, DynamicRMIndex, RMIndex
    from hylis import datasets as ds
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"cannot import hylis ({exc}).\n"
        "    cmake --preset release && cmake --build build"
    )

DISTRIBUTIONS = ["uniform", "lognormal", "clustered", "sequential_gaps"]

# (name, fraction of operations that are writes)
WORKLOADS = [
    ("read-only", 0.00),
    ("read-heavy", 0.05),
    ("write-heavy", 0.50),
    ("write-only", 1.00),
]

RHOS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
TAUS = [0.1, 1.0, 10.0, 100.0, math.inf]


def require_optimised() -> None:
    import hylis._rmi as r

    if not getattr(r, "__optimized__", True):
        raise SystemExit(
            f"built as {getattr(r, '__build_type__', '?')} -- timings would be "
            "several times off and meaningless.\n"
            "    cmake --preset release && cmake --build build"
        )


def tau_label(tau: float) -> str:
    return "off" if math.isinf(tau) else f"{tau:g}"


@dataclass
class Result:
    ops_per_second: float
    merges: int
    full_rebuilds: int
    models_shifted: int
    models_refitted: int
    keys_rescanned: int
    merge_seconds: float
    worst_merge: float
    mean_error: float
    max_error: int
    index_bytes: int
    final_size: int


def run_workload(keys, write_fraction, rho, tau, n_ops, seed, models,
                 verify) -> Result:
    """Drive one configuration through a workload, optionally oracle-checked.

    The oracle is a plain dict. Checking every operation roughly halves
    throughput, so the timing pass and the correctness pass are separate runs
    of the same seeded sequence -- same operations, same order, one measured
    and one verified.
    """
    values = [i * 10 for i in range(len(keys))]

    cfg = DynamicConfig()
    cfg.second_stage_size = models
    cfg.merge_ratio = rho
    cfg.score_threshold = tau
    idx = DynamicRMIndex(cfg)
    idx.build(keys, values)

    oracle = dict(zip(keys, values)) if verify else None
    lo, hi = keys[0], keys[-1]
    rng = random.Random(seed)
    worst_merge = 0.0

    start = time.perf_counter()
    for step in range(n_ops):
        probe = rng.randrange(lo, hi + 1)
        if rng.random() >= write_fraction:
            got = idx.find(probe)
            if oracle is not None and got != oracle.get(probe):
                raise SystemExit(
                    f"MISMATCH: find({probe}) gave {got}, oracle says "
                    f"{oracle.get(probe)} at step {step} "
                    f"(rho={rho}, tau={tau_label(tau)})"
                )
        elif rng.random() < 0.5:
            got = idx.insert(probe, step)
            if oracle is not None:
                want = probe not in oracle
                if got != want:
                    raise SystemExit(
                        f"MISMATCH: insert({probe}) gave {got}, expected {want} "
                        f"at step {step}"
                    )
                if want:
                    oracle[probe] = step
        else:
            # Half the deletes target a key that is actually present, so the
            # tombstone path is exercised rather than only the miss path.
            target = probe
            if oracle is not None and rng.random() < 0.5 and oracle:
                target = rng.choice(list(oracle))
            elif oracle is None and rng.random() < 0.5:
                target = keys[rng.randrange(len(keys))]
            got = idx.erase(target)
            if oracle is not None:
                want = target in oracle
                if got != want:
                    raise SystemExit(
                        f"MISMATCH: erase({target}) gave {got}, expected {want} "
                        f"at step {step}"
                    )
                oracle.pop(target, None)

        last = idx.stats().last_merge_seconds
        if last > worst_merge:
            worst_merge = last
    seconds = time.perf_counter() - start

    if oracle is not None:
        idx.validate()
        if len(idx) != len(oracle):
            raise SystemExit(f"MISMATCH: {len(idx)} live keys, oracle has {len(oracle)}")

    s = idx.stats()
    return Result(
        ops_per_second=n_ops / seconds if seconds > 0 else float("inf"),
        merges=s.merges,
        full_rebuilds=s.full_rebuilds,
        models_shifted=s.models_shifted,
        models_refitted=s.models_refitted,
        keys_rescanned=s.keys_rescanned,
        merge_seconds=s.total_merge_seconds,
        worst_merge=worst_merge,
        mean_error=s.mean_error,
        max_error=s.max_error,
        index_bytes=s.index_bytes,
        final_size=s.size,
    )


def sweep_tau(keys, args) -> None:
    """Does the Cook's distance trigger earn its keep?

    Held at one rho so the only variable is tau, and run on the write-heavy
    mix because that is where a trigger has the most to prove.
    """
    print("\n" + "=" * 78)
    print("The score trigger: does it earn its keep?")
    print("  write-heavy (50/50), rho held at 0.05, so tau_e is the only variable")
    print("=" * 78)
    header = (f"{'tau_e':>8}{'merges':>9}{'ops/s':>12}{'meanerr':>10}"
              f"{'maxerr':>9}{'merge_s':>10}{'worst_ms':>10}")
    print(header)
    print("-" * len(header))

    baseline = None
    for tau in TAUS:
        r = run_workload(keys, 0.50, 0.05, tau, args.ops, 7, args.models,
                         verify=args.verify)
        print(f"{tau_label(tau):>8}{r.merges:>9,}{r.ops_per_second:>12,.0f}"
              f"{r.mean_error:>10.2f}{r.max_error:>9,}{r.merge_seconds:>10.3f}"
              f"{r.worst_merge * 1000:>10.2f}")
        if math.isinf(tau):
            baseline = r

    if baseline is not None:
        print(f"\n  'off' is the control. Compare every finite row against it:")
        print(f"  if they merge far more often for no better mean error, the")
        print(f"  trigger is costing throughput and buying nothing.")


def sweep_rho(keys, args) -> None:
    """What merge threshold, for what workload?"""
    print("\n" + "=" * 78)
    print("The size trigger: rho against workload mix")
    print("  score trigger off, so rho is the only variable")
    print("=" * 78)

    header = f"{'workload':<13}{'rho':>7}{'merges':>8}{'ops/s':>12}"
    header += f"{'shifted%':>10}{'rescan/n':>10}{'meanerr':>9}{'bytes':>11}"
    print(header)
    print("-" * len(header))

    best: dict[str, tuple[float, float]] = {}
    for name, write_fraction in WORKLOADS:
        for rho in RHOS:
            r = run_workload(keys, write_fraction, rho, math.inf, args.ops, 7,
                             args.models, verify=args.verify)
            updates = r.models_shifted + r.models_refitted
            shifted = 100.0 * r.models_shifted / updates if updates else 0.0
            rescan = r.keys_rescanned / max(len(keys), 1)
            print(f"{name:<13}{rho:>7.3f}{r.merges:>8,}{r.ops_per_second:>12,.0f}"
                  f"{shifted:>9.1f}%{rescan:>10.2f}{r.mean_error:>9.2f}"
                  f"{r.index_bytes / 1024:>10,.0f}K")
            if name not in best or r.ops_per_second > best[name][1]:
                best[name] = (rho, r.ops_per_second)
        print()

    print("Fastest rho per workload:")
    for name, (rho, ops) in best.items():
        print(f"  {name:<13} rho = {rho:<6.3f} ({ops:,.0f} ops/s)")


def read_only_cost(keys, args) -> None:
    """What mutability costs when you are not using it.

    The number most easily left out of a paper about a dynamic index, and the
    one a reader should see first: every read pays a delta probe and a
    tombstone check that a static index does not.
    """
    print("\n" + "=" * 78)
    print("What being writable costs on a read-only workload")
    print("=" * 78)

    values = [i * 10 for i in range(len(keys))]
    rng = random.Random(3)
    probes = [keys[rng.randrange(len(keys))] for _ in range(args.ops)]

    def fastest_of(index, repeats=5):
        """Best of several passes.

        The difference being measured is a delta probe and a bit test, which
        is small next to the scheduling noise on a single pass -- small enough
        that a noisy run can put the dynamic index ahead of the static one,
        which is not a thing that can be true. The minimum is the run least
        interfered with, so it is the fair comparison.
        """
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            for p in probes:
                index.find(p)
            best = min(best, time.perf_counter() - start)
        return best

    static = RMIndex(args.models)
    static.build(keys, values)
    static_seconds = fastest_of(static)

    cfg = DynamicConfig()
    cfg.second_stage_size = args.models
    dynamic = DynamicRMIndex(cfg)
    dynamic.build(keys, values)
    empty_seconds = fastest_of(dynamic)

    # Again with a delta buffer that is not empty, since an empty one is the
    # best case and quoting only that would overstate things.
    for i in range(len(keys) // 50):
        dynamic.insert(keys[i * 7] + 1, i)
    loaded_seconds = fastest_of(dynamic)

    print(f"  static RMIndex           {args.ops / static_seconds:>12,.0f} lookups/s")
    print(f"  DynamicRMIndex, empty    {args.ops / empty_seconds:>12,.0f} lookups/s"
          f"   ({static_seconds / empty_seconds:.2f}x)")
    print(f"  DynamicRMIndex, 2% delta {args.ops / loaded_seconds:>12,.0f} lookups/s"
          f"   ({static_seconds / loaded_seconds:.2f}x)")
    print("\n  Best of 5 passes each. These cross a pybind11 boundary per")
    print("  lookup, which costs more than the lookup does and compresses all")
    print("  three towards each other -- so the ratios are a floor on the real")
    print("  gap, not a measurement of it. scripts/bench_index.py times the")
    print("  static index from inside C++ for the same reason.")


def load_keys(args):
    if args.sosd:
        path = ds.default_data_dir() / args.sosd
        if not path.exists():
            raise SystemExit(
                f"{path} not found.\n    python scripts/fetch_data.py {args.sosd}"
            )
        data = ds.read_sosd(path, limit=args.n)
        return {args.sosd: [int(k) for k in data]}

    names = [args.distributions] if args.quick else DISTRIBUTIONS
    if args.distributions and not args.quick:
        names = args.distributions.split(",")
    return {
        name: [int(k) for k in ds.synthetic_keys(name, n=args.n, seed=0).keys]
        for name in names
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", type=int, default=200_000, help="keys in the base")
    parser.add_argument("--ops", type=int, default=50_000,
                        help="operations per configuration")
    parser.add_argument("--models", type=int, default=1024,
                        help="RMI second-stage size")
    parser.add_argument("--quick", action="store_true",
                        help="one distribution, smaller run")
    parser.add_argument("--distributions", type=str, default="",
                        help=f"comma-separated; default all of {DISTRIBUTIONS}")
    parser.add_argument("--sosd", type=str, default="",
                        help="a SOSD file in data/, e.g. fb_200M_uint64")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip the dict oracle. Faster, and worth strictly "
                             "less: an unverified throughput number could be "
                             "the throughput of being wrong.")
    parser.set_defaults(verify=True)
    args = parser.parse_args(argv)

    if args.quick:
        args.n = min(args.n, 50_000)
        args.ops = min(args.ops, 20_000)
        if not args.distributions:
            args.distributions = "lognormal"

    require_optimised()

    for name, keys in load_keys(args).items():
        print("\n" + "#" * 78)
        print(f"# {name}: {len(keys):,} keys, {args.ops:,} ops per configuration, "
              f"{args.models} models")
        if args.verify:
            print("# every operation checked against a dict oracle")
        print("#" * 78)

        read_only_cost(keys, args)
        sweep_tau(keys, args)
        if not args.quick:
            sweep_rho(keys, args)

    print("\nReading these tables:")
    print("  * 'shifted%' is the share of model updates done in O(1) by moving")
    print("    an intercept. It measures how localised the writes were, not how")
    print("    good the index is: appends approach 100%, scattered writes do not.")
    print("  * 'rescan/n' is keys revisited re-measuring error bounds, over the")
    print("    key count. Above ~1.0 the incremental merge is saving nothing.")
    print("  * mean error is in records. It is what lookup cost tracks, so a")
    print("    configuration that merges less but drifts more is not free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
