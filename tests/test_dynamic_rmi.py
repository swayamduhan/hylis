"""Tests for the writable learned index, from Python.

The C++ suite in cpp/tests/test_dynamic_rmi.cpp owns the heavy differential
testing. These cover the binding surface and re-run the load-bearing property
— that mutation never costs exactness — against a Python ``dict``, so a
binding that dropped or reordered results is caught here rather than being
invisible on both sides of the bridge.
"""

import random

import pytest

from hylis import CompareOp, DynamicConfig, DynamicRMIndex, RMIndex
from hylis import datasets as ds

DISTRIBUTIONS = ["sequential_gaps", "uniform", "lognormal", "clustered"]


def make(keys, values=None, **config):
    cfg = DynamicConfig()
    for name, value in config.items():
        setattr(cfg, name, value)
    idx = DynamicRMIndex(cfg)
    keys = [int(k) for k in keys]
    if values is None:
        values = [i * 10 for i in range(len(keys))]
    idx.build(keys, [int(v) for v in values])
    return idx


def keys_for(distribution, n=5000, seed=0):
    return [int(k) for k in ds.synthetic_keys(distribution, n=n, seed=seed).keys]


# ------------------------------------------------------------------ basics


def test_starts_as_the_static_index_did():
    keys = keys_for("sequential_gaps")
    idx = make(keys)
    assert len(idx) == len(keys)
    for i, k in enumerate(keys):
        assert idx.find(k) == i * 10
        assert k in idx
    idx.validate()


def test_empty_index_answers_everything_with_nothing():
    idx = make([])
    assert len(idx) == 0
    assert idx.find(42) is None
    assert idx.range(0, 100) == []
    for op in (CompareOp.Eq, CompareOp.Lt, CompareOp.Le, CompareOp.Gt, CompareOp.Ge):
        assert idx.range_query(op, 42) == []
    assert idx.insert(42, 7)
    assert idx.find(42) == 7


def test_duplicates_are_refused_from_either_side():
    keys = keys_for("sequential_gaps", n=1000)
    idx = make(keys, merge_ratio=10.0)
    assert idx.insert(keys[10], 999) is False, "already in the base"
    assert idx.insert(keys[10] + 1, 999) is True
    assert idx.insert(keys[10] + 1, 111) is False, "already in the delta"


def test_erasing_what_is_absent_reports_so():
    keys = keys_for("sequential_gaps", n=1000)
    idx = make(keys, merge_ratio=10.0)
    assert idx.erase(keys[-1] + 12345) is False
    assert idx.erase(keys[5]) is True
    assert idx.erase(keys[5]) is False, "already tombstoned"


def test_reinserting_a_tombstoned_key_takes_the_new_value():
    keys = keys_for("sequential_gaps", n=2000)
    idx = make(keys, merge_ratio=10.0)
    k = keys[300]
    assert idx.erase(k)
    assert idx.find(k) is None
    assert idx.insert(k, -2)
    assert idx.find(k) == -2, "the delta holds the newer value"
    idx.merge()
    assert idx.find(k) == -2, "and the merge kept it"
    idx.validate()


# ------------------------------------------------------------ the oracle


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_agrees_with_a_dict_through_a_mixed_workload(distribution):
    keys = keys_for(distribution, n=4000, seed=1)
    oracle = {k: i * 10 for i, k in enumerate(keys)}
    idx = make(keys, second_stage_size=128, merge_ratio=0.03)

    rng = random.Random(7)
    lo, hi = keys[0], keys[-1]
    for step in range(6000):
        op = rng.randrange(100)
        probe = rng.randrange(lo, hi + 1)
        if op < 30:
            wanted = probe not in oracle
            got = idx.insert(probe, step)
            assert got == wanted, f"insert {probe} at step {step}"
            if wanted:
                oracle[probe] = step
        elif op < 55:
            target = rng.choice(list(oracle)) if (oracle and rng.random() < 0.5) else probe
            wanted = target in oracle
            assert idx.erase(target) == wanted, f"erase {target} at step {step}"
            oracle.pop(target, None)
        elif op < 90:
            assert idx.find(probe) == oracle.get(probe), f"find {probe} at step {step}"
        else:
            top = probe + rng.randrange(100000)
            wanted = [v for k, v in sorted(oracle.items()) if probe <= k <= top]
            assert idx.range(probe, top) == wanted, f"range at step {step}"

    assert len(idx) == len(oracle)
    idx.validate()
    assert idx.stats().merges > 0, "the workload never merged; it proves little"


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_every_predicate_agrees_with_a_dict(distribution):
    keys = keys_for(distribution, n=2000, seed=2)
    oracle = {k: i * 10 for i, k in enumerate(keys)}
    idx = make(keys, second_stage_size=64, merge_ratio=10.0)

    rng = random.Random(11)
    for i in range(150):
        k = rng.choice(keys)
        if rng.random() < 0.5:
            if oracle.pop(k, None) is not None:
                assert idx.erase(k)
        elif k + 1 not in oracle:
            oracle[k + 1] = -i
            assert idx.insert(k + 1, -i)

    stats = idx.stats()
    assert stats.delta_size > 0 and stats.tombstones > 0, "nothing to interleave"

    ordered = sorted(oracle.items())
    for _ in range(40):
        value = rng.choice(keys) + rng.choice([0, 1])
        checks = {
            CompareOp.Eq: lambda k, v=value: k == v,
            CompareOp.Lt: lambda k, v=value: k < v,
            CompareOp.Le: lambda k, v=value: k <= v,
            CompareOp.Gt: lambda k, v=value: k > v,
            CompareOp.Ge: lambda k, v=value: k >= v,
        }
        for op, predicate in checks.items():
            wanted = [v for k, v in ordered if predicate(k)]
            assert idx.range_query(op, value) == wanted, f"{op} at {value}"


def test_matches_the_static_index_on_the_same_contents():
    """Both are exact, so they must agree key for key — even though the
    dynamic one reused its models across a merge and the static one refitted
    stage 1 from scratch."""
    keys = keys_for("lognormal", n=4000, seed=3)
    oracle = {k: i * 10 for i, k in enumerate(keys)}
    idx = make(keys, second_stage_size=128, merge_ratio=10.0)

    rng = random.Random(13)
    for i in range(400):
        k = rng.choice(keys) + 1
        if k not in oracle:
            oracle[k] = -i
            idx.insert(k, -i)
    for _ in range(200):
        k = rng.choice(keys)
        if oracle.pop(k, None) is not None:
            idx.erase(k)
    idx.merge()

    ordered = sorted(oracle.items())
    static = RMIndex(128)
    static.build([k for k, _ in ordered], [v for _, v in ordered])

    for k, v in ordered:
        assert idx.find(k) == static.find(k) == v
    idx.validate()
    static.validate()


# ------------------------------------------------------------- machinery


def test_a_localised_batch_leaves_most_models_merely_shifted():
    keys = keys_for("sequential_gaps", n=20000, seed=4)
    idx = make(keys, second_stage_size=256, merge_ratio=10.0)
    for i in range(300):
        idx.insert(keys[10000] + i * 2 + 1, i)
    idx.merge()

    stats = idx.stats()
    assert stats.models_shifted > stats.models_refitted * 10
    assert stats.keys_rescanned < len(keys) // 4
    idx.validate()


def test_a_scattered_batch_does_not_get_that_saving():
    """The honest other half. Quoting the saving above without this one would
    be quoting a best case as if it were the rule."""
    keys = keys_for("sequential_gaps", n=20000, seed=5)
    idx = make(keys, second_stage_size=256, merge_ratio=10.0)
    rng = random.Random(17)
    for i in range(600):
        idx.insert(rng.choice(keys) + 1, i)
    idx.merge()

    stats = idx.stats()
    assert stats.models_refitted > stats.models_shifted
    idx.validate()


def test_merging_reclaims_tombstones():
    keys = keys_for("sequential_gaps", n=5000, seed=6)
    idx = make(keys, merge_ratio=10.0)
    for i in range(0, 500):
        idx.erase(keys[i * 7])

    assert idx.stats().tombstones == 500
    assert idx.stats().base_size == len(keys), "not compacted yet"

    idx.merge()
    assert idx.stats().tombstones == 0
    assert idx.stats().base_size == len(keys) - 500
    assert len(idx) == len(keys) - 500
    idx.validate()


def test_the_size_trigger_fires_on_its_own():
    keys = keys_for("sequential_gaps", n=5000, seed=7)
    idx = make(keys, merge_ratio=0.01)
    assert idx.stats().merges == 0
    for i in range(100):
        idx.insert(keys[i * 11] + 1, i)
    assert idx.stats().merges > 0
    assert idx.stats().delta_size < 100


def test_the_score_starts_at_zero_and_moves_only_on_deletion():
    keys = keys_for("uniform", n=8000, seed=8)
    idx = make(keys, second_stage_size=128, merge_ratio=10.0)
    assert idx.score() == 0.0

    for i in range(50):
        idx.insert(keys[i * 13] + 1, i)
    assert idx.score() == 0.0, (
        "inserts land in the delta, so the base models have not moved at all "
        "— this is exactly why the score trigger buys nothing here"
    )

    idx.erase(keys[1000])
    assert idx.score() > 0.0


def test_the_score_trigger_can_be_switched_on():
    keys = keys_for("uniform", n=8000, seed=9)
    idx = make(keys, second_stage_size=128, merge_ratio=10.0,
               score_threshold=1.0, score_check_interval=1)
    for i in range(200):
        idx.erase(keys[i * 20])
    assert idx.stats().merges > 0, "the score trigger never fired"
    idx.validate()


def test_the_default_leaves_the_score_trigger_off():
    """Pinned deliberately. It is a measured decision, not an oversight, and
    a silent change back to the paper's 1.0 would cost ~20x throughput for no
    accuracy — see the header comment on Config::score_threshold."""
    cfg = DynamicConfig()
    assert cfg.score_threshold == float("inf")


def test_growing_far_past_the_original_range_stays_exact():
    keys = keys_for("sequential_gaps", n=3000, seed=10)
    idx = make(keys, second_stage_size=64, merge_ratio=0.05)
    k = keys[-1]
    for i in range(3000):
        k += 1000
        assert idx.insert(k, i)
    idx.merge()

    assert len(idx) == len(keys) + 3000
    idx.validate()


def test_stats_report_what_maintenance_cost():
    keys = keys_for("uniform", n=5000, seed=12)
    idx = make(keys, merge_ratio=0.02)
    for i in range(500):
        idx.insert(keys[i * 9] + 1, i)

    stats = idx.stats()
    assert stats.merges > 0
    assert stats.total_merge_seconds >= 0.0
    assert stats.index_bytes > 0
    assert stats.baseline_mean_error >= 0.0
    assert repr(idx).startswith("DynamicRMIndex(")
    assert repr(stats).startswith("DynamicStats(")


def test_the_static_index_is_not_charged_for_dynamic_support():
    keys = keys_for("uniform", n=5000, seed=14)
    static = RMIndex(1024)
    static.build(keys, [i for i in range(len(keys))])
    assert static.stats().model_bytes > 0
