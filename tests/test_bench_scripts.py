"""Smoke tests for the benchmark and experiment scripts.

Same reasoning as the other script tests: these are what get run in front of
someone, and a script that breaks silently is worse than none. They also pin
the guard that refuses to report timings from an unoptimised build.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def run(script, args=()):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True, text=True, timeout=900,
    )
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined, combined
    return proc, combined


def optimised():
    sys.path.insert(0, str(REPO_ROOT / "python"))
    import hylis._rmi as rmi

    return getattr(rmi, "__optimized__", True)


def test_bench_index_runs():
    proc, out = run("bench_index.py", ["--quick"])
    if not optimised():
        assert proc.returncode != 0
        assert "meaningless" in out
        return
    assert proc.returncode == 0, out
    assert "ns/lookup" in out
    assert "sequential_gaps" in out
    assert "clustered" in out


def test_bench_index_accepts_explicit_sizes():
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("bench_index.py",
                    ["--sizes", "5000", "--distributions", "uniform"])
    assert proc.returncode == 0, out
    assert "uniform" in out


def test_curvature_experiment_shows_convergence_and_a_floor():
    proc, out = run("experiment_curvature.py",
                    ["-n", "20000", "--no-timing",
                     "--distributions", "lognormal,clustered"])
    assert proc.returncode == 0, out

    # The two headline behaviours must both appear in the output.
    assert "lognormal" in out
    assert "clustered" in out
    assert "max err" in out


def test_curvature_error_actually_falls_then_floors():
    """Parse the table rather than trusting that it printed something."""
    proc, out = run("experiment_curvature.py",
                    ["-n", "20000", "--no-timing", "--distributions", "clustered"])
    assert proc.returncode == 0, out

    means = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].replace(",", "").isdigit():
            means.append(float(parts[2].replace(",", "")))

    assert len(means) >= 4, out
    assert means[-1] < means[0], "adding models must help at least initially"
    assert means[-1] == pytest.approx(means[-2]), (
        "a stepped distribution must stop improving once every cluster owns "
        "a model"
    )


def test_stage1_experiment_runs_both_families():
    proc, out = run("experiment_stage1.py",
                    ["-n", "5000", "--epochs", "30",
                     "--distributions", "uniform,lognormal"])
    assert proc.returncode == 0, out
    assert "linear" in out
    assert "mlp" in out
    assert "busiest" in out


def test_bench_vector_produces_a_recall_curve():
    proc, out = run("bench_vector.py",
                    ["--random", "3000x16", "--skip-filtered"])
    if not optimised():
        assert proc.returncode != 0
        assert "meaningless" in out
        return
    assert proc.returncode == 0, out
    assert "recall@10" in out
    assert "hnsw" in out and "flat" in out


def test_bench_vector_recall_rises_with_ef():
    """Parse the curve rather than trusting that it printed one.

    ef is the quality knob and monotonicity is the property users rely on; a
    benchmark that silently printed a non-monotonic curve would be reporting
    a bug as a result.
    """
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("bench_vector.py", ["--random", "3000x16", "--skip-filtered"])
    assert proc.returncode == 0, out

    recalls = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "hnsw":
            recalls.append(float(parts[2]))

    assert len(recalls) >= 4, out
    assert recalls == sorted(recalls), f"recall fell as ef rose: {recalls}"
    assert recalls[-1] > 0.95


def test_bench_vector_filtered_crossover_runs():
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("bench_vector.py", ["--random", "4000x16"])
    assert proc.returncode == 0, out
    assert "selectivity" in out
    assert "flat" in out and "hnsw" in out


def test_scripts_report_their_own_usage():
    for script in ("bench_index.py", "experiment_curvature.py",
                   "experiment_stage1.py", "bench_vector.py"):
        proc, out = run(script, ["--help"])
        assert proc.returncode == 0, out
        assert "usage:" in out.lower()
