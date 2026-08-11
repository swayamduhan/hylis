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


def test_merge_threshold_sweep_runs_and_checks_itself():
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("experiment_merge_threshold.py",
                    ["--quick", "-n", "20000", "--ops", "4000"])
    assert proc.returncode == 0, out
    # The oracle guard has to have been armed, not merely not tripped.
    assert "checked against a dict oracle" in out
    assert "MISMATCH" not in out, out
    assert "tau_e" in out and "off" in out, "the control arm is missing"


def test_merge_threshold_sweep_reports_the_cost_of_being_writable():
    """The number most easily left out. If this row ever disappears, the
    sweep is quietly only reporting the flattering half."""
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("experiment_merge_threshold.py",
                    ["--quick", "-n", "20000", "--ops", "3000"])
    assert proc.returncode == 0, out
    assert "What being writable costs" in out
    assert "static RMIndex" in out


def test_bench_sosd_falls_back_to_synthetic_without_a_download():
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("bench_sosd.py", ["--synthetic", "--limit", "50000"])
    assert proc.returncode == 0, out
    assert "ns/lookup" in out
    assert "choose_index() picked" in out
    assert "synthetic:clustered" in out


def test_fetch_data_explains_sosd_without_downloading_anything():
    proc, out = run("fetch_data.py", ["sosd"])
    assert proc.returncode == 0, out
    assert "fb_200M_uint64" in out
    assert "dataverse" in out.lower()
    assert "duplicate keys" in out


def test_router_scaling_study_runs_and_matches_recall():
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("experiment_router_scaling.py",
                    ["--quick", "--random", "30000x16", "--queries", "60",
                     "--recall", "0.99", "--sizes", "4000,8000"])
    assert proc.returncode == 0, out
    assert "Matched recall@10" in out
    assert "p1 gain" in out, "the extra-entry-point control is missing"
    assert "descent" in out


def test_router_scaling_refuses_a_degenerate_comparison():
    """If the target recall is reachable at the smallest ef for every n, no
    row is at its matched-recall point and the gains are not comparable. The
    script has to say so rather than print a trend."""
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("experiment_router_scaling.py",
                    ["--quick", "--random", "20000x16", "--queries", "50",
                     "--recall", "0.10", "--sizes", "4000,8000"])
    assert proc.returncode == 0, out
    assert "WARNING" in out and "too easy" in out
    assert "benefit grows" not in out, "drew a trend from a degenerate table"


def test_discontinuity_experiment_settles_the_retracted_claim():
    """This project predicted in writing that a discontinuous CDF favours the
    B+ tree, then measured otherwise. The script that settled it has to keep
    working, or the correction rests on a number nobody can reproduce."""
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("experiment_discontinuity.py", ["--quick"])
    assert proc.returncode == 0, out
    assert "cliffs" in out and "max_err" in out
    assert "The B+ tree won 0 of" in out, (
        "the B+ tree won a row -- if that is real, cpp/hylis/index/rmi.hpp "
        "should have its original prediction restored:\n" + out
    )


def test_planner_bench_reports_agreement_and_shortfalls():
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("bench_planner.py", ["--quick"])
    assert proc.returncode == 0, out
    assert "selectivity" in out and "regret" in out
    assert "agreed with the measured winner" in out
    # The post-filter shortfall column has to be there: it is the correctness
    # cost of the plan a system without a planner uses, and the reason the
    # module exists at all.
    assert "short" in out


def test_planner_calibration_beats_the_inherited_threshold():
    """The 0.5 crossover came from a different corpus at a different ef.
    Calibration measures it here instead -- and if that ever stops helping,
    the whole measure-don't-model stance needs revisiting."""
    if not optimised():
        pytest.skip("unoptimised build; timings are refused by design")
    proc, out = run("bench_planner.py", ["--quick", "--calibrate"])
    assert proc.returncode == 0, out
    assert "calibrated: crossover measured at" in out


def test_printed_output_is_console_safe():
    """Windows consoles are cp1252 by default, and an em-dash in a print()
    has already cost this project one UnicodeEncodeError. Cheaper to assert
    than to rediscover."""
    for script in ("bench_sosd.py", "experiment_merge_threshold.py",
                   "fetch_data.py", "experiment_router_scaling.py",
                   "experiment_discontinuity.py", "bench_planner.py"):
        text = (SCRIPTS / script).read_text(encoding="utf-8")
        for number, line in enumerate(text.split("\n"), start=1):
            stripped = line.lstrip()
            if not stripped.startswith(("print(", '"', "f\"")):
                continue
            assert "—" not in line, f"{script}:{number} em-dash in output"


def test_scripts_report_their_own_usage():
    for script in ("bench_index.py", "experiment_curvature.py",
                   "experiment_stage1.py", "bench_vector.py",
                   "experiment_merge_threshold.py", "bench_sosd.py",
                   "experiment_router_scaling.py",
                   "experiment_discontinuity.py", "bench_planner.py"):
        proc, out = run(script, ["--help"])
        assert proc.returncode == 0, out
        assert "usage:" in out.lower()
