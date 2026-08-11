"""Smoke tests for scripts/try_vectors.py.

Same reasoning as tests/test_try_btree.py: a demo that breaks silently is
worse than no demo. These drive it the way a person would and fail on any
sign that the engine and the oracle disagreed.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "try_vectors.py"


def run(args=(), stdin=""):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=stdin, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr}"
    combined = proc.stdout + proc.stderr
    assert "MISMATCH" not in combined, combined
    assert "THIS IS A BUG" not in combined, combined
    assert "Traceback" not in combined, combined
    return combined


def test_demo_runs_clean():
    out = run(["--demo"])
    assert "same neighbours: True" in out
    assert "same answer: True" in out


def test_random_corpus_matches_the_oracle():
    out = run(["--random", "1000x16"], stdin="check 10\nbench 50 10\nquit\n")
    assert "recall 1.0000" in out


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_each_oracle_backed_metric_agrees(metric):
    out = run(["--random", "600x12", "--metric", metric], stdin="check 10\nquit\n")
    assert "same neighbours: True" in out


def test_inner_product_reports_that_the_oracle_does_not_cover_it():
    out = run(["--random", "300x8", "--metric", "ip"], stdin="check\nquit\n")
    assert "oracle covers L2 and cosine only" in out


def test_filter_plans_agree_across_selectivities():
    session = "".join(f"filter {s} 5\n" for s in (0.001, 0.01, 0.1, 0.5, 1.0))
    out = run(["--random", "2000x16"], stdin=session + "quit\n")
    assert out.count("same answer: True") == 5


def test_npy_and_fvecs_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 8)).astype(np.float32)

    npy = tmp_path / "v.npy"
    np.save(npy, data)

    sys.path.insert(0, str(REPO_ROOT / "python"))
    from hylis.datasets import write_fvecs
    fvecs = tmp_path / "v.fvecs"
    write_fvecs(fvecs, data)

    out = run(stdin=f"npy {npy}\nknn 0 3\ncheck 5\nfvecs {fvecs}\ncheck 5\nquit\n")
    assert out.count("200 x 8-d") == 2, "both loaders should report the same shape"
    assert out.count("recall 1.0000") == 2


def test_csv_round_trip(tmp_path):
    csv = tmp_path / "v.csv"
    csv.write_text("1,0,0\n0,1,0\n0,0,1\n5,5,5\n", encoding="utf-8")
    out = run(stdin=f"csv {csv}\nknn 0 2\nstats\nquit\n")
    assert "4 x 3-d" in out
    assert "id 0" in out


def test_typed_vectors_and_queries():
    out = run(stdin="add 1 0\nadd 0 1\nadd 5 5\nsearch 2 0.9 0.1\nstats\nquit\n")
    assert "added id 2" in out
    assert "3 vectors x 2 dims" in out


def test_sift_loads_or_skips_cleanly():
    out = run(["--sift"], stdin="truth 100\ncheck 10\nquit\n")
    if "fetch_data.py" in out:
        pytest.skip("SIFT corpus not downloaded")
    assert "recall 1.0000   (exact match)" in out


def test_bad_input_does_not_crash():
    """Fat-fingering must report and continue, not traceback.

    Ordered deliberately: the commands that need data come after the load,
    because an empty index is reported before arguments are looked at.
    """
    out = run(stdin="knn 0\n"                 # empty index
                    "nosuchcmd\n"             # unknown command
                    "npy /nope.npy\n"         # missing file
                    "random abc\n"            # unparseable argument
                    "random 200 8\n"          # ... now there is data
                    "filter 5 5\n"            # selectivity out of range
                    "search 1 1 2 3\n"        # wrong dimensionality
                    "knn 999\n"               # id out of range
                    "metric bogus\n"          # unknown metric
                    "check 5\n"               # still healthy afterwards
                    "quit\n")
    assert "index is empty" in out
    assert "unknown command" in out
    assert "no such file" in out
    assert "bad arguments" in out
    assert "selectivity must be in" in out
    assert "index holds 8-d vectors, got 3" in out
    assert "out of range" in out
    assert "unknown metric" in out
    assert "recall 1.0000" in out


def test_both_indexes_answer_the_same_query():
    """The standing requirement: a user can run a query both ways and see how
    they compare, rather than being handed one index and told to trust it."""
    out = run(["--random", "2000x16"],
              stdin="hnsw\nknn 0 5\nindex flat\nknn 0 5\nstats\nquit\n")
    assert "[hnsw]" in out
    assert "[flat]" in out
    assert "reachable" in out


def test_compare_reports_recall_against_the_exact_index():
    out = run(["--random", "2000x16"], stdin="compare 10 20\nquit\n")
    assert "exact, by definition" in out
    # The last row is the flat index and must be exact.
    assert "1.0000" in out


def test_ef_is_settable_and_affects_the_search():
    out = run(["--random", "1500x16"],
              stdin="hnsw\nef 10\nstats\nef 200\ncompare 10 20\nquit\n")
    assert "ef = 10" in out
    assert "ef = 200" in out


def test_building_the_graph_reports_its_shape():
    out = run(["--random", "3000x16"], stdin="hnsw 8 100\nstats\nquit\n")
    assert "levels" in out
    assert "mean degree at layer 0" in out


def test_unknown_index_kind_is_reported():
    out = run(["--random", "500x8"], stdin="index bogus\nquit\n")
    assert "unknown index" in out


def test_benchmarks_flag_an_unoptimised_build():
    """The guard itself must work, whichever way this build was compiled."""
    sys.path.insert(0, str(REPO_ROOT / "python"))
    import hylis._flat as flat

    out = run(["--random", "500x8"], stdin="bench 20 5\nquit\n")
    if flat.__optimized__:
        assert "NOT representative" not in out
    else:
        assert "NOT representative" in out
