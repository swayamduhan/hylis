"""Smoke tests for scripts/try_btree.py.

A demo script that silently breaks is worse than no demo, since it is the
thing most likely to be run in front of someone. These drive it the way a
person would and assert the differential check never reports a mismatch.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "try_btree.py"


def run(args=(), stdin=""):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=stdin, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr}"
    combined = proc.stdout + proc.stderr
    assert "MISMATCH" not in combined, combined
    assert "FAILED" not in combined, combined
    assert "Traceback" not in combined, combined
    return combined


def test_demo_runs_clean():
    out = run(["--demo"])
    assert "all invariants hold" in out or "invariants hold" in out
    assert "us/query" in out


@pytest.mark.parametrize("order", [3, 4, 7, 32])
def test_repl_survives_a_mixed_session(order):
    """Same operations at several orders, since order drives split/merge rates."""
    session = "\n".join([
        "insert 5 50", "insert 1 10", "insert 9 90", "insert 3 30",
        "insert 7 70", "insert 5 55",          # overwrite
        "find 5", "find 404", "has 3",
        "range 1 7", "lt 5", "ge 5", "eq 3", "gt 100",
        "erase 1", "erase 1",                  # second erase is a no-op
        "keys", "items", "stats", "validate", "check",
        "random 500", "check", "validate",
        "clear", "check",
        "quit",
    ])
    out = run(["--order", str(order)], stdin=session)
    assert "find(5) = 55" in out, "overwrite should have taken effect"
    assert "find(404) = None" in out
    assert "was not present" in out


def test_generated_dataset_loads_and_verifies():
    pytest.importorskip("numpy")
    out = run(["--load", "clustered:3000"],
              stdin="check\nvalidate\nbench 2000\nquit\n")
    assert "keys match the dict exactly" in out


def test_csv_by_index_and_by_header(tmp_path):
    named = tmp_path / "named.csv"
    named.write_text("id,score\n101,55\n102,66\nbad,row\n103,77\n", encoding="utf-8")
    plain = tmp_path / "plain.csv"
    plain.write_text("7,70\n3,30\n9,90\n", encoding="utf-8")

    out = run(stdin=f"csv {named} id score\nitems\ncsv {plain}\nitems\ncheck\nquit\n")
    assert "(101, 55)" in out
    assert "skipped 1 unusable row" in out, "the 'bad,row' line must be skipped"
    assert "(3, 30)" in out, "a headerless CSV's first row is data, not a header"


def test_bad_input_does_not_crash():
    """Fat-fingering at the prompt must report and continue, not traceback."""
    out = run(stdin="insert\nfind abc\nrange 1\nnosuchcmd\ncsv /nope.csv\n"
                    "load bogus 10\nunclosed 'quote\ninsert 1 1\ncheck\nquit\n")
    assert "unknown command" in out
    assert "bad arguments" in out
    assert "no such file" in out
    assert "keys match the dict exactly" in out
