"""Smoke tests for scripts/try_hybrid.py.

This is the demo -- the thing most likely to be run in front of someone, and
the only place the whole stack is visible at once. A demo that breaks silently
is worse than none, so these drive it the way a person would and assert that
its own differential check never reports a mismatch.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "try_hybrid.py"


def run(args=(), stdin=""):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=stdin, capture_output=True, text=True, timeout=900,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}\n{combined}"
    assert "Traceback" not in combined, combined
    assert "FAILED" not in combined, combined
    assert "MISMATCH" not in combined, combined
    return combined


def test_demo_runs_clean_and_shows_every_layer():
    out = run(["--demo"])
    # Each of these is one module made visible.
    assert "records written and fsynced" in out          # storage + WAL
    assert "chosen by building and timing every legal candidate" in out
    assert "answered by popcount" in out                 # the bitmap
    assert "sorted merge over 2 predicates" in out       # conjunctions
    assert "PreFilter" in out and "FilteredGraph" in out  # the planner
    assert "embeddings attached" in out                  # the vector column
    assert "recovered" in out                            # WAL replay
    assert "replaying the stored plan, not re-measuring it" in out
    assert "read back from the .fvecs sidecar" in out
    assert "every index agrees with a scan" in out


def test_the_demo_shows_the_vector_column_as_part_of_the_table():
    """Phase E moved the embeddings into the schema. Before it, the demo kept
    them beside the table and said so; if that ever regresses the walkthrough
    stops showing the thing the whole project is about."""
    out = run(["--demo"])
    assert "the row id is the record key" in out
    assert "brute force" in out, "the exact arm stopped being shown"
    assert "recall@" in out
    # More-like-this, and the reason its own seed is missing from the answer.
    assert "the seed is excluded" in out
    # Deletion leaves a hole the graph cannot close, and compaction is what
    # reclaims it -- both stated, because both are costs a reader should see.
    assert "orphaned rows" in out
    assert "a full graph rebuild" in out


def test_the_demo_checks_the_vector_half_against_numpy():
    """`check` grades the exact search against an independent oracle rather
    than against hylis. Without that the walkthrough would only be asserting
    that hylis agrees with itself."""
    out = run(["--demo"])
    assert "the exact vector search agrees with numpy" in out


def test_the_demo_reaches_the_bitmap_plan():
    """The plan phase D added. It needs an int64 low-cardinality column whose
    bit positions are the corpus's row ids, and if the demo ever stops
    arranging that, the plan silently never runs and nobody notices."""
    out = run(["--demo"])
    assert "BitmapFilteredGraph" in out
    assert "no row id is materialised at all" in out
    assert "the selectivity cost a popcount" in out


def test_every_plan_returns_the_same_rows():
    """The property that makes plan choice a performance decision rather than a
    semantic one. The demo prints an `agrees` column; nothing in it may say NO."""
    out = run(["--demo"])
    start = out.index("plan                            us")
    table = out[start:start + 600]
    assert "yes" in table
    assert "NO" not in table, table


def test_a_hand_session_survives_a_mixed_workload():
    session = "\n".join([
        "load 3000",
        "index price",
        "index category",
        "as band bitmap",
        "columns",
        "where category eq shoes",
        "where title contains ike",
        "where created_at isnull",
        "count band eq 0",
        "and category eq shoes, price lt 25000",
        "explain band lt 3 10",
        "hybrid band lt 3 5",
        "knn 5",
        "like 7 3",
        "vectors",
        "put 500 price=1 band=0 category=coats title=new thing in_stock=false"
        " created_at=2026-01-01",
        "get 500",
        "del 501",
        "compact",
        "check",
        "checkpoint",
        "recover",
        "check",
        "stats",
        "quit",
    ])
    out = run(stdin=session)
    assert "new thing" in out
    assert "every index agrees with a scan" in out


def test_bad_input_is_reported_rather_than_crashing():
    out = run(stdin="\n".join([
        "help",
        "load 2000",
        "nosuchcommand",
        "where nosuch eq 1",       # column not in the schema
        "where price nosuchop 1",  # operator that does not exist
        "where price lt abc",      # value that does not parse as int64
        "index nosuch",
        "as price nosuchkind",
        "like 999999 3",           # a record that does not exist
        "quit",
    ]))
    assert "unknown command" in out
    assert "no column named 'nosuch'" in out or "no column" in out.lower()
    assert "unknown op" in out
    assert "does not parse" in out


def test_commands_before_a_load_say_so_rather_than_failing():
    out = run(stdin="where price lt 1\nknn 5\ncheck\nquit\n")
    assert out.count("nothing loaded") >= 3


def test_printed_output_is_console_safe():
    """Windows consoles are cp1252 by default, and an em-dash in a print() has
    already cost this project one UnicodeEncodeError."""
    text = SCRIPT.read_text(encoding="utf-8")
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.lstrip()
        if not stripped.startswith(("print(", '"', 'f"')):
            continue
        assert "\u2014" not in line, f"try_hybrid.py:{number} em-dash in output"


def test_reports_its_own_usage():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert "usage:" in (proc.stdout + proc.stderr).lower()
