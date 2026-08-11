#!/usr/bin/env python3
"""Download benchmark datasets into ``data/``.

    python scripts/fetch_data.py siftsmall     # 10k vectors, ~5 MB
    python scripts/fetch_data.py sift          # 1M vectors, ~168 MB
    python scripts/fetch_data.py --list

``data/`` is git-ignored on purpose. Benchmark corpora do not belong in a
repository -- they are large, immutable and re-downloadable, and committing
them makes every future clone pay for them forever.

Nothing in hylis is blocked on this script: ``hylis.datasets`` generates
synthetic keys and vectors offline, and those are enough to develop and test
every module. Real data is for quotable numbers, not for getting started.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    # Directory the tarball unpacks to, relative to data/.
    unpacks_to: str
    # A file that must exist afterwards, used to detect an existing download.
    sentinel: str
    approx_mb: int
    description: str
    # Left None deliberately: pinning a checksum copied from a third-party
    # page and never verified is worse than not pinning one, because it fails
    # closed on a mirror difference you cannot debug. The script prints the
    # hash it computed -- paste it in here once you have downloaded it, and
    # every later run is verified against your own known-good copy.
    sha256: str | None = None


DATASETS: dict[str, Dataset] = {
    "siftsmall": Dataset(
        name="siftsmall",
        url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        unpacks_to="siftsmall",
        sentinel="siftsmall/siftsmall_base.fvecs",
        approx_mb=5,
        description="SIFT10K: 10k base / 100 query / 128-d, with ground truth",
        # Pinned from a download that then parsed to exactly the documented
        # shape (10000x128 base, 100 queries, 100x100 ground truth).
        sha256="b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e",
    ),
    "sift": Dataset(
        name="sift",
        url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        unpacks_to="sift",
        sentinel="sift/sift_base.fvecs",
        approx_mb=168,
        description="SIFT1M: 1M base / 10k query / 128-d, with ground truth",
        # Pinned from a download that then parsed to exactly the documented
        # shape (1000000x128 base, 10000 queries, 10000x100 ground truth).
        sha256="92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a",
    ),
}


# SOSD (Marcus et al., "Benchmarking Learned Indexes", VLDB 2021) is the
# standard corpus for the claim module 4 makes, and `fb` in particular is the
# adversarial case rmi.hpp already predicts a B+ tree should win.
#
# It is not listed in DATASETS because it is not automatically fetchable the
# way TEXMEX is: the files live on Harvard Dataverse behind per-file numeric
# ids that are not derivable from the dataset name. Guessing them would
# produce a script that fails confusingly, and pasting ids copied from a page
# without having downloaded through them would be pinning something unverified
# — the same reasoning as the sha256 policy above.
#
# So this prints what to fetch and where to put it, and hylis.datasets.read_sosd
# reads whatever is dropped into data/. Nothing about that is worse than an
# automated download except the convenience.
SOSD_DOI = "doi:10.7910/DVN/JGVF9A"
SOSD_LANDING = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=" + SOSD_DOI

# Plain ASCII in anything that gets printed: the Windows console is cp1252 by
# default, and an em-dash in a print() has already cost this project one
# UnicodeEncodeError.
SOSD_FILES = [
    ("books_200M_uint64", 1600, "Amazon book sale popularity: smooth, mildly skewed"),
    ("fb_200M_uint64", 1600, "Facebook user ids: heavy gaps, the adversarial case"),
    ("osm_cellids_200M_uint64", 1600, "OpenStreetMap cell ids: spatially clustered"),
    ("wiki_ts_200M_uint64", 1600, "Wikipedia edit timestamps: near-uniform with bursts"),
]


def print_sosd_help() -> None:
    print("SOSD key corpora (learned-index benchmark, VLDB 2021)\n")
    for name, mb, description in SOSD_FILES:
        print(f"  {name:<26} ~{mb:>5} MB   {description}")
    print(f"\nDownload from Harvard Dataverse:\n    {SOSD_LANDING}\n")
    print(f"and place the files directly in {DATA_DIR} (no subdirectory).\n")
    print("This has to be done in a browser. Harvard Dataverse sits behind an")
    print("AWS WAF that answers scripted clients with a bot challenge")
    print("(HTTP 202, 'x-amzn-waf-action: challenge', empty body), so curl and")
    print("the API both come back with nothing. A browser passes it normally.\n")
    print("Then, without needing all 200M keys resident:")
    print("    python scripts/bench_sosd.py --limit 10000000\n")
    print("Notes:")
    print("  * The key width is taken from the filename suffix and is not")
    print("    stored in the file, so do not rename them.")
    print("  * These corpora contain duplicate keys. Both indexes require")
    print("    strictly ascending unique keys, so the loader de-duplicates and")
    print("    reports how many it dropped -- the measured corpus is slightly")
    print("    smaller than the published one, and that is stated rather than")
    print("    quietly absorbed.")
    print("  * Nothing is blocked on this: hylis.datasets.synthetic_keys covers")
    print("    every distribution shape offline, including the clustered one")
    print("    that mimics `fb`.")


def existing_sosd() -> list[Path]:
    """SOSD-format files already sitting in data/."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.name.lower().endswith(("uint32", "uint64"))
    )


def human(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, printing progress.

    Uses urllib rather than requests because the canonical TEXMEX mirror is
    FTP, which requests does not speak.
    """
    print(f"  fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with open(dest, "wb") as fh:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    written += len(block)
                    if total:
                        pct = 100.0 * written / total
                        print(f"\r  {human(written)} / {human(total)} ({pct:.0f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r  {human(written)}", end="", flush=True)
            print()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"\ndownload failed: {exc}\n\n"
            f"Many networks block FTP. Fetch it manually instead:\n"
            f"    {url}\n"
            f"and unpack it so that {DATA_DIR / 'siftsmall'} exists.\n\n"
            f"You do not need it to work on hylis -- hylis.datasets generates\n"
            f"synthetic keys and vectors offline."
        ) from exc


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack ``archive`` into ``dest``, refusing entries that escape it.

    A tar member is free to name ``../../etc/passwd``; Python's own docs warn
    never to extract an untrusted archive without checking. Python 3.12+ has
    a filter for this, so use it where available and check by hand otherwise.
    """
    with tarfile.open(archive, "r:*") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
            return
        dest_resolved = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise SystemExit(f"refusing unsafe tar entry: {member.name!r}")
        tar.extractall(dest)


def fetch(dataset: Dataset, force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = DATA_DIR / dataset.sentinel

    if sentinel.exists() and not force:
        print(f"{dataset.name}: already present at {sentinel.parent}")
        return

    print(f"{dataset.name}: {dataset.description} (~{dataset.approx_mb} MB)")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{dataset.name}.tar.gz"
        download(dataset.url, archive)

        digest = sha256_of(archive)
        if dataset.sha256 is None:
            print(f"  sha256 {digest}")
            print("  (not pinned -- paste this into DATASETS to verify future runs)")
        elif digest != dataset.sha256:
            raise SystemExit(
                f"  checksum mismatch!\n"
                f"    expected {dataset.sha256}\n"
                f"    got      {digest}\n"
                f"  Refusing to unpack."
            )
        else:
            print(f"  sha256 ok ({digest[:16]}...)")

        target = DATA_DIR / dataset.unpacks_to
        if target.exists() and force:
            shutil.rmtree(target)

        print(f"  unpacking into {DATA_DIR}")
        safe_extract(archive, DATA_DIR)

    if not sentinel.exists():
        raise SystemExit(
            f"  unpacked, but {sentinel} is missing -- the archive layout is "
            f"not what was expected."
        )
    print(f"  done: {sentinel.parent}")


def verify(dataset: Dataset) -> bool:
    """Parse what was downloaded and report its real shape.

    Checking the files load and have consistent shapes catches a truncated or
    wrong-format download, which a size check alone would not.
    """
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        from hylis.datasets import load_sift
    except ImportError as exc:
        print(f"  cannot verify (numpy missing?): {exc}")
        return False

    try:
        ds = load_sift(dataset.name)  # type: ignore[arg-type]
    except (FileNotFoundError, ValueError) as exc:
        print(f"  verify failed: {exc}")
        return False

    gt = "none" if ds.ground_truth is None else f"{ds.ground_truth.shape}"
    print(
        f"  verified: {ds.n} base x {ds.dim}-d, "
        f"{ds.n_queries} queries, ground truth {gt}"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "datasets", nargs="*", default=[],
        help=f"one or more of: {', '.join(DATASETS)}",
    )
    parser.add_argument("--list", action="store_true", help="show available datasets")
    parser.add_argument("--sosd", action="store_true",
                        help="how to obtain the SOSD key corpora")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args(argv)

    if "sosd" in args.datasets or args.sosd:
        print_sosd_help()
        present = existing_sosd()
        if present:
            print(f"\nAlready in {DATA_DIR}:")
            for p in present:
                print(f"  {p.name:<26} {human(p.stat().st_size)}")
        args.datasets = [d for d in args.datasets if d != "sosd"]
        if not args.datasets:
            return 0

    if args.list or not args.datasets:
        print("Available datasets:\n")
        for ds in DATASETS.values():
            print(f"  {ds.name:<12} ~{ds.approx_mb:>4} MB   {ds.description}")
        print(f"  {'sosd':<12}  manual   "
              f"learned-index key corpora; run `fetch_data.py sosd` for how")
        print(f"\nDownloaded into {DATA_DIR} (git-ignored).")
        print("hylis.datasets works offline without any of these.")
        return 0

    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(DATASETS)}", file=sys.stderr)
        return 2

    for name in args.datasets:
        dataset = DATASETS[name]
        fetch(dataset, force=args.force)
        verify(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
