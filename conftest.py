"""Put the `python/` source tree on sys.path so `pytest` works from a bare
checkout with no install step and no PYTHONPATH.

The compiled pybind11 extensions are built directly into `python/hylis/`
(see cpp/bindings/CMakeLists.txt), so once the C++ build has run, `import
hylis` resolves to the package plus its extensions with nothing further to do.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "sift: needs the SIFT corpus in data/ (python scripts/fetch_data.py "
        "siftsmall); skipped automatically when it is absent",
    )
