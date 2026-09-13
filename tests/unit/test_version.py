"""The version written into the source tree must match the packaged version.

``wardcat.__version__`` comes from the installed distribution, and falls back to a
literal when the package is imported from a checkout that was never installed.
The release workflow checks the tag against ``pyproject.toml`` but cannot see the
fallback, so a bump that misses it would ship a checkout reporting the old number.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_fallback_version_matches_pyproject():
    packaged = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    source = (ROOT / "src" / "wardcat" / "__init__.py").read_text(encoding="utf-8")
    fallback = re.search(r'^\s*__version__ = "([^"]+)"', source, re.MULTILINE)
    assert fallback, "no __version__ fallback literal found in wardcat/__init__.py"
    assert fallback.group(1) == packaged
