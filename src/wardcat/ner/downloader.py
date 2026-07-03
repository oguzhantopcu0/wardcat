"""
Programmatic SpaCy model installation.

Drives :class:`~wardcat.guard.Wardcat`'s auto-download path (when a language is
selected). Handles the awkward cases:

* models hosted on HuggingFace via an explicit wheel URL (e.g. Turkish),
* uv-based environments where ``pip`` is absent (falls back to ``uv pip``),
* the GitHub release wheel fallback for standard SpaCy models.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from wardcat.exceptions import ModelDownloadError
from wardcat.ner.spacy_catalog import get_spacy_model

logger = logging.getLogger(__name__)


def is_installed(model_name: str) -> bool:
    """Return ``True`` if the SpaCy model package is importable/installed."""
    try:
        import spacy.util

        return model_name in set(spacy.util.get_installed_models())
    except Exception:
        return False


def ensure_model(model_name: str, *, auto_download: bool = False, verbose: bool = False) -> bool:
    """Ensure a SpaCy model is available, downloading it if allowed.

    :param model_name:    SpaCy model package name (e.g. ``"de_core_news_md"``).
    :param auto_download: If ``True`` and the model is missing, download it.
    :param verbose:       Print high-level progress to stdout (CLI use).
    :returns:             ``True`` if the model is installed afterwards.
    """
    if is_installed(model_name):
        return True
    if not auto_download:
        return False
    download_model(model_name, verbose=verbose)
    return is_installed(model_name)


def download_model(model_name: str, *, verbose: bool = False) -> None:
    """Download and install a SpaCy model. Raises on failure.

    :raises ImportError:        if SpaCy itself is not installed.
    :raises ModelDownloadError: if the model is incompatible or the install fails
                                (subclass of ``RuntimeError``).
    """

    def _say(msg: str) -> None:
        # A library should not print — route to the logger. ``verbose`` selects
        # the level so INFO output appears when the caller opts in.
        logger.info(msg) if verbose else logger.debug(msg)

    info = get_spacy_model(model_name)

    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spacy is required for NER detection.\n"
            "Install with: uv add 'wardcat[ner]'  or  pip install 'wardcat[ner]'"
        ) from None

    # Hard incompatibility — model cannot be loaded regardless of --no-deps.
    if info and info.incompatible:
        raise ModelDownloadError(
            f"{model_name!r} is not compatible with SpaCy {spacy.__version__}.\n{info.note}"
        )

    if info and info.note:
        _say(f"Note: {info.note}")
    _say(f"Downloading SpaCy model: {model_name}")

    uv_bin = shutil.which("uv")
    env_skip = {**os.environ, "UV_SKIP_WHEEL_FILENAME_CHECK": "1"}

    # Check pip availability once upfront to avoid noisy "No module named pip" output.
    _pip_ok = (
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
        ).returncode
        == 0
    )

    def _run(cmd: list[str], *, skip_check: bool = False) -> int:
        e = env_skip if skip_check else None
        return subprocess.run(cmd, check=False, env=e).returncode

    def _install(packages: list[str], *, no_deps: bool = False) -> int:
        """Install packages using pip if available, otherwise fall back to uv."""
        cmd_suffix = ["--no-deps"] if no_deps else []
        if _pip_ok:
            rc = _run([sys.executable, "-m", "pip", "install"] + packages + cmd_suffix)
            if rc == 0:
                return 0
        if uv_bin:
            _say("Using uv pip install…")
            return _run(
                [uv_bin, "pip", "install"] + packages + cmd_suffix,
                skip_check=True,
            )
        return 1

    result_code = 1

    if info and info.wheel_url:
        # Models with explicit wheel URL (e.g. Turkish / HuggingFace).
        # Installed with --no-deps because the wheel metadata may pin an old
        # SpaCy version that conflicts with what is currently installed.
        result_code = _install([info.wheel_url], no_deps=True)

        # Install extra packages declared by this model (e.g. spacy-transformers for trf).
        if result_code == 0 and info.extra_packages:
            _say(f"Installing extra dependencies: {', '.join(info.extra_packages)}")
            result_code = _install(list(info.extra_packages))
    else:
        # Standard SpaCy models (explosion/spacy-models GitHub releases).
        # 1. Try `python -m spacy download` (works in pip-based virtualenvs).
        if _pip_ok:
            result_code = _run([sys.executable, "-m", "spacy", "download", model_name])

        # 2. In uv environments pip is absent — construct the GitHub wheel URL
        #    using the installed SpaCy minor version (models follow x.y.0 tags).
        if result_code != 0 and uv_bin:
            v = spacy.__version__.split(".")
            model_ver = f"{v[0]}.{v[1]}.0"
            gh_url = (
                f"https://github.com/explosion/spacy-models/releases/download/"
                f"{model_name}-{model_ver}/{model_name}-{model_ver}-py3-none-any.whl"
            )
            _say(f"Using uv pip install from GitHub ({model_ver})…")
            result_code = _run([uv_bin, "pip", "install", gh_url])

    if result_code != 0:
        raise ModelDownloadError(
            f"SpaCy model download failed: {model_name!r}\n"
            "The model may not be compatible with your installed SpaCy version.\n"
            "Check SpaCy version:     python -m spacy info"
        )
