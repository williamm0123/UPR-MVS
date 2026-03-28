"""Namespace shim so `upr_mvs.*` resolves to top-level source directories."""

from __future__ import annotations

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent

# Make Python search submodules (datasets/engine/models/utils) at repository root.
__path__ = [str(_pkg_dir), str(_repo_root)]

__all__ = ["datasets", "engine", "models", "utils"]
