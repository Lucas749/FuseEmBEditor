"""Compatibility adapters for external benchmark models."""

from .crispron_be import build_crispron
from .deepbaseeditor import build_deepbaseeditor
from .deepbe_pam import build_deepbe

__all__ = ["build_crispron", "build_deepbaseeditor", "build_deepbe"]
