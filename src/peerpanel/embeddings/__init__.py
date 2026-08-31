"""Embedding fixtures: build, save with a sha256 manifest, load, check."""

from .store import build, check, load, save

__all__ = ["build", "check", "load", "save"]
