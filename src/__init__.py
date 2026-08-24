"""Gemma Tounsi 1.0 — project-specific tooling.

This package holds ONLY the logic that Soup cannot provide. Generic data
engineering (dedup, stats, split, convert, validate-by-format) is delegated to
the Soup CLI; see ``src/data/dedupe.py`` and ``src/data/split.py``.
"""
