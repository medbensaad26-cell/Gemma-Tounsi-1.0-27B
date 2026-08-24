"""Deduplication interface — thin wrapper over Soup, plus canonical id dedup.

Two distinct notions of "duplicate" exist in this project:

1. Exact duplicate ``id`` — a *canonical-schema* concern. Soup has no concept of
   our ``id`` field, so this small, deterministic pass is ours to own. It is the
   only "custom" dedup here and it is trivial (hash-set on ``id``), not an
   algorithm.

2. Near-duplicate *content* (paraphrases, boilerplate) — a GENERIC data-
   engineering concern. Soup already implements this well:

       soup data dedup PATH [--output OUT] [--threshold T] [--field F] [--semantic]

   (verified against Soup 0.73.3: MinHash Jaccard by default, embedding cosine
   under ``--semantic``). We therefore DELEGATE to it rather than reimplementing
   MinHash/SemDeDup. This module only builds and runs that command.

Nothing here downloads a dataset or trains anything.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # noqa: S404 - used to invoke the pinned Soup CLI, never shell=True
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .schema import load_jsonl, write_jsonl

__all__ = [
    "SoupDedupCommand",
    "build_soup_dedup_command",
    "dedupe_by_id",
    "run_soup_dedup",
    "main",
]


# --------------------------------------------------------------------------- #
# 1. Canonical id-level dedup (ours — Soup cannot see our `id` field)
# --------------------------------------------------------------------------- #


def dedupe_by_id(
    input_path: Path | str,
    output_path: Path | str,
) -> Tuple[int, int]:
    """Drop rows whose ``id`` was already seen, keeping the FIRST occurrence.

    Deterministic and order-preserving. Rows without a usable string ``id`` are
    kept as-is (schema validation, not dedup, is responsible for flagging them).

    Returns:
        ``(kept, removed)`` record counts.
    """
    loaded = load_jsonl(input_path)
    seen: set[str] = set()
    kept_records: List[dict] = []
    removed = 0
    for record in loaded.records:
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id:
            if record_id in seen:
                removed += 1
                continue
            seen.add(record_id)
        kept_records.append(record)

    write_jsonl(output_path, kept_records)
    return len(kept_records), removed


# --------------------------------------------------------------------------- #
# 2. Near-duplicate content dedup (delegated to Soup)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SoupDedupCommand:
    """A ready-to-run ``soup data dedup`` invocation.

    Attributes:
        argv: full command line, starting with the ``soup`` executable.
        output_path: path Soup will write the deduplicated dataset to.
        semantic: whether the semantic (embedding) backend was requested.
    """

    argv: Tuple[str, ...]
    output_path: str
    semantic: bool

    def as_str(self) -> str:
        """Return the command as a copy-pasteable string."""
        return " ".join(self.argv)


def build_soup_dedup_command(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    *,
    threshold: float = 0.8,
    field: Optional[str] = None,
    semantic: bool = False,
    soup_executable: str = "soup",
) -> SoupDedupCommand:
    """Build a ``soup data dedup`` command line WITHOUT running it.

    Only flags that exist in Soup 0.73.3 are emitted:
    ``--output/-o``, ``--threshold``, ``--field/-f`` and ``--semantic``. No flag
    is invented; unsupported ideas are simply not expressible here.

    Args:
        input_path: dataset Soup will read.
        output_path: where Soup writes output; defaults to Soup's own
            ``<input>_deduped.jsonl`` convention when omitted.
        threshold: similarity threshold in [0, 1] (MinHash Jaccard by default,
            embedding cosine under ``semantic``).
        field: single field to hash/embed; ``None`` uses Soup's default of all
            text fields concatenated.
        semantic: use embedding cosine (SemDeDup) instead of MinHash.
        soup_executable: the Soup entry point (overridable for tests).

    Raises:
        ValueError: threshold is outside [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_deduped.jsonl")
    output_path = Path(output_path)

    argv: List[str] = [
        soup_executable,
        "data",
        "dedup",
        str(input_path),
        "--output",
        str(output_path),
        "--threshold",
        str(threshold),
    ]
    if field:
        argv += ["--field", field]
    if semantic:
        argv.append("--semantic")

    return SoupDedupCommand(
        argv=tuple(argv), output_path=str(output_path), semantic=semantic
    )


def run_soup_dedup(
    command: SoupDedupCommand,
    *,
    check: bool = True,
    dry_run: bool = False,
) -> int:
    """Run a prepared Soup dedup command.

    Intended to run INSIDE the Soup container, where ``soup`` is on PATH. On a
    host without ``soup`` it fails clearly instead of pretending to work.

    Args:
        command: a command produced by :func:`build_soup_dedup_command`.
        check: raise ``subprocess.CalledProcessError`` on a non-zero exit.
        dry_run: print the command and return 0 without executing.

    Returns:
        The Soup process exit code (0 on ``dry_run``).
    """
    if dry_run:
        print(f"[dry-run] {command.as_str()}")
        return 0

    if shutil.which(command.argv[0]) is None:
        raise FileNotFoundError(
            f"'{command.argv[0]}' not found on PATH. Run this inside the Soup "
            f"container, e.g.: docker compose run --rm soup-cpu {command.as_str()}"
        )

    completed = subprocess.run(command.argv, check=check)  # noqa: S603
    return completed.returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the deduplication interface."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.dedupe",
        description=(
            "Deduplicate canonical datasets. 'id' mode removes exact duplicate "
            "ids (ours). 'content' mode delegates near-duplicate removal to "
            "'soup data dedup' (Soup 0.73.3)."
        ),
    )
    parser.add_argument(
        "mode",
        choices=["id", "content"],
        help="'id' = exact id dedup (local); 'content' = near-dup via Soup",
    )
    parser.add_argument("input", help="input canonical JSONL file")
    parser.add_argument("-o", "--output", default=None, help="output JSONL path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="[content] similarity threshold 0..1 (default: 0.8)",
    )
    parser.add_argument(
        "--field", default=None, help="[content] single field to hash/embed"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="[content] use embedding cosine (SemDeDup) instead of MinHash",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="[content] print the Soup command without running it",
    )
    args = parser.parse_args(argv)

    if args.mode == "id":
        output = args.output or str(
            Path(args.input).with_name(f"{Path(args.input).stem}_iddedup.jsonl")
        )
        try:
            kept, removed = dedupe_by_id(args.input, output)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"id dedup: kept {kept}, removed {removed} -> {output}")
        return 0

    # content mode
    try:
        command = build_soup_dedup_command(
            args.input,
            args.output,
            threshold=args.threshold,
            field=args.field,
            semantic=args.semantic,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"soup command: {command.as_str()}")
    try:
        return run_soup_dedup(command, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:  # pragma: no cover - Soup runtime
        print(f"error: soup data dedup failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
