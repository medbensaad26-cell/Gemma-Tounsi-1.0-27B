#!/usr/bin/env python3
"""Apply the Soup 0.73.3 model-size detection patch reproducibly.

This script patches `soup_cli/utils/gpu.py` in the currently installed
Soup package to fix the model-size substring collision bug:

    model_size_from_name("google/gemma-3-27b-it") -> 27 (was 7)

The patch:
  * Adds ("27b", 27) to the size_markers list (longer markers first).
  * Replaces substring matching (`marker in name_lower`) with
    boundary-safe regex matching so "7b" never matches inside "27b".

Usage:
    python scripts/apply_soup_compat_patch.py

The script is idempotent: running it twice is safe — the second run
detects the patch is already applied and exits successfully.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

PATCH_FILE = Path(__file__).resolve().parents[1] / "patches" / "soup-0.73.3-model-size.patch"
TARGET_MODULE = "soup_cli.utils.gpu"
TARGET_FUNCTION = "model_size_from_name"


def find_gpu_py() -> Path:
    """Return the absolute path to the installed soup_cli/utils/gpu.py."""
    try:
        module = importlib.import_module(TARGET_MODULE)
    except ImportError as exc:
        print(f"ERROR: cannot import {TARGET_MODULE}: {exc}", file=sys.stderr)
        sys.exit(1)

    file_path = Path(module.__file__).resolve()
    if not file_path.exists():
        print(f"ERROR: {file_path} does not exist", file=sys.stderr)
        sys.exit(1)
    return file_path


def patch_already_applied(text: str) -> bool:
    """Return True if the boundary-safe regex pattern is already present."""
    return 'r"(?<![a-z0-9.])"' in text and 're.escape(marker)' in text


def find_target_block(text: str) -> tuple[int, int]:
    """Find the line range of the size_markers + for loop block to patch.

    Returns (start_line, end_line) 0-indexed, inclusive of both lines.
    Raises SystemExit if the expected block is not found.
    """
    lines = text.splitlines()

    # Locate the start of size_markers
    start = None
    for i, line in enumerate(lines):
        if "size_markers = [" in line:
            start = i
            break
    if start is None:
        print("ERROR: 'size_markers = [' block not found in gpu.py", file=sys.stderr)
        sys.exit(1)

    # Locate the closing "]" of size_markers
    end_markers = None
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("[") - lines[i].count("]")
        if depth == 0:
            end_markers = i
            break
    if end_markers is None:
        print("ERROR: could not find closing ']' of size_markers", file=sys.stderr)
        sys.exit(1)

    # Locate the for loop with the substring match
    for_loop = None
    for i in range(end_markers, len(lines)):
        if "for marker, size in size_markers:" in lines[i]:
            for_loop = i
            break
    if for_loop is None:
        print("ERROR: 'for marker, size in size_markers:' not found", file=sys.stderr)
        sys.exit(1)

    # Locate the "if marker in name_lower:" line
    substring_line = None
    for i in range(for_loop, min(for_loop + 3, len(lines))):
        if "if marker in name_lower:" in lines[i]:
            substring_line = i
            break
    if substring_line is None:
        print("ERROR: 'if marker in name_lower:' not found", file=sys.stderr)
        sys.exit(1)

    return start, substring_line


def apply_patch(text: str) -> str:
    """Apply the patch to the source text. Returns the patched text."""
    if patch_already_applied(text):
        return text

    start, substring_line = find_target_block(text)
    lines = text.splitlines()

    # Insert ("27b", 27,) after the ("33b", 33), line within size_markers
    insert_idx = None
    for i in range(start, len(lines)):
        if '("33b", 33),' in lines[i]:
            insert_idx = i + 1
            break
    if insert_idx is None:
        print("ERROR: '(\"33b\", 33),' line not found in size_markers", file=sys.stderr)
        sys.exit(1)

    lines.insert(insert_idx, '        ("27b", 27),')
    lines.insert(insert_idx + 1, '        ("17b", 17),')
    lines.insert(insert_idx + 2, '        ("14b", 14),')
    lines.insert(insert_idx + 3, '        ("0.8b", 0.8),')

    # Replace ONLY the size_markers loop's substring match line.
    # substring_line points to the correct occurrence (right after size_markers for loop).
    # Adjust for the 4 lines we inserted above (all before substring_line).
    target_idx = substring_line + 4
    line = lines[target_idx]
    if "if marker in name_lower:" not in line:
        print(f"ERROR: expected 'if marker in name_lower:' at line {target_idx}, got: {line!r}", file=sys.stderr)
        sys.exit(1)
    indent = line[: len(line) - len(line.lstrip())]
    lines[target_idx] = (
        f'{indent}pattern = r"(?<![a-z0-9.])" + re.escape(marker) + r"(?![a-z0-9])"\n'
        f'{indent}if re.search(pattern, name_lower):'
    )

    return "\n".join(lines) + "\n"


def verify_patch(text: str) -> None:
    """Verify the patched text contains all required changes."""
    checks = [
        ('("27b", 27)', "27b marker added to size_markers"),
        ('r"(?<![a-z0-9.])"', "lookbehind boundary assertion"),
        ("re.escape(marker)", "regex escaping of marker"),
        (r'(?![a-z0-9])', "lookahead boundary assertion"),
        ("if re.search(pattern, name_lower):", "regex-based matching"),
    ]
    for pattern, description in checks:
        if pattern not in text:
            print(f"ERROR: patch verification failed — {description} not found", file=sys.stderr)
            sys.exit(1)

    # Verify the OLD buggy pattern is gone from the size_markers loop only.
    # The whisper_markers loop correctly uses substring matching (no collision risk).
    # Find the size_markers for loop and check its body uses re.search, not substring match.
    lines = text.splitlines()
    size_markers_loop_found = False
    for i, line in enumerate(lines):
        if "for marker, size in size_markers:" in line:
            size_markers_loop_found = True
            continue
        if size_markers_loop_found:
            # The next non-empty line should be the pattern/re.search line, not substring match
            stripped = line.strip()
            if stripped:
                if "if marker in name_lower:" in stripped:
                    print("ERROR: patch verification failed — size_markers loop still uses substring match", file=sys.stderr)
                    sys.exit(1)
                break
    if not size_markers_loop_found:
        print("ERROR: patch verification failed — size_markers for loop not found", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    gpu_py = find_gpu_py()
    print(f"Target file: {gpu_py}")

    text = gpu_py.read_text(encoding="utf-8")

    if patch_already_applied(text):
        print("Patch already applied — nothing to do.")
        return

    patched = apply_patch(text)
    verify_patch(patched)

    gpu_py.write_text(patched, encoding="utf-8")
    print("Patch applied successfully.")

    # Smoke test: verify model_size_from_name behavior
    import subprocess

    test_code = (
        "from soup_cli.utils.gpu import model_size_from_name as m; "
        "tests = [('google/gemma-3-27b-it', 27), ('google/gemma-3-7b-it', 7), "
        "('meta-llama/Llama-3.1-8B', 8), ('meta-llama/Llama-3.1-70B', 70), "
        "('Qwen/Qwen2.5-14B', 14), ('bigscience/bloom-17b', 17), "
        "('facebook/opt-1.7b', 1.7), ('unknown-model-xyz', 7)]; "
        "fails = [(n, m(n), e) for n, e in tests if m(n) != e]; "
        "print('FAILS:', fails) if fails else print('All model size tests passed')"
    )
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
