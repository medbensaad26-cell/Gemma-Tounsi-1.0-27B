#!/usr/bin/env python3
"""Safely patch the Modal launcher to resume Gemma Tounsi from checkpoint >= 900.

Run from the Gemma-Tounsi-1.0-27B project root:
    python apply_gemma_tounsi_resume_patch.py

The script edits modal_train_custom_75_25_english.py only, creates a timestamped
backup first, and refuses to patch if the expected original code is not found.
"""
from __future__ import annotations

import ast
import base64
import re
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("modal_train_custom_75_25_english.py")

OLD_RESUME_BLOCK = '''        # Determine whether to resume from an existing checkpoint or start fresh.
        latest_step = find_latest_checkpoint_step(output_dir)
        if latest_step is not None:
            print(
                f"Resuming training from checkpoint at step {latest_step}."
            )
            resume_args = ["--resume", "auto"]
        else:
            print("No existing checkpoint found. Starting fresh training.")
            resume_args = []
'''

NEW_RESUME_BLOCK = '''        # Never cold-start: resume the newest complete checkpoint, but never
        # accept a recovery point older than the verified Step 900 checkpoint.
        latest_step = find_latest_checkpoint_step(output_dir)
        if latest_step is None:
            raise RuntimeError(
                "No complete checkpoint found. Refusing to start training from scratch."
            )
        if latest_step < MIN_RESUME_STEP:
            raise RuntimeError(
                f"Latest complete checkpoint is {latest_step}, below the required "
                f"recovery step {MIN_RESUME_STEP}. Refusing to start."
            )

        resume_checkpoint = output_dir / f"checkpoint-{latest_step}"
        if not checkpoint_is_complete(resume_checkpoint):
            raise RuntimeError(
                f"Selected checkpoint is incomplete: {resume_checkpoint}"
            )

        trainer_state_path = resume_checkpoint / "trainer_state.json"
        try:
            trainer_state = json.loads(
                trainer_state_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read trainer state from {trainer_state_path}: {exc}"
            ) from exc

        recorded_step = trainer_state.get("global_step")
        if recorded_step != latest_step:
            raise RuntimeError(
                f"Checkpoint directory says step {latest_step}, but trainer_state.json "
                f"records global_step={recorded_step!r}. Refusing to resume."
            )

        print(
            f"[resume] Resuming explicitly from {resume_checkpoint} "
            f"(global_step={recorded_step}).",
            flush=True,
        )
        resume_args = ["--resume", str(resume_checkpoint)]
'''

OLD_CHECKPOINT_FUNCTION = '''def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    """Return True only if all required checkpoint files exist.

    HF Trainer writes trainer_state.json LAST, so its presence indicates
    the checkpoint was fully written and closed.
    """
    for fname in CHECKPOINT_REQUIRED_FILES:
        if not (checkpoint_dir / fname).exists():
            return False
    return True


'''

NEW_CHECKPOINT_FUNCTION = r'''def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    """Validate checkpoint files and ensure trainer_state matches its directory."""
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_dir.name)
    if not checkpoint_dir.is_dir() or match is None:
        return False

    expected_step = int(match.group(1))
    for fname in CHECKPOINT_REQUIRED_FILES:
        file_path = checkpoint_dir / fname
        try:
            if not file_path.is_file() or file_path.stat().st_size <= 0:
                return False
        except OSError:
            return False

    try:
        state = json.loads(
            (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    return state.get("global_step") == expected_step


'''


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence of {description}; found {count}. "
            "No file was changed. Your source may differ from the version supplied."
        )
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(
            f"{TARGET} was not found. Run this script from the project root: "
            "D:\\Gemma-Tounsi-1.0-27B"
        )

    raw = TARGET.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")

    # Fail closed if this is not the expected original source version.
    if "MIN_RESUME_STEP: int = 900" in text and "resume_checkpoint = output_dir / f\"checkpoint-{latest_step}\"" in text:
        print("The explicit checkpoint-resume patch already appears to be applied.")
        ast.parse(text)
        return

    patched = text

    patched = replace_once(
        patched,
        "import os\nimport signal\n",
        "import os\nimport re\nimport signal\n",
        "standard-library import block",
    )

    patched = replace_once(
        patched,
        '  6. Automatic checkpoint resumption via "soup train --resume auto" + manifest',
        '  6. Explicit resume from the newest complete checkpoint (minimum Step 900)',
        "top-level resume description",
    )

    patched = replace_once(
        patched,
        "STOP_AFTER_STEPS: Optional[int] = None  # Set to stop after N completed steps\n",
        "STOP_AFTER_STEPS: Optional[int] = None  # Set to stop after N completed steps\n"
        "MIN_RESUME_STEP: int = 900  # Do not cold-start below the verified recovery point\n",
        "STOP_AFTER_STEPS setting",
    )

    patched = replace_once(
        patched,
        OLD_CHECKPOINT_FUNCTION,
        NEW_CHECKPOINT_FUNCTION,
        "checkpoint_is_complete function",
    )

    patched = replace_once(
        patched,
        OLD_RESUME_BLOCK,
        NEW_RESUME_BLOCK,
        "training resume-selection block",
    )

    patched = replace_once(
        patched,
        "    # Config fingerprint (hash of soup.modal.yaml)\n"
        "    config_sha256 = \"\"\n"
        "    config_path = Path(\"/root/soup.modal.yaml\")\n",
        "    # Config fingerprint (hash of the exact embedded Modal training config).\n"
        "    config_sha256 = \"\"\n"
        "    config_path = Path(\"/root/soup.custom_75_25_english.modal.yaml\")\n",
        "manifest config fingerprint path",
    )

    patched = replace_once(
        patched,
        "    BLOCKER 6: Automatic checkpoint resumption via --resume auto.\n",
        "    BLOCKER 6: Explicit checkpoint-path resumption; no fresh-start fallback.\n",
        "train() resume docstring",
    )

    # Ensure the edited source is syntactically valid before touching the file.
    ast.parse(patched)

    # Verify that the embedded configuration still points to the known persistent
    # output path and dataset path. Do not rewrite the embedded YAML.
    parsed = ast.parse(patched)
    config_node = next(
        node.value for node in parsed.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_CONFIG_B64" for t in node.targets)
    )
    embedded_yaml = base64.b64decode(ast.literal_eval(config_node)).decode("utf-8")
    if "output: /checkpoints/gemma-tounsi-1.0-27b-custom-75-25-english" not in embedded_yaml:
        raise RuntimeError(
            "Embedded Soup YAML output path does not match the checkpoint Volume path. "
            "No file was changed."
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = TARGET.with_name(f"{TARGET.name}.{timestamp}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_bytes(patched.replace("\n", newline).encode("utf-8"))

    # Read back and compile once more after writing.
    written = TARGET.read_text(encoding="utf-8-sig")
    ast.parse(written)

    print(f"Patched successfully: {TARGET}")
    print(f"Original backup:      {backup}")
    print("Resume policy: latest complete checkpoint with global_step >= 900.")
    print("If checkpoint-900 is still the newest complete checkpoint, this run selects 900.")
    print("If a later complete checkpoint exists, it selects that newer step instead.")
    print("No Modal run was started and no Volume was modified.")


if __name__ == "__main__":
    main()
