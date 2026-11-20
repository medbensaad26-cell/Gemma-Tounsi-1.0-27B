from __future__ import annotations

"""Apply the Gemma Tounsi runtime/shutdown fix to soup_modal_app.py.

This script only edits soup_modal_app.py. It does not import Modal, connect to
any service, build an image, allocate a GPU, or launch training.
"""

import ast
import os
from pathlib import Path

TARGET = Path("soup_modal_app.py")
BACKUP = Path("soup_modal_app.py.before_runtime_shutdown_fix.bak")

REPLACEMENTS = [
    ('MAX_RUNTIME_SECONDS: int = 82800  # 23 hours (training soft-stop)\nMODAL_FUNCTION_TIMEOUT_SECONDS: int = 86400  # 24 hours (Modal hard timeout)\nSTOP_AFTER_STEPS: Optional[int] = None  # Set to stop after N completed steps', "MAX_RUNTIME_SECONDS: int = 82800  # 23 hours (training soft-stop)\nMODAL_FUNCTION_TIMEOUT_SECONDS: int = 86400  # 24 hours (Modal hard timeout)\nSTOP_AFTER_STEPS: Optional[int] = None  # Set to stop after N completed steps\n# Reserve time before Modal's hard timeout for final Volume commits/manifest.\nFINALIZATION_RESERVE_SECONDS: int = 900  # 15 minutes\nPROCESS_POLL_INTERVAL_SECONDS: int = 10"),
    ('training_start_time = time.time()', 'training_start_time = time.monotonic()'),
    ('        else:\n            print("No existing checkpoint found. Starting fresh training.")\n            resume_args = []\n\n        # Build soup train command (BLOCKER 6)', '        else:\n            print("No existing checkpoint found. Starting fresh training.")\n            resume_args = []\n\n        # Remember the latest checkpoint that existed before this run.\n        # At the soft runtime limit, wait for one newer checkpoint to be\n        # fully written and committed before requesting process termination.\n        initial_checkpoint_step = latest_step\n\n        # Build soup train command (BLOCKER 6)'),
    ('        start_time = time.time()\n        process = subprocess.Popen(cmd)\n\n        # Monitor process for soft-stop conditions and per-checkpoint commits\n        while process.poll() is None:\n            elapsed = time.time() - start_time\n            if elapsed >= max_runtime_seconds:\n                print(\n                    f"MAX_RUNTIME_SECONDS ({max_runtime_seconds}) reached. "\n                    "Sending SIGTERM for graceful stop at next checkpoint."\n                )\n                process.send_signal(signal.SIGTERM)\n                break\n\n            if stop_after_steps is not None:\n                latest_step = find_latest_checkpoint_step(output_dir)\n                if latest_step is not None and latest_step >= stop_after_steps:\n                    print(\n                        f"STOP_AFTER_STEPS ({stop_after_steps}) reached at "\n                        f"step {latest_step}. Sending SIGTERM for graceful stop."\n                    )\n                    process.send_signal(signal.SIGTERM)\n                    break\n\n            # Per-checkpoint durability: commit any newly completed\n            # checkpoints to the Volume while training continues.\n            # HF Trainer writes trainer_state.json LAST, so its presence\n            # confirms the checkpoint is fully written and closed.\n            committed_steps = commit_checkpoint_volume(committed_steps)\n\n            time.sleep(10)\n\n        process.wait()\n\n        if process.returncode not in (0, -signal.SIGTERM):\n            raise subprocess.CalledProcessError(process.returncode, cmd)\n\n        print(f"Training completed with return code {process.returncode}")', '        # The internal soft-stop must leave time for the parent function to\n        # commit Volumes and write the manifest before Modal\'s hard timeout.\n        forced_stop_elapsed = (\n            MODAL_FUNCTION_TIMEOUT_SECONDS - FINALIZATION_RESERVE_SECONDS\n        )\n        if max_runtime_seconds <= 0 or max_runtime_seconds >= forced_stop_elapsed:\n            raise ValueError(\n                "max_runtime_seconds must be positive and leave the configured "\n                "finalization reserve before Modal\'s hard timeout."\n            )\n\n        process = subprocess.Popen(cmd)\n        termination_requested = False\n        termination_reason: Optional[str] = None\n\n        def request_sigterm(reason: str) -> None:\n            """Request process termination after checkpoint durability when possible.\n\n            Transformers 4.x, which is the range allowed by Soup 0.73.3,\n            does not expose the newer JIT SIGTERM checkpoint option. This\n            SIGTERM is therefore not assumed to trigger a Trainer save.\n            The normal stop path sends it only after a newer checkpoint has\n            completed and been committed. The final-deadline path preserves\n            the last committed checkpoint and may discard newer in-flight work.\n            """\n            nonlocal termination_requested, termination_reason\n            termination_requested = True\n            termination_reason = reason\n            print(f"[shutdown] Sending SIGTERM to Soup: {reason}")\n            try:\n                process.send_signal(signal.SIGTERM)\n            except ProcessLookupError:\n                # The process exited between poll() and signal delivery.\n                pass\n\n        soft_stop_requested = False\n\n        # Monitor the subprocess and commit every complete checkpoint.\n        while process.poll() is None:\n            elapsed = time.monotonic() - training_start_time\n\n            latest_step_now = find_latest_checkpoint_step(output_dir)\n            committed_steps = commit_checkpoint_volume(committed_steps)\n\n            # Explicit step-limited runs stop only after the target checkpoint\n            # has passed validation and has been committed to the Volume.\n            if stop_after_steps is not None:\n                if (\n                    latest_step_now is not None\n                    and latest_step_now >= stop_after_steps\n                    and latest_step_now in committed_steps\n                ):\n                    request_sigterm(\n                        f"STOP_AFTER_STEPS={stop_after_steps} reached; "\n                        f"checkpoint {latest_step_now} is committed"\n                    )\n                    break\n\n            if elapsed >= max_runtime_seconds and not soft_stop_requested:\n                soft_stop_requested = True\n                print(\n                    f"[shutdown] {max_runtime_seconds}s soft runtime reached. "\n                    "Waiting for a newer complete checkpoint to be committed."\n                )\n\n            if soft_stop_requested:\n                has_new_committed_checkpoint = (\n                    latest_step_now is not None\n                    and (\n                        initial_checkpoint_step is None\n                        or latest_step_now > initial_checkpoint_step\n                    )\n                    and latest_step_now in committed_steps\n                )\n                if has_new_committed_checkpoint:\n                    request_sigterm(\n                        f"new checkpoint {latest_step_now} is complete and committed"\n                    )\n                    break\n\n            # Last-resort stop before Modal\'s hard timeout. This may interrupt\n            # the current step, but the last fully committed checkpoint stays\n            # the recovery point; incomplete checkpoint directories are ignored.\n            if elapsed >= forced_stop_elapsed:\n                request_sigterm(\n                    "finalization reserve reached; preserving the latest "\n                    "fully committed checkpoint"\n                )\n                break\n\n            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)\n\n        if termination_requested:\n            try:\n                process.wait(timeout=60)\n            except subprocess.TimeoutExpired:\n                print(\n                    "[warn] Soup did not exit after SIGTERM within 60 seconds; "\n                    "sending SIGKILL. The last committed checkpoint remains "\n                    "the recovery point."\n                )\n                process.kill()\n                process.wait()\n        else:\n            process.wait()\n\n        expected_signal_exit = termination_requested and process.returncode in (\n            -signal.SIGTERM,\n            -signal.SIGKILL,\n        )\n        if process.returncode != 0 and not expected_signal_exit:\n            raise subprocess.CalledProcessError(process.returncode, cmd)\n\n        if termination_requested:\n            last_committed_step = max(committed_steps) if committed_steps else None\n            print(\n                f"Training process stopped ({termination_reason}). "\n                f"Last committed checkpoint step: {last_committed_step}."\n            )\n        else:\n            print(f"Training completed with return code {process.returncode}")'),
]

def main() -> None:
    if not TARGET.is_file():
        raise SystemExit("soup_modal_app.py was not found in the current directory.")
    if BACKUP.exists():
        raise SystemExit(f"Backup already exists: {BACKUP}. Review it before running again.")

    original = TARGET.read_text(encoding="utf-8")
    updated = original
    for index, (old, new) in enumerate(REPLACEMENTS, start=1):
        count = updated.count(old)
        if count != 1:
            raise SystemExit(
                f"Replacement {index} expected exactly one match, found {count}. "
                "No files were changed. Your source may differ from the reviewed version."
            )
        updated = updated.replace(old, new, 1)

    ast.parse(updated, filename=str(TARGET))
    required = (
        "FINALIZATION_RESERVE_SECONDS: int = 900",
        "training_start_time = time.monotonic()",
        "initial_checkpoint_step = latest_step",
        "time.monotonic() - training_start_time",
        "new checkpoint {latest_step_now} is complete and committed",
    )
    missing = [item for item in required if item not in updated]
    if missing:
        raise SystemExit(f"Post-edit validation failed: {missing}. No files were changed.")

    BACKUP.write_text(original, encoding="utf-8")
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, TARGET)
    print("Runtime/shutdown fix applied successfully.")
    print(f"Backup: {BACKUP}")
    print("Python syntax: OK")
    print("Only soup_modal_app.py was modified.")
    print("No Modal calls, image builds, GPU allocations, or training launches were performed.")


if __name__ == "__main__":
    main()
