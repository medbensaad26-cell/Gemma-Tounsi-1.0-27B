from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parents[2] / "soup_modal_app.py"
    if not app_path.is_file():
        raise SystemExit(f"ERROR: {app_path.name} was not found at the repository root.")

    old = "FINALIZATION_RESERVE_SECONDS: int = 900  # 15 minutes"
    new = "FINALIZATION_RESERVE_SECONDS: int = 1800  # 30 minutes"

    source = app_path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            "ERROR: Expected exactly one 15-minute reserve declaration, "
            f"but found {count}. No files were changed."
        )

    backup_path = app_path.with_name("soup_modal_app.py.before_30min_reserve.bak")
    if backup_path.exists():
        raise SystemExit(
            f"ERROR: Backup already exists: {backup_path.name}. "
            "No files were changed; preserve the existing backup and inspect manually."
        )

    shutil.copy2(app_path, backup_path)
    updated = source.replace(old, new, 1)

    # Validate syntax before replacing the original file.
    compile(updated, str(app_path), "exec")
    app_path.write_text(updated, encoding="utf-8", newline="")

    print("Updated FINALIZATION_RESERVE_SECONDS from 900 to 1800.")
    print(f"Backup: {backup_path.name}")
    print("Python syntax: OK")
    print("Only the finalization-reserve declaration was changed.")
    print("No Modal commands, image builds, GPU allocations, or training launches were performed.")


if __name__ == "__main__":
    main()
