# Modal SDK Setup Report — Gemma Tounsi 1.0 27B

**Generated:** 2026-11-16
**Project:** Gemma-Tounsi-1.0-27B
**OS:** Windows 10
**Shell:** PowerShell

---

## 1. Windows Python Version Used

| Detail | Value |
|--------|-------|
| Python version | 3.14.3 |
| Executable | `C:\Users\PC VENTE\AppData\Local\Python\bin\python.exe` |
| pip version | 26.3 |
| Modal support | Python 3.10+ supported ✓ |

Python 3.10 was not available on the system; Python 3.14.3 (within Modal's supported 3.10+ range) was used instead. No system Python was installed or upgraded.

---

## 2. Modal SDK Version

| Component | Version |
|-----------|---------|
| `modal` Python SDK | 1.6.4 |
| `modal` CLI | 1.6.4 |

---

## 3. Virtual Environment Location

| Detail | Value |
|--------|-------|
| Path | `%USERPROFILE%\.venvs\gemma-modal` |
| Full path | `C:\Users\PC VENTE\.venvs\gemma-modal` |
| Inside repository | No ✓ |
| Inside Git repo | No ✓ |

---

## 4. Modal CLI Version

```
modal client version: 1.6.4
```

---

## 5. Authentication Status

**Status: Authenticated ✓**

- `modal setup` completed successfully
- Web browser authentication flow completed
- Token verified against `https://api.modal.com` and `https://api.modal2.com`
- Token written to `C:\Users\PC VENTE/.modal.toml` in profile `medbensaad26-cell`

**No API token values are printed or stored in this report.**

---

## 6. Active Modal Profile

| Setting | Value |
|---------|-------|
| Active profile | `medbensaad26-cell` |
| Workspace | `medbensaad26-cell` |
| Profile count | 1 |

---

## 7. Workspace Verification Result

| Setting | Value |
|---------|-------|
| default-environment | `main` |
| image-builder-version | `2025.06` |

All verification commands succeeded:
- `modal profile current` → `medbensaad26-cell`
- `modal profile list` → 1 profile: `medbensaad26-cell`
- `modal workspace settings list` → default-environment: `main`

---

## 8. Soup 0.73.3 Modal Dependency Requirements

### Soup 0.73.3 installation

| Detail | Value |
|--------|-------|
| Version | 0.73.3 |
| Location (Docker) | `/usr/local/lib/python3.10/dist-packages/` |
| Python | 3.10.12 |
| Base image | `ghcr.io/makazhanalpamys/soup:0.73.3` |

### Modal cloud integration (`soup_cli/cloud/modal.py`)

Soup 0.73.3 includes built-in Modal cloud support (issue #16, v0.71.18). Two workflows:

#### Workflow A — Plan-only generation

```
soup train --config soup.yaml --cloud modal --gpu l40s
```

- **Modal SDK locally required:** No
- **`soup-cli[modal]` extra required:** No
- **What happens:** Renders `soup_modal_app.py` stub from `soup.yaml`, prints `modal run soup_modal_app.py`
- **Soup extra used in rendered stub:** `soup-cli[train]==0.73.3` (only `[train]`, not `[modal]`)

#### Workflow B — Direct cloud submission

```
soup train --config soup.yaml --cloud modal --gpu l40s --cloud-submit
```

- **Modal SDK locally required:** Yes (`import modal` check)
- **`soup-cli[modal]` extra required:** Suggested by error message (`pip install "soup-cli[modal]"`)
- **Authentication required:** Modal token (`modal setup` or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`)

### Intended workflow

The user's intended workflow is:

1. `soup train --config soup.yaml --cloud modal --gpu l40s` (plan-only) → generates `soup_modal_app.py`
2. `modal run soup_modal_app.py` from Windows → runs via locally installed Modal SDK

**Conclusion:** `soup-cli[modal]` extra is **not required** for this workflow. The plan-only stub generation does not need Modal SDK locally. Running `modal run` needs the Modal SDK (already installed in the isolated venv).

### Docker requirements

- **Docker does NOT need `soup-cli[modal]`** — the container image already has `soup-cli[train]` (see Dockerfile comment line 11)
- The Modal stub renders `pip_install("soup-cli[train]==0.73.3")` inside the Modal container
- No Dockerfile changes needed for Modal support

---

## 9. Required Installation Changes

None required beyond what was done:

| Component | Status |
|-----------|--------|
| Modal SDK in isolated venv | ✓ Installed (1.6.4) |
| Modal CLI accessible | ✓ `modal.exe` in venv Scripts |
| Modal authentication | ✓ Verified |
| Soup 0.73.3 Docker image | ✓ Unchanged |
| `soup-cli[modal]` extra | ✗ Not needed for plan-only workflow |
| Dockerfile changes | ✗ None needed |
| `soup.yaml` changes | ✗ None needed |

---

## 10. Confirmation — No GPU Allocated

| Resource | Status |
|----------|--------|
| Modal Volumes created | No ✓ |
| Modal Secrets created | No ✓ |
| Modal App deployed | No ✓ |
| Modal container started | No ✓ |
| L40S GPU allocated | No ✓ |
| Training dataset uploaded | No ✓ |
| Training launched | No ✓ |
| GPU compute credits consumed | No ✓ |

This was strictly an SDK installation, authentication, and verification task.

---

## 11. Remaining Setup Requirements

| Item | Status | Notes |
|------|--------|-------|
| Generate `soup_modal_app.py` | Pending | Run `soup train --config soup.yaml --cloud modal --gpu l40s` (plan-only) |
| Run Modal app | Pending | `modal run soup_modal_app.py` from Windows venv |
| Hugging Face token for training | Pending | Will be provided via Modal Secret at deploy time (not created yet) |
| `soup-cli[modal]` extra | Not needed | Only required if using `--cloud-submit` flag directly |

---

## Files Modified

| File | Action |
|------|--------|
| None | No project files were modified |

## Files Created (outside repository)

| File | Location |
|------|----------|
| Virtual environment | `%USERPROFILE%\.venvs\gemma-modal\` |
| Modal config (credentials) | `%USERPROFILE%\.modal.toml` |

**No files were created or modified inside the `Gemma-Tounsi-1.0-27B` repository.**

## Commands Executed

```powershell
# Phase 1 — Environment inspection
py -0p
python --version
where.exe python
docker --version
docker compose version

# Phase 2 — Create isolated Modal venv
New-Item -ItemType Directory -Path "$env:USERPROFILE\.venvs" -Force | Out-Null
python -m venv "$env:USERPROFILE\.venvs\gemma-modal"

# Upgrade pip
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\python.exe" -m pip install --upgrade pip

# Install Modal SDK
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\python.exe" -m pip install modal

# Phase 3 — Verify installation
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\python.exe" --version
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\python.exe" -m pip show modal
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\python.exe" -c "import modal; print('Modal SDK import successful')"
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\modal.exe" --version

# Phase 4 — Authenticate (with encoding workaround)
$env:PYTHONIOENCODING="utf-8"; $env:NO_COLOR="1"
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\python.exe" -m modal.cli.entry_point setup

# Phase 5 — Verify authentication
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\modal.exe" profile current
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\modal.exe" profile list
& "$env:USERPROFILE\.venvs\gemma-modal\Scripts\modal.exe" workspace settings list

# Phase 6 — Inspect Soup Modal integration
docker compose run --rm soup-cpu pip show soup-cli
docker compose run --rm soup-cpu cat /usr/local/lib/python3.10/dist-packages/soup_cli/cloud/modal.py
docker compose run --rm soup-cpu sh -c "grep -n 'cloud\|modal\|submit' /usr/local/lib/python3.10/dist-packages/soup_cli/commands/train.py"
```
