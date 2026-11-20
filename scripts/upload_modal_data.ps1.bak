# =============================================================================
# scripts/upload_modal_data.ps1
#
# Uploads the combined training+validation dataset to the Modal Volume
# "gemma-tounsi-data" at path /mnt/data/soup_trainval.jsonl.
#
# Prerequisites:
#   - Modal SDK installed and authenticated (modal setup completed)
#   - Modal profile "medbensaad26-cell" active
#   - Python 3.10+ with modal package available
#
# Usage:
#   .\scripts\upload_modal_data.ps1
#
# This script:
#   1. Validates that data/splits/soup_trainval.jsonl exists
#   2. Computes SHA256 fingerprint
#   3. Creates the Modal Volume (if needed)
#   4. Uploads the file to /mnt/data/soup_trainval.jsonl
#   5. Uploads the SHA256 fingerprint to /mnt/data/soup_trainval.jsonl.sha256
#   6. Commits the Volume
#
# NOT executed yet. Run manually after reviewing.
# =============================================================================

$ErrorActionPreference = "Stop"

# --- Configuration ----------------------------------------------------------
$DATASET_SOURCE = Join-Path -Path $PSScriptRoot -ChildPath "../data/splits/soup_trainval.jsonl" | Resolve-Path
$DATASET_REMOTE_PATH = "/mnt/data/soup_trainval.jsonl"
$SHA256_REMOTE_PATH = "/mnt/data/soup_trainval.jsonl.sha256"
$MODAL_VOLUME_NAME = "gemma-tounsi-data"
$MODAL_APP_NAME = "gemma-tounsi-train"

# --- Helpers ----------------------------------------------------------------
function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host $Text
    Write-Host ("=" * 70)
}

function Write-Success {
    param([string]$Text)
    Write-Host ("[OK] " + $Text) -ForegroundColor Green
}

function Write-Warning {
    param([string]$Text)
    Write-Host ("[WARN] " + $Text) -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Text)
    Write-Host ("[ERROR] " + $Text) -ForegroundColor Red
}

# --- Validate prerequisites -------------------------------------------------
Write-Header "Gemma Tounsi — Modal Dataset Upload"

# Check dataset exists
if (-not (Test-Path -Path $DATASET_SOURCE)) {
    Write-Error "Dataset not found: $DATASET_SOURCE"
    exit 1
}
Write-Success "Dataset found: $DATASET_SOURCE"

# Get file size
$fileInfo = Get-Item -Path $DATASET_SOURCE
$fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
Write-Host "  Size: $fileSizeMB MB"

# Check Modal CLI
$modalExe = Get-Command "modal" -ErrorAction SilentlyContinue
if (-not $modalExe) {
    Write-Error "Modal CLI not found on PATH. Activate the gemma-modal venv first."
    exit 1
}
Write-Success "Modal CLI found: $($modalExe.Source)"

# Check authentication
try {
    $profileOutput = & modal profile current 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Modal not authenticated. Run: modal setup"
        exit 1
    }
    Write-Success "Modal authenticated: $profileOutput.Trim()"
} catch {
    Write-Error "Modal authentication check failed: $_"
    exit 1
}

# --- Compute SHA256 ---------------------------------------------------------
Write-Header "Computing SHA256 Fingerprint"
$sha256 = [System.Security.Cryptography.SHA256]::Create()
$stream = [System.IO.File]::OpenRead($DATASET_SOURCE)
try {
    $hashBytes = $sha256.ComputeHash($stream)
    $hashString = [BitConverter]::ToString($hashBytes) -replace "-", ""
    $hashString = $hashString.ToLower()
}
finally {
    $stream.Dispose()
    $sha256.Dispose()
}

Write-Success "SHA256: $hashString"

# Save SHA256 to local file for reference
$sha256LocalFile = $DATASET_SOURCE + ".sha256"
$hashString | Out-File -FilePath $sha256LocalFile -Encoding utf8 -NoNewline
Write-Success "SHA256 saved locally: $sha256LocalFile"

# --- Upload to Modal Volume -------------------------------------------------
Write-Header "Uploading to Modal Volume"

# Create volume if it doesn't exist
Write-Host "Checking/creating Modal Volume: $MODAL_VOLUME_NAME"
& modal volume create $MODAL_VOLUME_NAME 2>&1
if ($LASTEXITCODE -ne 0) {
    # Volume may already exist — that's OK
    $volList = & modal volume list 2>&1
    if ($volList -match $MODAL_VOLUME_NAME) {
        Write-Success "Volume '$MODAL_VOLUME_NAME' already exists"
    } else {
        Write-Error "Failed to create/check volume '$MODAL_VOLUME_NAME'"
        exit 1
    }
}

# Upload dataset file
Write-Host "Uploading dataset to $DATASET_REMOTE_PATH ..."
& modal volume put $MODAL_VOLUME_NAME $DATASET_SOURCE $DATASET_REMOTE_PATH 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload dataset to Modal Volume"
    exit 1
}
Write-Success "Dataset uploaded to $DATASET_REMOTE_PATH"

# Upload SHA256 fingerprint
$tempSha256 = [System.IO.Path]::GetTempFileName()
$hashString | Out-File -FilePath $tempSha256 -Encoding utf8 -NoNewline
Write-Host "Uploading SHA256 fingerprint to $SHA256_REMOTE_PATH ..."
& modal volume put $MODAL_VOLUME_NAME $tempSha256 $SHA256_REMOTE_PATH 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload SHA256 fingerprint"
    exit 1
}
Write-Success "SHA256 fingerprint uploaded to $SHA256_REMOTE_PATH"

# Clean up temp file
Remove-Item -Path $tempSha256 -Force

# Commit the Volume
Write-Host "Committing Modal Volume ..."
& modal volume commit $MODAL_VOLUME_NAME 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to commit Modal Volume"
    exit 1
}
Write-Success "Volume committed"

# --- Summary ----------------------------------------------------------------
Write-Header "Upload Complete"
Write-Host "Source:      $DATASET_SOURCE"
Write-Host "Remote:      $DATASET_REMOTE_PATH"
Write-Host "SHA256:      $hashString"
Write-Host "Volume:      $MODAL_VOLUME_NAME"
Write-Host ""
Write-Success "Dataset is ready for Modal training."
Write-Host "Next step: Review soup_modal_app.py, then run:"
Write-Host "  modal run --detach soup_modal_app.py"
