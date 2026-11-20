# =============================================================================
# scripts/upload_modal_data.ps1
#
# Uploads the combined training+validation dataset to the Modal Volume
# gemma-tounsi-data at path /mnt/data/soup_trainval.jsonl.
#
# Prerequisites:
#   - Modal SDK installed and authenticated (modal setup completed)
#   - Modal profile medbensaad26-cell active
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
$DATASET_SOURCE = (Resolve-Path -Path (Join-Path $PSScriptRoot "../data/splits/soup_trainval.jsonl")).Path
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

# --- Invoke-ModalCli --------------------------------------------------------
# Runs the Modal CLI via Start-Process so Python's DeprecationWarning on stderr
# cannot become a PowerShell NativeCommandError terminating error when
# $ErrorActionPreference = "Stop".
#
# Success is determined solely by ExitCode == 0. Stderr content (e.g. Python
# DeprecationWarning) does NOT cause failure.
#
# Returns a hashtable with:
#   ExitCode  - process exit code (int)
#   Stdout    - stdout content as string
#   Stderr    - stderr content as string
#
# The executable is resolved to a full path string so executable paths
# containing spaces are passed to Start-Process correctly. Arguments are
# passed as an array so arguments containing spaces (e.g. Windows dataset
# paths) are preserved as separate arguments in their original order.
# Modal credentials are never passed as arguments.
function Invoke-ModalCli {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [string]$Executable = "modal"
    )

    # Resolve the executable to a full path string. Start-Process receives the
    # path as a single string, so executable paths containing spaces are handled
    # correctly without shell quoting.
    $exePath = $Executable
    if ($exePath -eq "modal" -or -not (Test-Path -Path $exePath -PathType Leaf)) {
        $cmd = Get-Command "modal" -ErrorAction SilentlyContinue
        if ($cmd) { $exePath = $cmd.Source }
    }

    $stdoutFile = Join-Path $env:TEMP ("modal-cli-" + [guid]::NewGuid().ToString("N") + ".out")
    $stderrFile = Join-Path $env:TEMP ("modal-cli-" + [guid]::NewGuid().ToString("N") + ".err")

    $exitCode = 1
    $stdout = ""
    $stderr = ""

    # PowerShell 5.1's Start-Process does not quote -ArgumentList elements that
    # contain spaces, so they would be split by the target's command-line
    # parser. Pre-quote any argument containing whitespace so each argument
    # arrives at the target as a single argv element, preserving original order.
    $separatorPattern = "[" + [char]32 + [char]9 + "]"
    $quotedArguments = $Arguments | ForEach-Object {
        if ($_ -match $separatorPattern) {
            '"' + ($_ -replace '"', '""') + '"'
        } else {
            $_
        }
    }

    try {
        $process = Start-Process `
            -FilePath $exePath `
            -ArgumentList $quotedArguments `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile
        $exitCode = $process.ExitCode
    } catch {
        $exitCode = 1
        $stderr = $_.Exception.Message
    } finally {
        if (Test-Path -Path $stdoutFile -PathType Leaf) {
            $stdout = Get-Content -Raw $stdoutFile -ErrorAction SilentlyContinue
            if (-not $stdout) { $stdout = "" }
        }
        if (Test-Path -Path $stderrFile -PathType Leaf) {
            $stderr = Get-Content -Raw $stderrFile -ErrorAction SilentlyContinue
            if (-not $stderr) { $stderr = "" }
        }
        Remove-Item $stdoutFile, $stderrFile -Force -ErrorAction SilentlyContinue
    }

    return @{
        ExitCode = $exitCode
        Stdout   = $stdout
        Stderr   = $stderr
    }
}

# --- Validate prerequisites -------------------------------------------------
Write-Header "Gemma Tounsi - Modal Dataset Upload"

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
# Use Invoke-ModalCli so Python's DeprecationWarning on stderr cannot become
# a PowerShell NativeCommandError terminating error.
# Exit code is the authoritative success/failure signal.
$authResult = Invoke-ModalCli -Arguments @("profile", "current") -Executable $modalExe.Source

if ($authResult.ExitCode -ne 0) {
    Write-Error "Modal not authenticated. Exit code: $($authResult.ExitCode)`n$($authResult.Stderr)"
    exit 1
}

if ($authResult.Stderr) {
    Write-Warning $authResult.Stderr
}

if (-not $authResult.Stdout -or $authResult.Stdout.Trim() -eq "") {
    Write-Error "Modal authentication check failed: no profile output"
    exit 1
}

Write-Success "Modal authenticated: $($authResult.Stdout.Trim())"

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

# Check if volume already exists before attempting creation
Write-Host "Checking Modal Volume: $MODAL_VOLUME_NAME"
$volumeListResult = Invoke-ModalCli -Arguments @("volume", "list") -Executable $modalExe.Source

if ($volumeListResult.ExitCode -ne 0) {
    Write-Error "Failed to list Modal volumes. Exit code: $($volumeListResult.ExitCode)`n$($volumeListResult.Stderr)"
    exit 1
}

if ($volumeListResult.Stderr) {
    Write-Warning $volumeListResult.Stderr
}

$volumeExists = $volumeListResult.Stdout -match $MODAL_VOLUME_NAME

if ($volumeExists) {
    Write-Success "Volume $MODAL_VOLUME_NAME already exists"
} else {
    Write-Host "Creating Modal Volume: $MODAL_VOLUME_NAME ..."
    $createResult = Invoke-ModalCli -Arguments @("volume", "create", $MODAL_VOLUME_NAME) -Executable $modalExe.Source

    if ($createResult.ExitCode -ne 0) {
        Write-Error "Failed to create Modal Volume $MODAL_VOLUME_NAME. Exit code: $($createResult.ExitCode)`n$($createResult.Stderr)"
        exit 1
    }

    if ($createResult.Stderr) {
        Write-Warning $createResult.Stderr
    }

    Write-Success "Volume $MODAL_VOLUME_NAME created"
}

# Verify the volume exists
$verifyResult = Invoke-ModalCli -Arguments @("volume", "list") -Executable $modalExe.Source
if ($verifyResult.ExitCode -ne 0) {
    Write-Error "Failed to verify Modal Volume. Exit code: $($verifyResult.ExitCode)`n$($verifyResult.Stderr)"
    exit 1
}

if ($verifyResult.Stderr) {
    Write-Warning $verifyResult.Stderr
}

if ($verifyResult.Stdout -notmatch $MODAL_VOLUME_NAME) {
    Write-Error "Modal Volume $MODAL_VOLUME_NAME does not exist after creation attempt"
    exit 1
}

# Upload dataset file
Write-Host "Uploading dataset to $DATASET_REMOTE_PATH ..."
$uploadResult = Invoke-ModalCli -Arguments @("volume", "put", $MODAL_VOLUME_NAME, $DATASET_SOURCE, $DATASET_REMOTE_PATH) -Executable $modalExe.Source

if ($uploadResult.ExitCode -ne 0) {
    Write-Error "Failed to upload dataset to Modal Volume. Exit code: $($uploadResult.ExitCode)`n$($uploadResult.Stderr)"
    exit 1
}

if ($uploadResult.Stderr) {
    Write-Warning $uploadResult.Stderr
}
Write-Success "Dataset uploaded to $DATASET_REMOTE_PATH"

# Upload SHA256 fingerprint
$tempSha256 = [System.IO.Path]::GetTempFileName()
$hashString | Out-File -FilePath $tempSha256 -Encoding utf8 -NoNewline
Write-Host "Uploading SHA256 fingerprint to $SHA256_REMOTE_PATH ..."
$uploadShaResult = Invoke-ModalCli -Arguments @("volume", "put", $MODAL_VOLUME_NAME, $tempSha256, $SHA256_REMOTE_PATH) -Executable $modalExe.Source

if ($uploadShaResult.ExitCode -ne 0) {
    Write-Error "Failed to upload SHA256 fingerprint. Exit code: $($uploadShaResult.ExitCode)`n$($uploadShaResult.Stderr)"
    exit 1
}

if ($uploadShaResult.Stderr) {
    Write-Warning $uploadShaResult.Stderr
}
Write-Success "SHA256 fingerprint uploaded to $SHA256_REMOTE_PATH"

# Clean up temp file
Remove-Item -Path $tempSha256 -Force

# Commit the Volume
Write-Host "Committing Modal Volume ..."
$commitResult = Invoke-ModalCli -Arguments @("volume", "commit", $MODAL_VOLUME_NAME) -Executable $modalExe.Source

if ($commitResult.ExitCode -ne 0) {
    Write-Error "Failed to commit Modal Volume. Exit code: $($commitResult.ExitCode)`n$($commitResult.Stderr)"
    exit 1
}

if ($commitResult.Stderr) {
    Write-Warning $commitResult.Stderr
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
