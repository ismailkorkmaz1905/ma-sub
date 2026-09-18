$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

$userEnvironmentNames = @(
    "RUNPOD_POD_ID",
    "RUNPOD_API_KEY",
    "MAS_RUNPOD_SSH_KEY",
    "MAS_YTDLP_COOKIES",
    "MAS_GMAIL_ADDRESS",
    "MAS_GMAIL_APP_PASSWORD",
    "MAS_NOTIFY_TO",
    "MAS_DRIVE_STRICT_REMOTE",
    "MAS_RCLONE_CONFIG",
    "MAS_RUNPOD_AUTO_MIGRATE",
    "MAS_RUNPOD_NETWORK_VOLUME_ID",
    "MAS_RUNPOD_DATA_CENTER_ID",
    "MAS_RUNPOD_GPU_TYPE_ID",
    "MAS_RUNPOD_GPU_TYPE_IDS",
    "MAS_RUNPOD_MAX_COST_PER_HR",
    "MAS_RUNPOD_IMAGE",
    "MAS_RUNPOD_REGISTRY_AUTH_ID",
    "MAS_RUNPOD_STORAGE_QUOTE",
    "MAS_DELIVERY_EXECUTION_PLAN",
    "MAS_PRODUCTION_PRIORITY",
    "MAS_EPISODE_BUDGET_SECONDS"
)
foreach ($name in $userEnvironmentNames) {
    if ($name -eq "RUNPOD_POD_ID") {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
        if ($value) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    } elseif (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
        if ($value) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Missing .venv. Run: uv venv --python 3.11 .venv"
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $python -m mas.cli @args
exit $LASTEXITCODE
