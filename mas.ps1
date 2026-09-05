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
    "MAS_RCLONE_CONFIG"
)
foreach ($name in $userEnvironmentNames) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
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
& $python -m mas.cli @args
exit $LASTEXITCODE
