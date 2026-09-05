$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Missing .venv. Run: uv venv --python 3.11 .venv"
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
& $python -m mas.cli @args
exit $LASTEXITCODE
