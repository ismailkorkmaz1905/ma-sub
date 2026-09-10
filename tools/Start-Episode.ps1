param(
    [ValidateRange(1, 999)]
    [int]$Episode = 13,
    [string]$Model = 'gpt-5.6-sol',
    [switch]$RetryIfSourceMissing,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$promptPath = Join-Path $projectRoot 'docs\EPISODE_OPERATOR_PROMPT.md'
$sourceUrlPath = Join-Path $projectRoot ("EPISODES\Muhtemel Ask {0}.Bolum\source\source.url" -f $Episode)
$mutex = New-Object System.Threading.Mutex($false, ("Local\MaSubEpisode{0}" -f $Episode))
$hasMutex = $false
try {
    $hasMutex = $mutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $hasMutex = $true
}
if (-not $hasMutex) {
    Write-Output ("Episode {0} launcher is already running. Duplicate start skipped." -f $Episode)
    [Environment]::Exit(0)
}
if ($RetryIfSourceMissing -and (Test-Path -LiteralPath $sourceUrlPath)) {
    Write-Output ("Episode {0} source is already resolved. Retry skipped." -f $Episode)
    [Environment]::Exit(0)
}
try {
    $episodePrompt = [System.IO.File]::ReadAllText($promptPath).Replace('{{EPISODE}}', [string]$Episode)
    $codexArgs = @('exec', '--yolo', '-C', $projectRoot, '--config', 'model_reasoning_effort="high"')
    if ($Model) { $codexArgs += @('--model', $Model) }
    if ($DryRun) {
        Write-Output ('Project: ' + $projectRoot)
        Write-Output ('Episode: ' + $Episode)
        Write-Output ('Prompt: ' + $promptPath)
        Write-Output ('Arguments: ' + ($codexArgs -join ' '))
        if ($episodePrompt.Contains('{{EPISODE}}')) { throw 'Episode placeholder remains' }
        return
    }
    $codexArgs += $episodePrompt
    & codex @codexArgs
    $exitCode = $LASTEXITCODE
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
exit $exitCode
