param(
    [ValidateRange(1, 999)]
    [int]$Episode = 13,
    [string]$Model = 'gpt-6-astra',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$promptPath = Join-Path $projectRoot 'docs\EPISODE_OPERATOR_PROMPT.md'
$episodePrompt = [System.IO.File]::ReadAllText($promptPath).Replace('{{EPISODE}}', [string]$Episode)
$codexArgs = @('--yolo', '-C', $projectRoot)
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
exit $LASTEXITCODE
