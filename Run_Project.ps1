param([ValidateSet('prepare','eda','smoke','tune','train','evaluate','report','all','verify')][string]$Stage='all')
$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$taskTemp=Join-Path $taskRoot '.tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$taskPreviousTemp=$env:TEMP
$taskPreviousTmp=$env:TMP
$taskPreviousMpl=$env:MPLCONFIGDIR
Push-Location -LiteralPath $taskRoot
try {
    $env:TEMP=$taskTemp
    $env:TMP=$taskTemp
    $env:MPLCONFIGDIR=$taskTemp
    $taskPython=Join-Path $taskRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the Python 3.12 environment first; see README.' }
    if ($Stage -eq 'verify') { & $taskPython -m pytest -q }
    else { & $taskPython -m src.pipeline $Stage }
    if ($LASTEXITCODE -ne 0) { throw "Project stage failed with exit code $LASTEXITCODE" }
} finally {
    $env:TEMP=$taskPreviousTemp
    $env:TMP=$taskPreviousTmp
    $env:MPLCONFIGDIR=$taskPreviousMpl
    Pop-Location
}
