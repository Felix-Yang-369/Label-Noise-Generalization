param([ValidateRange(1024,65535)][int]$Port = 5172)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) { throw 'Install the locked Python environment first; see README.md.' }
    & '.\.venv\Scripts\python.exe' -m app --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Demo stopped with exit code $LASTEXITCODE" }
} finally { Pop-Location }
