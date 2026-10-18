[CmdletBinding()]
param(
    [string]$DestinationPath,
    [string]$PythonExecutable,
    [switch]$PrepareOnly
)

# Create an input-only sibling run. Historical science files are never edited.
$ErrorActionPreference = 'Stop'
$evidenceRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
if (-not (Test-Path -LiteralPath (Join-Path $evidenceRoot 'src\pipeline.py'))) {
    throw 'Place Reproduce.ps1 at the repository root before running it.'
}

if (-not $DestinationPath) {
    $runName = (Split-Path -Leaf $evidenceRoot) + '-reproduction-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    $DestinationPath = Join-Path (Split-Path -Parent $evidenceRoot) $runName
}
$newRun = [System.IO.Path]::GetFullPath($DestinationPath).TrimEnd('\', '/')
if ($newRun.Equals($evidenceRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $newRun.StartsWith($evidenceRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
    $newRun.StartsWith($evidenceRoot + '/', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose a destination outside the evidence checkout.'
}
if (Test-Path -LiteralPath $newRun) {
    throw 'Destination already exists. Use a new directory; no existing files will be overwritten.'
}
$destinationParent = Split-Path -Parent $newRun
if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
    throw 'The destination parent directory must already exist.'
}

$inputFiles = @(
    'Run_Project.ps1', 'Run_Demo.ps1', 'pytest.ini',
    'requirements.txt', 'requirements.lock.txt',
    'config\protocol.json', 'config\main_matrix.json',
    'data\raw\UCI_Dry_Bean_Original.zip', 'data\raw\Dry_Bean_Dataset.xlsx'
)
foreach ($relative in $inputFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $evidenceRoot $relative) -PathType Leaf)) {
        throw "Required input is missing: $relative"
    }
}
foreach ($folder in @('src', 'tests', 'app')) {
    if (-not (Test-Path -LiteralPath (Join-Path $evidenceRoot $folder) -PathType Container)) {
        throw "Required source directory is missing: $folder"
    }
}
$protocol = Get-Content -LiteralPath (Join-Path $evidenceRoot 'config\protocol.json') -Raw | ConvertFrom-Json
$archiveHash = (Get-FileHash -LiteralPath (Join-Path $evidenceRoot 'data\raw\UCI_Dry_Bean_Original.zip') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($archiveHash -ne $protocol.dataset.archive_sha256) {
    throw 'Raw archive SHA-256 differs from the declared protocol.'
}

if (-not $PythonExecutable) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $selectedPython = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)'
        if ($LASTEXITCODE -ne 0 -or -not $selectedPython) {
            throw 'The Python launcher could not select Python 3.12. Pass -PythonExecutable with an installed Python 3.12 path.'
        }
        $PythonExecutable = "$selectedPython".Trim()
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw 'Install Python 3.12 or pass -PythonExecutable with its absolute path.' }
        $PythonExecutable = $pythonCommand.Source
    }
}
$versionCheck = & $PythonExecutable -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"; print(sys.version.split()[0])'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required for this documented reproduction.' }

New-Item -ItemType Directory -Path $newRun | Out-Null
foreach ($folder in @('config', 'data\raw', 'logs')) {
    New-Item -ItemType Directory -Path (Join-Path $newRun $folder) -Force | Out-Null
}
foreach ($relative in $inputFiles) {
    Copy-Item -LiteralPath (Join-Path $evidenceRoot $relative) -Destination (Join-Path $newRun $relative)
}
foreach ($folder in @('src', 'tests', 'app')) {
    $sourceFolder = Join-Path $evidenceRoot $folder
    $sourceFiles = Get-ChildItem -LiteralPath $sourceFolder -File -Recurse |
        Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -ne '.pyc' }
    foreach ($file in $sourceFiles) {
        $relative = $file.FullName.Substring($evidenceRoot.Length).TrimStart('\', '/')
        $target = Join-Path $newRun $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}

$receipt = [ordered]@{
    created_utc = [DateTime]::UtcNow.ToString('o')
    source_checkout = $evidenceRoot
    new_run_directory = $newRun
    python_executable = $PythonExecutable
    python_version = "$versionCheck"
    raw_archive_sha256 = $archiveHash
    copied_scientific_source = @{}
    copied_original_results = $false
    status = 'input-only workspace created'
}
foreach ($name in @('core.py', 'pipeline.py')) {
    $sourceHash = (Get-FileHash -LiteralPath (Join-Path $evidenceRoot "src\$name") -Algorithm SHA256).Hash.ToLowerInvariant()
    $copyHash = (Get-FileHash -LiteralPath (Join-Path $newRun "src\$name") -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $copyHash) { throw 'Scientific source copy did not preserve exact bytes.' }
    $receipt.copied_scientific_source[$name] = $copyHash
}
$receiptPath = Join-Path $newRun 'REPRODUCTION_SETUP.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Host "New run: $newRun"
if ($PrepareOnly) {
    Write-Host 'Input-only workspace created. No environment installation or research fits were run.'
    return
}

Push-Location -LiteralPath $newRun
try {
    & $PythonExecutable -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    & '.\Run_Project.ps1' -Stage all
    & '.\Run_Project.ps1' -Stage verify
    $receipt.status = 'full pipeline and verification completed'
    $receipt['completed_utc'] = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    Write-Host "Reproduction completed: $newRun"
    Write-Host 'Start the local demonstration from that directory with .\Run_Demo.ps1.'
} finally {
    Pop-Location
}
