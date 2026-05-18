param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [int]$Duration = 180,

    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

$args = @(
    "-u",
    "scripts\\manual_bus_logger.py",
    "--host",
    $RemoteHost,
    "--duration",
    [string]$Duration
)

if ($OutDir -and $OutDir.Trim().Length -gt 0) {
    $args += @("--out-dir", $OutDir)
}

Write-Host ("[manual-bus-logger] repo={0}" -f $repoRoot)
Write-Host ("[manual-bus-logger] python={0}" -f $pythonExe)
Write-Host ("[manual-bus-logger] host={0} duration={1}s" -f $RemoteHost, $Duration)
if ($OutDir -and $OutDir.Trim().Length -gt 0) {
    Write-Host ("[manual-bus-logger] out={0}" -f $OutDir)
}
Write-Host "[manual-bus-logger] starting..."

& $pythonExe @args
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host ("[manual-bus-logger] process exited with code {0}" -f $exitCode)
    exit $exitCode
}

Write-Host "[manual-bus-logger] finished"
