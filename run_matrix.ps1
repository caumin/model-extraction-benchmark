$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir
try {
    & (Join-Path $scriptDir "scripts/launch/run_matrix.ps1") @args
}
finally {
    Pop-Location
}
