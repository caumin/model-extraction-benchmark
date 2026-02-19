$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir
try {
    & (Join-Path $scriptDir "archive/20260220_legacy_launchers/run_swiftthief.ps1") @args
}
finally {
    Pop-Location
}
