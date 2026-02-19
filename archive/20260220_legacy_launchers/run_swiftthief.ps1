$configs = Get-ChildItem "configs/matrix/*swiftthief*.yaml"

Write-Host "Found $( $configs.Count ) SwiftThief configurations."

foreach ($config in $configs) {
    $name = $config.BaseName
    if (Test-Path "runs/$name/*/seed_*/summary.json") {
        Write-Host "[SKIP] $name already completed."
        continue
    }

    Write-Host "=========================================================="
    Write-Host "Running: $name"
    Write-Host "=========================================================="

    python -m mebench run --config $config.FullName --device cuda:0

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $name failed."
    }
}
