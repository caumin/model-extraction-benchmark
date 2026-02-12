$ErrorActionPreference = "Stop"

function Get-EnvOrDefault {
    param(
        [string]$Name,
        [string]$Default
    )
    $value = [System.Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

$matrixDir = Get-EnvOrDefault -Name "MATRIX_DIR" -Default "configs/matrix"
$device = Get-EnvOrDefault -Name "MEBENCH_DEVICE" -Default "cuda:0"
$pattern = Get-EnvOrDefault -Name "MATRIX_PATTERN" -Default "*.yaml"
$pythonBin = Get-EnvOrDefault -Name "PYTHON_BIN" -Default "python"

[int]$maxRuns = [int](Get-EnvOrDefault -Name "MATRIX_LIMIT" -Default "0")
[int]$poolBudget = [int](Get-EnvOrDefault -Name "POOL_BUDGET" -Default "20000")
[int]$syntheticBudget = [int](Get-EnvOrDefault -Name "SYNTHETIC_BUDGET" -Default "20000000")
[int]$generateConfigs = [int](Get-EnvOrDefault -Name "GENERATE_CONFIGS" -Default "1")
[int]$includeBothHard = [int](Get-EnvOrDefault -Name "INCLUDE_BOTH_HARD" -Default "1")

if ($generateConfigs -ne 0) {
    $args = @(
        "generate_configs.py",
        "--out", $matrixDir,
        "--device", $device,
        "--pool-budget", "$poolBudget",
        "--synthetic-budget", "$syntheticBudget"
    )
    if ($includeBothHard -ne 0) {
        $args += "--include-both-hard"
    }

    & $pythonBin @args
    if ($LASTEXITCODE -ne 0) {
        throw "Config generation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $matrixDir)) {
    throw "Matrix directory not found: $matrixDir"
}

$configs = Get-ChildItem -Path $matrixDir -Filter $pattern -File | Sort-Object Name

Write-Host "Starting Experimental Matrix Execution..."
Write-Host "Total experiments: $($configs.Count)"

$count = 0
foreach ($config in $configs) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($config.Name)

    $summaryPattern = Join-Path -Path "runs" -ChildPath "$name/*/seed_*/summary.json"
    if (Test-Path $summaryPattern) {
        Write-Host "[SKIP] $name already completed."
        continue
    }

    Write-Host "=========================================================="
    Write-Host "Running: $name"
    Write-Host "=========================================================="

    & $pythonBin "-m" "mebench" "run" "--config" $config.FullName "--device" $device
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $name failed."
    }

    $count += 1
    if ($maxRuns -gt 0 -and $count -ge $maxRuns) {
        Write-Host "[INFO] MATRIX_LIMIT reached ($maxRuns)."
        break
    }
}

Write-Host "Matrix execution complete."
