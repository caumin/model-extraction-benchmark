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

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")

Push-Location $repoRoot
try {
    $matrixDir = Get-EnvOrDefault -Name "MATRIX_DIR" -Default "configs/matrix"
    $device = Get-EnvOrDefault -Name "MEBENCH_DEVICE" -Default "cuda:0"
    $imagenetRoot = Get-EnvOrDefault -Name "IMAGENET_ROOT" -Default "C:/imagenet"
    $pattern = Get-EnvOrDefault -Name "MATRIX_PATTERN" -Default "*.yaml"
    $pythonBin = Get-EnvOrDefault -Name "PYTHON_BIN" -Default "python"

    [int]$maxRuns = [int](Get-EnvOrDefault -Name "MATRIX_LIMIT" -Default "0")
    [int]$setAPoolBudget = [int](Get-EnvOrDefault -Name "SET_A_POOL_BUDGET" -Default "10000")
    [int]$setASyntheticBudget = [int](Get-EnvOrDefault -Name "SET_A_SYNTHETIC_BUDGET" -Default "10000000")
    [int]$setBPoolBudget = [int](Get-EnvOrDefault -Name "SET_B_POOL_BUDGET" -Default "20000")
    [int]$setBSyntheticBudget = [int](Get-EnvOrDefault -Name "SET_B_SYNTHETIC_BUDGET" -Default "20000000")
    $poolBudgetOverride = [System.Environment]::GetEnvironmentVariable("POOL_BUDGET")
    $syntheticBudgetOverride = [System.Environment]::GetEnvironmentVariable("SYNTHETIC_BUDGET")
    [int]$generateConfigs = [int](Get-EnvOrDefault -Name "GENERATE_CONFIGS" -Default "0")
    [int]$includeBothHard = [int](Get-EnvOrDefault -Name "INCLUDE_BOTH_HARD" -Default "1")

    if ($generateConfigs -ne 0) {
        $args = @(
            "generate_configs.py",
            "--out", $matrixDir,
            "--device", $device,
            "--imagenet-root", $imagenetRoot,
            "--set-a-pool-budget", "$setAPoolBudget",
            "--set-a-synthetic-budget", "$setASyntheticBudget",
            "--set-b-pool-budget", "$setBPoolBudget",
            "--set-b-synthetic-budget", "$setBSyntheticBudget"
        )
        if (-not [string]::IsNullOrWhiteSpace($poolBudgetOverride)) {
            $args += @("--pool-budget", "$poolBudgetOverride")
        }
        if (-not [string]::IsNullOrWhiteSpace($syntheticBudgetOverride)) {
            $args += @("--synthetic-budget", "$syntheticBudgetOverride")
        }
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
    if ($configs.Count -eq 0) {
        Write-Host "No configs found in $matrixDir matching $pattern"
        exit 0
    }

    Write-Host "Running matrix from $matrixDir on $device"
    Write-Host "Total configs: $($configs.Count)"

    $attempted = 0
    $failed = 0
    foreach ($config in $configs) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($config.Name)

        $summaryPattern = Join-Path -Path "runs" -ChildPath "$name/*/seed_*/summary.json"
        if (Test-Path $summaryPattern) {
            Write-Host "[SKIP] $name"
            continue
        }

        Write-Host "[RUN ] $name"
        & $pythonBin "-m" "mebench" "run" "--config" $config.FullName "--device" $device
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[ OK ] $name"
        }
        else {
            Write-Host "[FAIL] $name"
            $failed += 1
        }

        $attempted += 1
        if ($maxRuns -gt 0 -and $attempted -ge $maxRuns) {
            Write-Host "MATRIX_LIMIT reached: $maxRuns"
            break
        }
    }

    Write-Host "Matrix run finished. attempted=$attempted failed=$failed"
    if ($failed -gt 0) {
        exit 1
    }
}
finally {
    Pop-Location
}
