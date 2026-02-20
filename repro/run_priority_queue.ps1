param(
    [ValidateSet("smoke", "full")]
    [string]$RunProfile = "smoke",

    [string]$Device = "cuda:0",

    [int]$SmokeEpochs = 2,

    [int]$SmokeBatchSize = 64,

    [switch]$DryRun,

    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$priority = @(
    "2021_truong_dfme",
    "2021_kariyappa_maze",
    "2022_sanyal_dfms",
    "2022_xie_game",
    "2024_lee_swiftthief",
    "2021_gong_inversenet"
)

$stagesByPaper = @{
    "2021_truong_dfme" = "victim_eval,attack,collect,compare"
    "2021_kariyappa_maze" = "victim_train,victim_eval,attack,collect,compare"
    "2022_sanyal_dfms" = "victim_eval,attack,collect,compare"
    "2022_xie_game" = "victim_train,victim_eval,attack,collect,compare"
    "2024_lee_swiftthief" = "victim_train,victim_eval,attack,collect,compare"
    "2021_gong_inversenet" = "victim_train,victim_eval,attack,collect,compare"
}

$queueStart = Get-Date
$total = $priority.Count

for ($i = 0; $i -lt $total; $i++) {
    $paperId = $priority[$i]
    $index = $i + 1
    $label = "$index/$total $paperId"

    $stages = $stagesByPaper[$paperId]
    Write-Host ""
    Write-Host "=== [$label] start ==="
    Write-Host "[$label] planned_stages=$stages"

    $cmdArgs = @(
        "repro/run_experiment.py",
        "run",
        "--paper-id", $paperId,
        "--profile", $RunProfile,
        "--device", $Device,
        "--smoke-epochs", "$SmokeEpochs",
        "--smoke-batch-size", "$SmokeBatchSize",
        "--live-output",
        "--stages", $stages
    )

    if ($DryRun) {
        $cmdArgs += "--dry-run"
    }

    Write-Host ("$PythonExe " + ($cmdArgs -join " "))

    $paperStart = Get-Date
    & $PythonExe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE for paper '$paperId'"
    }

    $paperElapsed = [int]((Get-Date) - $paperStart).TotalSeconds
    Write-Host "=== [$label] done (${paperElapsed}s) ==="
}

$totalElapsed = [int]((Get-Date) - $queueStart).TotalSeconds
Write-Host ""
Write-Host "Queue finished: $total papers, elapsed=${totalElapsed}s"
