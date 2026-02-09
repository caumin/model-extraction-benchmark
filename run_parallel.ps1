# PowerShell Parallel Execution Script for Model Extraction Benchmark
# Usage: .\run_parallel.ps1

# 1. Regenerate configs to ensure latest settings
Write-Host "Generating configurations..." -ForegroundColor Cyan
python generate_configs.py

# 2. Define the GPU/Device settings
# Assuming 1 GPU available (cuda:0). If multiple, change accordingly.
$Device = "cuda:0"

# 3. Create output directory for logs
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# 4. List all config files
$Configs = Get-ChildItem -Path "configs\matrix\*.yaml" | Select-Object -ExpandProperty FullName
$Total = $Configs.Count
Write-Host "Found $Total configurations." -ForegroundColor Green

# 5. Split configs into 4 chunks
$ChunkSize = [math]::Ceiling($Total / 4)
$Chunk1 = $Configs | Select-Object -First $ChunkSize
$Chunk2 = $Configs | Select-Object -Skip $ChunkSize -First $ChunkSize
$Chunk3 = $Configs | Select-Object -Skip ($ChunkSize * 2) -First $ChunkSize
$Chunk4 = $Configs | Select-Object -Skip ($ChunkSize * 3)

# 6. Function to generate command block
function Get-CommandBlock($Chunk, $Id) {
    $Cmds = @()
    $Cmds += "Write-Host 'Starting Worker $Id' -ForegroundColor Cyan"
    $Cmds += "$env:CUDA_VISIBLE_DEVICES='0'" # Force device visibility if needed
    foreach ($Cfg in $Chunk) {
        $Cmds += "Write-Host 'Running: $([System.IO.Path]::GetFileName($Cfg))'"
        $Cmds += "python -m mebench run --config `"$Cfg`" --device $Device"
    }
    $Cmds += "Write-Host 'Worker $Id Finished' -ForegroundColor Green"
    $Cmds += "Read-Host 'Press Enter to exit...'"
    return $Cmds -join "; "
}

# 7. Generate commands for each worker
$Cmd1 = Get-CommandBlock $Chunk1 1
$Cmd2 = Get-CommandBlock $Chunk2 2
$Cmd3 = Get-CommandBlock $Chunk3 3
$Cmd4 = Get-CommandBlock $Chunk4 4

# 8. Launch 4 separate PowerShell windows
Write-Host "Launching 4 parallel workers..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList "-NoExit", "-Command", "$Cmd1"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$Cmd2"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$Cmd3"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$Cmd4"

Write-Host "All workers launched. Check the new windows for progress." -ForegroundColor Green
