param(
    [Parameter(Mandatory = $true)][string]$dataset,
    [Parameter(Mandatory = $true)][string]$partition,
    [Parameter(Mandatory = $true)][int]$rounds,
    [Parameter(Mandatory = $true)][string]$method,
    [Parameter(Mandatory = $true)][string]$seed
)

$ErrorActionPreference = "Stop"

# Create and prepare virtual environment
if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$venv_python = ".\.venv\Scripts\python.exe"

Write-Host "Upgrading pip and installing requirements..."
& $venv_python -m pip install --upgrade pip
& $venv_python -m pip install -r requirements.txt

$seeds = @(42, 107, 123, 2025, 9928)

function Run-Experiment {
    param ([int]$seedValue)

    Write-Host "▶ Running experiment with seed $seedValue and method $method..."

    $env:seed = "$seedValue"  # <-- make sure Python sees the seed!

    $script_base = "C:\BME\7. felev\szakdoga\FLRobustness"
    $robustness = Join-Path $script_base "robustness.py"
    $accuracy   = Join-Path $script_base "accuracy.py"
    $loss       = Join-Path $script_base "loss.py"

    if ($method -in @("gtg_rob", "l1o_rob",
                       "l1o_adv", "gtg_adv",
                       "l1o_adv_pgd", "gtg_adv_pgd",
                       "l1o_adv_cw", "gtg_adv_cw",
                       "l1o_fair", "gtg_fair",
                       "l1o_fair_eo", "gtg_fair_eo")) {
        & $venv_python $robustness --dataset=$dataset --partition=$partition --num_rounds=$rounds --seed=$seedValue --method=$method
    }
    elseif ($method -eq "gtg_acc") {
        & $venv_python $accuracy --dataset=$dataset --partition=$partition --num_rounds=$rounds --seed=$seedValue --method=$method
    }
    elseif ($method -in @("gtg_loss", "l1o_loss")) {
        & $venv_python $loss --dataset=$dataset --partition=$partition --num_rounds=$rounds --seed=$seedValue --method=$method
    }
    else {
        Write-Host " Unknown method: $method"
        exit 1
    }
}

if ($seed -eq "all") {
    foreach ($s in $seeds) {
        Write-Host "`n---------------------------------------------"
        Write-Host " Running with seed $s"
        Write-Host "---------------------------------------------`n"
	
	

        Run-Experiment -seedValue $s
        Write-Host " Finished seed $s"
    }
}
else {
    if ($seed -notmatch '^\d+$') {
        Write-Host " Error: SEED must be an integer or 'all'"
        exit 1
    }
    Write-Host "`n---------------------------------------------"
    Write-Host " Running with seed $seed"
    Write-Host "---------------------------------------------`n"

    $seed = [int]$seed

    Run-Experiment -seedValue $seed
    Write-Host " Finished seed $seed"
}


