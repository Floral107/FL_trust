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

    Write-Host "> Running experiment with seed $seedValue and method $method..."

    $env:seed = "$seedValue"  # <-- make sure Python sees the seed!

    # robustness.py is THE post-hoc driver for every method, acc/loss included.
    # (accuracy.py / loss.py were stale duplicates carrying their own sign flips
    # and are archived; routing here keeps the loss-sign convention in one place.)
    $robustness = Join-Path $PSScriptRoot "robustness.py"

    $known = @("gtg_rob", "l1o_rob",
               "l1o_adv", "gtg_adv",
               "l1o_adv_pgd", "gtg_adv_pgd",
               "l1o_adv_cw", "gtg_adv_cw",
               "l1o_fair", "gtg_fair",
               "l1o_fair_eo", "gtg_fair_eo",
               "l1o_priv", "gtg_priv",
               "l1o_acc", "gtg_acc",
               "l1o_loss", "gtg_loss")

    if ($method -notin $known) {
        Write-Host "Unknown method: $method"
        exit 1
    }

    & $venv_python $robustness --dataset=$dataset --partition=$partition --num_rounds=$rounds --seed=$seedValue --method=$method
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


