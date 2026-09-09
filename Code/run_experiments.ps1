# run_experiments.ps1
# PowerShell script for Windows
#
# Usage: .\run_experiments.ps1 <dataset> <partition> <clients> [seed|all] [strategy]
# Example: .\run_experiments.ps1 adult iid 4
# Example: .\run_experiments.ps1 adult noniid 20 42
# Example: .\run_experiments.ps1 adult iid 4 all fedprox
# Example: .\run_experiments.ps1 adult iid 4 42 both
#
# <dataset>    base dataset name: adult, cifar, imdb, celeba
# <partition>  iid | noniid  (noniid appends "noniid" to the dataset folder name)
# <clients>    number of supernodes (e.g. 4 or 20)
# [seed]       optional; defaults to "all" (loops 42, 107, 123, 2025, 9928)
# [strategy]   optional; fedavg | fedprox | both  (default: fedavg)

param(
    [Parameter(Mandatory = $true)][string]$dataset,
    [Parameter(Mandatory = $true)][string]$partition,
    [Parameter(Mandatory = $true)][int]$clients,
    [Parameter(Mandatory = $false)][string]$seed = "all",
    [Parameter(Mandatory = $false)][string]$strategy = "fedavg"
)

$ErrorActionPreference = "Stop"

switch ($partition) {
    "iid"    { $DATA = $dataset }
    "noniid" { $DATA = "${dataset}noniid" }
    default  { Write-Host "partition must be 'iid' or 'noniid'"; exit 1 }
}

$ALL_SEEDS = @(42, 107, 123, 2025, 9928)
if ($seed -eq "all") {
    $SEEDS = $ALL_SEEDS
} else {
    if ($seed -notmatch '^\d+$') {
        Write-Host "Error: seed must be an integer or 'all'"
        exit 1
    }
    $SEEDS = @([int]$seed)
}

switch ($strategy) {
    "fedavg"  { $STRATEGIES = @("fedavg") }
    "fedprox" { $STRATEGIES = @("fedprox") }
    "both"    { $STRATEGIES = @("fedavg", "fedprox") }
    default   { Write-Host "strategy must be fedavg, fedprox, or both"; exit 1 }
}
# FedProx proximal-term coefficient. Ignored for FedAvg.
$PROXIMAL_MU = 0.01

# Go to script directory so paths are consistent
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# --- Virtualenv + deps ---
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip *> $null
python -m pip install -r requirements.txt *> $null

foreach ($SEED in $SEEDS) {
    foreach ($STRAT in $STRATEGIES) {
        Write-Host "Running: Dataset=$DATA, Seed=$SEED, Clients=$clients, Strategy=$STRAT"

        # Environment variables
        $env:seed = "$SEED"
        $env:GLOBAL_SEED = "$SEED"
        $env:PYTHONHASHSEED = "$SEED"
        $env:TF_ENABLE_ONEDNN_OPTS = "0"
        $env:TF_DETERMINISTIC_OPS = "1"

        # set_num_cl.py
        python set_num_cl.py --data=$DATA --clients=$clients *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] set_num_cl.py failed for $DATA, seed $SEED, $clients clients"
            continue
        }

        if (-not (Test-Path $DATA)) {
            Write-Host "[ERROR] Could not cd to $DATA (directory does not exist)"
            continue
        }

        # proximal-mu must be 0 for fedavg; client_app keys save-dir off mu > 0
        $MU = if ($STRAT -eq "fedprox") { $PROXIMAL_MU } else { 0 }
        $RunConfig = "strategy='$STRAT' proximal-mu=$MU"
        Push-Location $DATA
        try {
            flwr run . --run-config $RunConfig
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Flower run OK for $DATA, seed $SEED, $clients clients, strategy=$STRAT"
            } else {
                Write-Host "[ERROR] Flower run failed for $DATA, seed $SEED, $clients clients, strategy=$STRAT"
            }
        }
        finally {
            Pop-Location
        }
    }
}

Write-Host "Experiments completed."
