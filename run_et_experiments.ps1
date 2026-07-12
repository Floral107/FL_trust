# run_et_experiments.ps1
# PowerShell script for Windows. Mirrors run_et_experiments.sh.
#
# Usage:  .\run_et_experiments.ps1 [-dataset <d>] [-clients <n>] [-et_mode <m>] [-seed <s>]
# All params optional; default "all" loops over every combination.
#
# Examples:
#   .\run_et_experiments.ps1                                      # full grid
#   .\run_et_experiments.ps1 -dataset adultnoniid -clients 4 -et_mode fair -seed 42
#   .\run_et_experiments.ps1 -dataset adultnoniid                # all clients/modes/seeds for adult-noniid

param(
    [string]$dataset = "all",
    [string]$clients = "all",
    [string]$et_mode = "all",
    [string]$seed = "all"
)

$ErrorActionPreference = "Stop"

$DATASETS_ALL = @("adultnoniid", "celebanoniid", "imdbnoniid")
$CLIENTS_ALL  = @(4, 20)
$ET_MODES_ALL = @("fair", "adv", "dp")
$SEEDS_ALL    = @(42, 107, 123, 2025, 9928)

if ($dataset -eq "all") { $DATASETS = $DATASETS_ALL } else { $DATASETS = @($dataset) }
if ($clients -eq "all") { $CLIENT_LIST = $CLIENTS_ALL } else { $CLIENT_LIST = @([int]$clients) }
if ($et_mode -eq "all") { $ET_MODES = $ET_MODES_ALL } else { $ET_MODES = @($et_mode) }
if ($seed    -eq "all") { $SEEDS = $SEEDS_ALL } else { $SEEDS = @([int]$seed) }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
}

foreach ($DATA in $DATASETS) {
    foreach ($NC in $CLIENT_LIST) {
        foreach ($ET in $ET_MODES) {
            if ($DATA -eq "imdbnoniid" -and $ET -eq "fair") {
                Write-Host "[SKIP] imdbnoniid + fair (no sensitive attribute)"
                continue
            }
            foreach ($SEED in $SEEDS) {
                Write-Host "================================================================"
                Write-Host "[ET] dataset=$DATA  clients=$NC  et_mode=$ET  seed=$SEED"
                Write-Host "================================================================"

                $env:seed = "$SEED"
                $env:GLOBAL_SEED = "$SEED"
                $env:PYTHONHASHSEED = "$SEED"
                $env:TF_ENABLE_ONEDNN_OPTS = "0"
                $env:TF_DETERMINISTIC_OPS = "1"

                python set_num_cl.py --data=$DATA --clients=$NC *> $null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[ERROR] set_num_cl.py failed for $DATA, seed $SEED, $NC clients"
                    continue
                }

                if (-not (Test-Path $DATA)) {
                    Write-Host "[ERROR] dataset directory '$DATA' does not exist"
                    continue
                }

                $RunConfig = "strategy='fedavg' et-mode='$ET'"
                Push-Location $DATA
                try {
                    flwr run . --run-config $RunConfig
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "[OK] $DATA / nc=$NC / et=$ET / seed=$SEED"
                    } else {
                        Write-Host "[ERROR] flwr run failed: $DATA / nc=$NC / et=$ET / seed=$SEED"
                    }
                }
                finally { Pop-Location }
            }
        }
    }
}

Write-Host "================================================================"
Write-Host "ET experiments completed."
Write-Host "Next: python reweight_eval.py --et-mode fair --dataset adultnoniid --num_clients 4"
