# Full HBM bandwidth-ladder sweep -> $Csv (default data\run.csv).
# Env overrides: BIN CSV GPU FP ITERS REP
$ErrorActionPreference = "Stop"

$Bin   = if ($env:BIN)   { $env:BIN }   else { ".\bin\bif.exe" }
$Csv   = if ($env:CSV)   { $env:CSV }   else { "data\run.csv" }
$Gpu   = if ($env:GPU)   { $env:GPU }   else { "0" }
$Fp    = if ($env:FP)    { $env:FP }    else { "2147483648" }   # 2 GiB
$Iters = if ($env:ITERS) { $env:ITERS } else { "512" }
$Rep   = if ($env:REP)   { $env:REP }   else { "20" }

New-Item -ItemType Directory -Force -Path data | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue $Csv

$common = @("--gpu", $Gpu, "--footprint-bytes", $Fp, "--iters", $Iters, "--repeats", $Rep, "--csv", $Csv)

# 1) stride ladder
& $Bin @common --vec 16 --stride 1,2,4,8,16,32,64,128,256 --warps 0 --ilp 4
if ($LASTEXITCODE -ne 0) { throw "bif failed (stride ladder)" }

# 2) access-width ladder
& $Bin @common --vec 4,8,16 --stride 1 --warps 0 --ilp 4
if ($LASTEXITCODE -ne 0) { throw "bif failed (vec ladder)" }

# 3) Little's-Law occupancy curve
& $Bin @common --vec 16 --stride 1 --warps 0 --ilp 1,2,4,8
if ($LASTEXITCODE -ne 0) { throw "bif failed (occupancy curve)" }

Write-Host "wrote $Csv"
Write-Host "next: python analysis\plot_ladder.py $Csv ; python analysis\plot_littles_law.py $Csv"
