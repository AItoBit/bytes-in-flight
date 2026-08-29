#!/usr/bin/env bash
# Full HBM bandwidth-ladder sweep -> $CSV (default data/run.csv).
#
# Env overrides: BIN CSV GPU FP ITERS REP
set -euo pipefail

BIN=${BIN:-./bin/bif}
CSV=${CSV:-data/run.csv}
GPU=${GPU:-0}
FP=${FP:-2147483648}          # 2 GiB working buffer
ITERS=${ITERS:-512}
REP=${REP:-20}

mkdir -p data
rm -f "$CSV"

common=(--gpu "$GPU" --footprint-bytes "$FP" --iters "$ITERS" --repeats "$REP" --csv "$CSV")

# 1) stride ladder            (vec 16 B, full-occupancy warp ladder, ILP 4)
"$BIN" "${common[@]}" --vec 16 --stride 1,2,4,8,16,32,64,128,256 --warps 0 --ilp 4

# 2) access-width ladder      (stride 1, full-occupancy warp ladder, ILP 4)
"$BIN" "${common[@]}" --vec 4,8,16 --stride 1 --warps 0 --ilp 4

# 3) Little's-Law occupancy curve  (stride 1, vec 16 B, warp ladder x ILP)
"$BIN" "${common[@]}" --vec 16 --stride 1 --warps 0 --ilp 1,2,4,8

echo "wrote $CSV"
echo "next: python analysis/plot_ladder.py $CSV && python analysis/plot_littles_law.py $CSV"
