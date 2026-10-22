#!/bin/sh
# resumable queue: skips runs whose result CSV already exists
run() { t=$1; k=$2; s=$3
  f=output/rolling_nn_${t}_${k}_sight_month_s${s}.csv
  [ -f "$f" ] || python3 run_rolling_nn.py $t $k sight_month $s > logs/nn_${t}_${k}_sight_month_s${s}.log 2>&1; }
run any convlstm 1; run any convlstm 2; run breeding convlstm 0
run any celllstm 1; run any celllstm 2; run breeding convlstm 1; run breeding convlstm 2
run breeding celllstm 1; run breeding celllstm 2
