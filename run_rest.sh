#!/bin/bash
# sequential driver for the remaining studies
cd /home/claude/code/qbn
python3 -I run_pc.py results/pc_metr.pkl > results/pc.log 2>&1
python3 -I run_full.py metr-la results/full_metr.pkl 12 > results/full_metr.log 2>&1
python3 -I run_main.py --dataset pems-bay --models M1,M2,M3,M4,C1,C2,C4 --out results/main_pems.pkl > results/main_pems.log 2>&1
python3 -I run_main.py --gamma 0.8 --models M1,M4,C1,C2 --out results/sens_g080.pkl > results/sens_g080.log 2>&1
python3 -I run_main.py --gamma 0.95 --models M1,M4,C1,C2 --out results/sens_g095.pkl > results/sens_g095.log 2>&1
python3 -I run_main.py --T 864 --models M1,M4,C1,C2 --best_from results/main_metr.pkl --out results/sens_T864.pkl > results/sens_T864.log 2>&1
python3 -I run_main.py --T 2016 --models M1,M4,C1,C2 --best_from results/main_metr.pkl --seeds 0 --out results/sens_T2016.pkl > results/sens_T2016.log 2>&1
python3 -I run_full.py pems-bay results/full_pems.pkl 12 > results/full_pems.log 2>&1
echo ALLDONE > results/rest.done
