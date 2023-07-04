#!/bin/bash

python collect_retrain_summary.py results_retrain/exp_aids/ 0 75 1 10
python collect_retrain_summary.py results_retrain/wb_aids/ 0 75 1 10
python collect_retrain_summary.py results_retrain/deepsurv_aids/ 0 75 1 10

python collect_retrain_summary.py results_retrain/exp_gbcs/ 0 75 1 6
python collect_retrain_summary.py results_retrain/wb_gbcs/ 0 75 1 6
python collect_retrain_summary.py results_retrain/deepsurv_gbcs/ 0 75 1 6

python collect_retrain_summary.py results_retrain/exp_pbc/ 0 75 1 6
python collect_retrain_summary.py results_retrain/wb_pbc/ 0 75 1 6
python collect_retrain_summary.py results_retrain/deepsurv_pbc/ 0 75 1 6

python collect_retrain_summary.py results_retrain/exp_whas/ 0 75 1 6
python collect_retrain_summary.py results_retrain/wb_whas/ 0 75 1 6
python collect_retrain_summary.py results_retrain/deepsurv_whas/ 0 75 1 6
