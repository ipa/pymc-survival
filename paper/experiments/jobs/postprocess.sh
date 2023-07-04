#!/bin/bash

python recalc_overfit.py cox_aids
python recalc_overfit.py cox_gbcs
python recalc_overfit.py cox_pbc
python recalc_overfit.py cox_veteran
python recalc_overfit.py cox_whas

python recalc_overfit.py deepsurv_aids
python recalc_overfit.py deepsurv_gbcs
python recalc_overfit.py deepsurv_pbc
python recalc_overfit.py deepsurv_veteran
python recalc_overfit.py deepsurv_whas

python recalc_overfit.py exp_aids
python recalc_overfit.py exp_gbcs
python recalc_overfit.py exp_pbc
python recalc_overfit.py exp_veteran
python recalc_overfit.py exp_whas

python recalc_overfit.py nnwb_aids
python recalc_overfit.py nnwb_gbcs
python recalc_overfit.py nnwb_pbc
python recalc_overfit.py nnwb_veteran
python recalc_overfit.py nnwb_whas

python recalc_overfit.py rsf_aids
python recalc_overfit.py rsf_gbcs
python recalc_overfit.py rsf_pbc
python recalc_overfit.py rsf_veteran
python recalc_overfit.py rsf_whas

python recalc_overfit.py wb_aids
python recalc_overfit.py wb_gbcs
python recalc_overfit.py wb_pbc
python recalc_overfit.py wb_veteran
python recalc_overfit.py wb_whas
