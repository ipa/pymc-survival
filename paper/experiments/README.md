# Experiments

This folder contains the code for the experiments. Each experiment has to be run individually for each dataset. This is split up to make it easier to distribute the computations on a cluster. 

**Warning:** These scripts are computationally very expensive. It's best to run them on a high performance cluster.

**Note:** To not waste computation time on our cluster, a sample size calculation was performed before running the experiments. See analysis/sample-size-calc.qmd (Requires R, RStudio and Quarto).

All experiments are run inside a container (e.g. Docker, Singularity). To run the docker container simply run:

    docker run -it --mount type=bind,source=path/to/pymc-survival/paper,target=/root/pymc-survival [image_name] /bin/bash

### Models

* pmsurv_exponential - Bayesian Exponential model
* pmsurv_weibull_linear - Bayesian Weibull model
* pmsurv_nn - Bayesian Neural Network model
* cox - Cox Proportional Hazards
* rsf - Random Survival Forest
* deepsurv - DeepSurv

### Datasets

* actg - AIDS Clinical Trials Group Study (ACTG)
* gbcs - German Breast Cancer Study data (GBCS)
* veteran - Veteran lung cancer (Veteran)
* whas - Worcester Heart attack study (WHAS)
* pbc - Primary Biliary Cirrhosis (PBC)

## Comparision with other models

To run the individual experiments per model and dataset use the following command:

    python run_experiment.py [model] [experiment_name] /path/to/dataset --runs [n_runs] --jobs [n_jobs] --n-iter [n_iter]

Once this script is finished, the overfitting can be re-calculated using

    python recalc_overfit.py [experiment_name]

The results will be stored in the folder `results`

## Re-training

To run the individual experiments for re-training per model and dataset use the following command:

    python run_retrain_experiment.py [model] [experiment_name] /path/to/dataset --runs [n_runs] --n-partitions [n_partitions]

The results will be stored in the folder `results_retrain`

## Analysis

The analysis code is hosted in the directory `paper/analysis`