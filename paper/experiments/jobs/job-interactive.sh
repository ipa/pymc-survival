#!/bin/bash

module load singularity/3.7.0

singularity exec \
    --env WORKING_DIRECTORY=/home/pymc/src/examples/experiments \
    --env PYTHONPATH=/home/pymc/src \
    --env AESARA_FLAGS="base_compiledir=/tmp/.aesara" \
    --no-home \
    --bind $HOME/src/pymc-survival:/home/pymc/src \
    modelling_pymc4-jupyter-latest.sif \
    /bin/bash  


