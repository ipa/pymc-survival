#!/bin/bash

module load singularity/3.7.0

singularity exec \
    --env WORKING_DIRECTORY=/home/pymc/src/paper/experiments \
    --env PYTHONPATH=/home/pymc/src \
    --env PYCOX_DATA_DIR=/tmp/.pycox \
    --env MPLCONFIGDIR=/tmp/.matplotlib \
    --env PYTENSOR_FLAGS="base_compiledir=/tmp/.pytensor" \
    --env NUMBA_CACHE_DIR=/tmp/.numba \
    --env PMSURV_SAMPLER=pymc \
    --no-home \
    --bind $HOME/src/pymc-survival:/home/pymc/src \
    pymc-survival.sif \
    /bin/bash  


