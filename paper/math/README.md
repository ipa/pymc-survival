# Math

This folder contains notebooks with some of the formulas for the manuscript formatted using SciPy and SymPy

## Docker

To run the notebook use this container. Replace [src] with the source directory of the repository.

    docker run -it --rm --mount type=bind,source=[src]\paper\math,target=/home/jovyan/math -p 8888:8888 jupyter/scipy-notebook:python-3.11 /bin/bash

