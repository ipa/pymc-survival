# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 AS pymc-cuda

RUN apt-get update -y && \
    apt-get install -y g++ gcc vim libhdf5-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get -y --force-yes install intel-mkl

RUN apt-get install -y python3 python3-distutils python3-dev wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    apt-get install -y python-is-python3

RUN pip install arviz \
                bambi \
                blackjax \
                h5py \
                lifelines \
                matplotlib \
                numpy \
                numpyro  \
                pandas \
                pqdm \
                pymc \
                pyyaml \
                scipy \
                scikit-learn \
                scikit-survival \
                scikit-optimize \
                seaborn \
                tables 

# RUN pip install --upgrade jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

FROM pymc-cuda AS pymc-cuda-jupyter

RUN pip install jupyterlab \
                ipywidgets
