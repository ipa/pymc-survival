# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 AS pymc-cuda

FROM python:3.11-bullseye as pymc

RUN apt-get update -y && \
    apt-get install -y g++ gcc vim wget libhdf5-dev

COPY docker/non-free.sources /etc/apt/sources.list.d/non-free.sources

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install intel-mkl

RUN apt-get install -y python3-distutils python3-dev 

RUN pip install 'arviz>=0.12.0' \
                bambi \
                # blackjax \
                h5py \
                'lifelines>=0.27.0' \
                matplotlib \
                'numpy<1.24' \
                # numpyro  \
                'pandas>=1.0.0' \
                pqdm \
                "pymc>=5.1.2" \
                'pyyaml>=6.0' \
                'scipy>=1.7'\
                'scikit-learn>=1.0.0' \
                scikit-survival \
                scikit-optimize \
                seaborn \
                tables 

# RUN pip install --upgrade "jax[cpu]"

# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#NOTE: JAX sampling with CUDA not yet working
# RUN pip install --upgrade jax==0.3.8 -f https://storage.googleapis.com/jax-cuda_releases.html && \
#     pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl

FROM pymc as pymc-nutpie

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && \
    apt-get install -y curl python3-venv && \
    curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    pip install maturin && \
    pip install numba && \
    cd /tmp &&\
    wget https://github.com/pymc-devs/nutpie/archive/refs/tags/v0.5.1.tar.gz && \
    tar -xf v0.5.1.tar.gz && \
    cd nutpie-0.5.1 && \
    source $HOME/.cargo/env && \
    python -m venv .venv && \
    maturin build --release && \
    pip install target/wheels/nutpie-0.5.1-cp311-cp311-manylinux_2_31_x86_64.whl && \
    rm -rf /tmp/nutpie-0.5.1

# FROM  pymc-cuda AS pymc-cuda-jupyter

# RUN pip install jupyterlab \
#                 ipywidgets

FROM pymc-nutpie AS pymc-survival

COPY src /opt/src/pymc-survival/src
COPY pyproject.toml /opt/src/pymc-survival

RUN pip install /opt/src/pymc-survival

FROM pymc-nutpie AS pymc-survival-paper

RUN pip install torch torchtuples && \
    pip install pycox

COPY src /opt/src/pymc-survival/src
COPY pyproject.toml /opt/src/pymc-survival

RUN pip install /opt/src/pymc-survival
