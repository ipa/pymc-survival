# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 AS pymc-cuda

RUN apt-get update -y && \
    apt-get install -y g++ gcc vim libhdf5-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get -y --force-yes install intel-mkl

RUN apt-get install -y python3 python3-distutils python3-dev wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    apt-get install -y python-is-python3

RUN pip install 'arviz>=0.12.0' \
                bambi \
                blackjax \
                h5py \
                'lifelines>=0.27.0' \
                matplotlib \
                'numpy<1.24' \
                numpyro  \
                'pandas>=1.0.0' \
                pqdm \
                "pymc>=4.3.0,<5.0" \
                'pyyaml>=6.0' \
                'scipy>=1.7'\
                'scikit-learn>=1.0.0' \
                scikit-survival \
                scikit-optimize \
                seaborn \
                tables 

# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#NOTE: JAX sampling with CUDA not yet working
RUN pip install --upgrade jax==0.3.8 -f https://storage.googleapis.com/jax-cuda_releases.html && \
    pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.8+cuda11.cudnn805-cp38-none-manylinux2014_x86_64.whl

FROM  pymc-cuda AS pymc-cuda-jupyter

RUN pip install jupyterlab \
                ipywidgets

FROM pymc-cuda-jupyter AS pymc-survival

COPY src /opt/src/pymc-survival/src
COPY pyproject.toml /opt/src/pymc-survival

RUN pip install /opt/src/pymc-survival

FROM pymc-survival AS pymc-survival-paper

RUN pip install pycox \
                torch \
                torchtuples
