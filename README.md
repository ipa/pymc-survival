[![Python package](https://github.com/ipa/pymc-survival/actions/workflows/python-package.yml/badge.svg)](https://github.com/ipa/pymc-survival/actions/workflows/python-package.yml)

# PyMC Survival

## Overview

PyMC Survival is a collection of Bayesian parametric survival models written in Python using the scikit-learn API. The library is based on [PyMC](https://github.com/pymc-devs/pymc). 

## Create Docker container

docker build --build-arg GIT_ACCESS_TOKEN=[GITHUB_TOKEN] --target pymc-survival-paper -t ipaoluccimda/pymc-survival:initial-paper .

## Installation

PyMC Survival requires Python 3.8 or higher (lower versions might work but are not tested). 

Installation via pip

    pip install pmsurv


Installation from source

    pip install https://github.com/ipa/pymc-survival.git


### Dependencies

PyMC survival requires ArviZ, NumPy, pandas, PyMC, and scikit-learn. All dependencies are listed in `requirements.txt` and in `pyproject.toml`. They will be installed automatically. 

## Example

In the following two examples we assume the following basic setup

```python

    # Work in progress

```

## Documentation

An official documentation is work in progress. See example notebooks for reference.

## Citation

If you use PyMC Survival please cite: 

Paolucci, I., Lin, YM., Albuquerque Marques Silva, J. et al. Bayesian parametric models for survival prediction in medical applications. BMC Med Res Methodol 23, 250 (2023). https://doi.org/10.1186/s12874-023-02059-4

## Contributions

PyMC Survival started out of a research project. Contributions are welcome. 

## License

[MIT License](https://github.com/ipa/pymc-survival/blob/master/LICENSE)
