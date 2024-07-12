# Starter

## Set Up

Create the conda environment. I *highly* recommend using [Mamba](https://mamba.readthedocs.io/en/latest/installation.html), which is [faster](https://blog.hpc.qmul.ac.uk/mamba.html#:~:text=mamba%20is%20a%20re%2Dimplementation,Red%20Hat%2C%20Fedora%20and%20OpenSUSE), but if you insist on Conda, then replace `mamba` commands below with `conda`
```sh
mamba env create -f isaac.yml
mamba activate isaac
```

Then, in the repo's root, install the repo as a pip package
```sh
pip install -e .
```

## Downloads

## Rollout Controller
