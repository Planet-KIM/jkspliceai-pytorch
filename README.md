# JKSPLICE

## Description

## Method

## Installation
The source code is currently hosted on GitHub. You can also download release version.
```sh
git clone https://github.com/jklabkaist/jksplice.git --branch develop
```

Then, install requirments in your conda environment
```sh
cd ./jksplice
conda install --file config --file ./config/requirements.txt
conda install -c nvidia cudnn
conda upgrade cudnn
```

You can install the module to site-packages of your python version and freely import by using Python Package Index (PyPI)
```sh
pip install --editable . 
```
