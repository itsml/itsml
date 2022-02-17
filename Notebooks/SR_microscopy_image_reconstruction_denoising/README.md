# Disclaimer

These notebooks require gpu support and tensorflow-gpu. Without it, the code will not work.
Both were tested in Linux and Windows machines, running fine.
As gpu support for tensorflow on Mac seems to be [a complex matter](https://stackoverflow.com/questions/44744737/tensorflow-mac-os-gpu-support), the present code is not suitable to run on Mac! Sorry for any inconvenience!

# Directory structure

This directory contains two notebooks, one illustrating

# Running the notebook

Activate a virtual environment using conda.

## conda

Create a virtual environment from the provided environment file using conda:

> `conda env create -f environment.yml`

Activate the its-ml-env environment:

> `conda activate its-ml-env`

Within the activated environment, install ipykernel:

> `conda install ipykernel`

Register the its-ml-env environment as a kernel for ipython notebooks:

> `ipython kernel install --user --name=its-ml-env`

Start the notebook server:

> `jupyter notebook ./`

When you are done studying the code, exit the virtual environment, remove it, and uninstall it as a kernel:

> `conda deactivate`

> `conda remove --name its-ml-env --all`

> `jupyter kernelspec remove its-ml-env`
