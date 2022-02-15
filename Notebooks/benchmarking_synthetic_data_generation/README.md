# Running the notebook

* Note: this is a newer notebook and has been tested with python 3.9 instead of python 3.7. *

## Installing graphviz

This notebook requires graphviz, an open source graph visualization software.
Please make sure to install it according to your system:

https://graphviz.gitlab.io/download/

When graphviz is installed, you may continue by activating a virtual environment using conda.

## conda

Create a virtual environment using conda:

> `conda create -n its-ml python=3.9`

Activate the its-ml python environment:

> `conda activate its-ml`

Install the necessary packages with conda:

> `pip install -r requirements.txt`

Register the its-ml environment as a kernel for ipython notebooks:

> `python -m ipykernel install --user --name=its-ml`

Start the notebook server:

> `jupyter notebook ./`

When you are done studying the code, exit the virtual environment, remove it, and uninstall it as a kernel:

> `conda deactivate`

> `conda env remove -n its-ml`

> `jupyter kernelspec remove its-ml`
