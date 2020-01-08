# Running the notebook

Activate a virtual environment using either virtualenv or conda.

## virtualenv

Create a virtual environment using virtualenv:

> `virtualenv -p python3 its-ml`

Activate the its-ml python environment:

> `source its-ml/bin/activate`

Install the necessary packages with pip:

> `pip3 install numpy scipy jupyter && pip3 install -r requirements.txt`

Register the its-ml environment as a kernel for ipython notebooks:

> `ipython kernel install --user --name=its-ml`

Start the notebook server:

> `jupyter notebook ./`

When you are done studying the code, exit the virtual environment, uninstall it as a kernel, and remove the corresponding folder:

> `deactivate`

> `jupyter kernelspec remove its-ml`

> `rm -rf its-ml`

## conda

Create a virtual environment using conda:

> `conda create -n conda_its-ml python=3.7`

Activate the its-ml python environment:

> `conda activate conda_its-ml`

Install the necessary packages with conda:

> `pip install numpy scipy jupyter && pip install -r requirements.txt`

Register the its-ml environment as a kernel for ipython notebooks:

> `python3 -m ipykernel install --user --name=its-ml`

Start the notebook server:

> `jupyter notebook ./`

When you are done studying the code, exit the virtual environment, remove it, and uninstall it as a kernel:

> `conda deactivate`

> `conda env remove -n its-ml`

> `jupyter kernelspec remove its-ml`
