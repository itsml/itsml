# Disclaimer

This demo is taken from the GitHub repository of the full DeepView toolbox. 
You can access the full toolbox [here](https://github.com/LucaHermes/DeepView).
Below are step-by-step instuctions on how to get access and run the DeepView demo notebook.

# Getting and running the notebook

Activate a virtual environment using either virtualenv or conda.

## virtualenv

Create a virtual environment using virtualenv:

> `virtualenv -p python3 its-ml`

Activate the its-ml python environment:

> `source its-ml/bin/activate`

Clone the DeepView repository:

> `git clone https://github.com/LucaHermes/DeepView.git`

Install the necessary packages with pip:

> `pip3 install -r requirements.txt`
> `pip3 install -r DeepView/requirements.txt`

Register the its-ml environment as a kernel for ipython notebooks:

> `python3 -m ipykernel install --user --name=its-ml`

Start the notebook server:

> `jupyter notebook ./`

Go to DeepView/DeepView Demo.ipynb.

Note that you might need to select the correct kernel 'its-ml' from the corresponding drop down menu.

When you are done studying the code, exit the virtual environment, uninstall it as a kernel, and remove the corresponding folder:

> `deactivate`

> `jupyter kernelspec remove its-ml`

> `rm -rf its-ml`

If you want to remove the DeepView toolbox after trying it out, don't forget to:

> `rm -rf DeepView`

## conda

Create a virtual environment using conda:

> `conda create -n its-ml python=3.7`

Activate the its-ml python environment:

> `conda activate its-ml`

Clone the DeepView repository:

> `git clone https://github.com/LucaHermes/DeepView.git`

Install the necessary packages with conda:

> `pip3 install -r requirements.txt`
> `pip3 install -r DeepView/requirements.txt`

Register the its-ml environment as a kernel for ipython notebooks:

> `python3 -m ipykernel install --user --name=its-ml`

Start the notebook server:

> `jupyter notebook ./`

Go to DeepView/DeepView Demo.ipynb.

Note that you might need to select the correct kernel 'its-ml' from the corresponding drop down menu.

When you are done studying the code, exit the virtual environment, remove it, and uninstall it as a kernel:

> `conda deactivate`

> `conda env remove -n its-ml`

> `jupyter kernelspec remove its-ml`

If you want to remove the DeepView toolbox after trying it out, don't forget to:

> `rm -rf DeepView`