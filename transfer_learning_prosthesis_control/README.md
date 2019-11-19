Activate the its-ml python envrionment:

> `source activate its-ml`

Install the necessary packages with pip:

> `pip install numpy scipy && pip install -r req.txt`

Register the its-ml environment as a kernel for ipython notebooks:

> `python -m ipykernel install --user --name=its-ml`

Start the notebook server:

> `jupyter notebook ./`