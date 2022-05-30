# Uncertainty Quantification

Due to the steadily increasing relevance of machine learning for practical applications,
many of which are coming with safety requirements, the notion of uncertainty has received
increasing attention in machine learning research in the last couple of years.
In particular, the idea of distinguishing between two important types of uncertainty,
often refereed to as aleatoric(randomness) and epistemic(lack of information),
has recently been studied in the setting of supervised learning.
In this work, we propose to quantify these uncertainties with random forests.
More specifically, we show a Bayesian approach for measuring the learner’s aleatoric
and epistemic uncertainty in a prediction can be instantiated with decision trees and random forests as learning algorithms
in a classification setting.

To run the code please run the "uncQ" notebook.

This work has been part of the ITSML project under the supervision of Prof. Dr. Eyke Hüllermeier.

# Running the notebook

* Note: this is a newer notebook and has been tested with python 3.9 instead of python 3.7. *

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
