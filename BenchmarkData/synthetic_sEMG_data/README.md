# Synthetic sEMG-data

The data included in this directory are synthetically generated data used for benchmarking purposes.

The original data set these were modelled after is the collection of surface electromyographic (sEMG) signals of arm muscles (9 motions, 8 sensory), acquired in the wake of the ITS.ML project "[Counteracting Electrode Shifts in Upper-Limb Prosthesis Control via Transfer Learning](https://github.com/itsml/itsml/blob/master/Notebooks/transfer_learning_prosthesis_control/transfer_learning_prosthesis_control.ipynb)".

Two synthetic datasets are available:

* synthetic_myo_data_HMA1.pkl: generated with the HMA1 algorithm

* synthetic_myo_data_TVAE.pkl: generated with the TVAE algorithm, based on variational autoencoders

The exact process how these data were generated is shown in the [accompanying jupyter notebook](https://github.com/itsml/itsml/tree/master/Notebooks/benchmarking_synthetic_data_generation).
