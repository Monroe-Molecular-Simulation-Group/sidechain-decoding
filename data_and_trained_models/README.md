Data and Trained Models
=========================

This directory contains data related to building data sets and training models, as well as the trained models themselves.

The directory "query_pdb" contains the query of the PDB and post-processing to obtain a list of unique PDB IDs to clean up for training.
The resulting list "unique_pdbids.txt" can be supplied to "prep_pdb.py" to prepare clean pdb structures.
These can then be passed to "data_io.py" to generate training data sets in Tensorflow's data format.

The directories "energy_min_models" and "traj_trained_models" contain both pickled MDAnalysis BAT objects, which are necessary for much of the analysis, as well as directories of models trained in Tensorflow checkpoint format.
Note that only parameters of models are saved.
To use trained models, a model must first be be built using the `build_model` method of "model_training.py" (or the equivalent method in "unconditional.py" for models not conditioned on their local environment, like for GLY).
This is specific to each residue type and requires information on the number of atoms and number of bonds involving hydrogens in the sidechain.
Once the model is created and built using input data of the correct shape, the weights can be loaded with the `load_weights` method of of the Tensorflow Keras model class.
An example is provided in the `analyze_model` method in "analysis_tools.py".

