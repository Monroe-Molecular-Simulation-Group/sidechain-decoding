#!/bin/bash

# conda activate tf_protein_env

datadir="${HOME}/Sidechain_Decoding/energy_min_training_inputs"

# Training special case residues - GLY, N-terminal hydrogens, and NPRO
# All will be trained without conditioning on local environment

# GLY
python -m scdecode.unconditional train GLY -r ${datadir}/GLY -s ./ > train_GLY.out 2>&1

# N-terminal
python -m scdecode.unconditional train Nterm -r ${datadir}/Nterm -s ./ > train_Nterm.out 2>&1

# NPRO, an N-terminal special case (won't decode sidechain, just 2 hydrogens instead of 3)
python -m scdecode.unconditional train NPRO -r ${datadir}/NPRO -s ./ > train_NPRO.out 2>&1
