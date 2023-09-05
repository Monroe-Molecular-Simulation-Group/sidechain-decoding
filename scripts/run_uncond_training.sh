#!/bin/bash

conda activate tf_protein_env

datadir="~/Sidechain_Decoding/training_inputs"
savedir="~/Sidechain_Decoding/training_input"

# Training special case residues - GLY, N-terminal hydrogens, and NPRO
# All will be trained without conditioning on local environment

# GLY
python -m scdecode.unconditional train GLY -r ${datadir}/GLY -s ./ > train_GLY.out

# N-terminal
python -m scdecode.unconditional train Nterm -r ${datadir}/Nterm -s ./ > train_Nterm.out

# NPRO, an N-terminal special case (won't decode sidechain, just 2 hydrogens instead of 3)
python -m scdecode.unconditional train NPRO -r ${datadir}/NPRO -s ./ > train_NPRO.out
