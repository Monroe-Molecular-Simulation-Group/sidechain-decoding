#!/bin/bash

# conda activate tf_protein_env

datadir="${HOME}/Sidechain_Decoding/energy_min_training_inputs"
modeldir="${HOME}/Sidechain_Decoding/energy_min_trained_models"

# Analyzing special case residues - GLY, N-terminal hydrogens, and NPRO

# GLY
# python -m scdecode.analysis_tools analyze_model GLY -r ${datadir}/GLY -m ${modeldir}/GLY_decoder/GLY_weights.ckpt --unconditional --h_bonds > analysis_GLY.out 2>&1

# N-terminal
python -m scdecode.analysis_tools analyze_model Nterm -r ${datadir}/Nterm -m ${modeldir}/Nterm_decoder/Nterm_weights.ckpt --unconditional --h_bonds > analysis_Nterm.out 2>&1

# NPRO, an N-terminal special case (won't decode sidechain, just 2 hydrogens instead of 3)
python -m scdecode.analysis_tools analyze_model NPRO -r ${datadir}/NPRO -m ${modeldir}/NPRO_decoder/NPRO_weights.ckpt --unconditional --h_bonds > analysis_NPRO.out 2>&1
