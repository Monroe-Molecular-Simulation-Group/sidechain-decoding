#!/bin/bash

conda activate tf_protein_env

datadir="~/Sidechain_Decoding/clean_pdbs"
savedir="~/Sidechain_Decoding/training_inputs"

# Training special case residues - GLY, N-terminal hydrogens, and NPRO
# All will be trained without conditioning on local environment

# GLY
python -m scdecode.unconditional gen_input ${datadir}/residue_modifications.json GLY -r ${datadir} -s ${savedir}/GLY > gen_inputs_GLY.out

# N-terminal
python -m scdecode.unconditional gen_input ${datadir}/residue_modifications.json Nterm -r ${datadir} -s ${savedir}/Nterm --n_terminal > gen_inputs_Nterm.out

# NPRO, an N-terminal special case (won't decode sidechain, just 2 hydrogens instead of 3)
python -m scdecode.unconditional gen_input ${datadir}/residue_modifications.json NPRO -r ${datadir} -s ${savedir}/NPRO --bat_atom_str @N,CA,C,H2,H3 > gen_inputs_NPRO.out
