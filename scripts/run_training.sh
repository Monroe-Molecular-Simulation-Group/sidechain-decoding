#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=%x_%a.out
#SBATCH --partition agpu72
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=72:00:00
#SBATCH --array=0-26

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module purge
module load gcc/11.2.1
module load mkl/19.0.5
module load python/3.12-anaconda
conda activate new_tf_protein_env

datadir="/storage/jm217/data_Sidechain_Decoding/energy_min_training_inputs"
resnames=("ALA" "ARG" "ASH" "ASN" "ASP" "CYM" "CYS" "GLH" "GLN" "GLU" "GLY" "HID" "HIE" "HIP" "HYP" "ILE" "LEU" "LYN" "LYS" "MET" "PHE" "PRO" "SER" "THR" "TRP" "TYR" "VAL")

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

echo "Training sidechain ${resnames[$SLURM_ARRAY_TASK_ID]}"

python -m scdecode.model_training "${resnames[$SLURM_ARRAY_TASK_ID]}"  -r "${datadir}/${resnames[$SLURM_ARRAY_TASK_ID]}" -s ./ --cg_target --h_bonds

echo "Ended at time $(date)"
