#!/bin/bash
#SBATCH --job-name=full_decode
#SBATCH --output=%x.out
#SBATCH --partition condo
#SBATCH --qos condo
#SBATCH --constraint jm217
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module purge
module load gcc/11.2.1
module load mkl/19.0.5
module load nvhpc/22.7
module load python/3.12-anaconda
conda activate new_tf_protein_env

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

inputdir="/storage/jm217/data_Sidechain_Decoding/energy_min_training_inputs"
modeldir="${HOME}/Sidechain_Decoding/logprob_trained_models"

# python -m scdecode.full_protein_decoding trajectory ~/Sidechain_Decoding/simulations/1UAO.pdb ~/Sidechain_Decoding/simulations/openmm_restrained/1UAO_restrained.nc -b $inputdir -m $modeldir

python -m scdecode.full_protein_decoding dataset ~/Sidechain_Decoding/Jones_PDB_test_pdbs -b $inputdir -m $modeldir

echo "Ended at time $(date)"
