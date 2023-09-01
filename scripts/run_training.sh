#!/bin/bash
#SBATCH --job-name=train_SER
#SBATCH --output=train_SER.out
#SBATCH --partition gpu72
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=72:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module purge
module load gcc/11.2.1
module load mkl/19.0.5
module load python/3.12-anaconda
conda activate new_tf_protein_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib

datadir="/storage/jm217/data_Sidechain_Decoding/training_inputs"

python -m scdecode.model_training SER -r ${datadir}/SER -s ./

echo "Ended at time $(date)"
