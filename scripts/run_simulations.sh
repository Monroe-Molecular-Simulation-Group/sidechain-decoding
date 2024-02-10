#!/bin/bash
#SBATCH --job-name=protein_sims
#SBATCH --output=%x.out
#SBATCH --partition condo
#SBATCH --qos condo
#SBATCH --constraint jm217
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=72:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module purge
module load gcc/11.2.1
module load mkl/19.0.5
module load nvhpc/22.7
module load python/3.12-anaconda
conda activate openmm_sim

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

python ~/bin/sidechain-decoding/simulations/openmm_sims.py ~/Sidechain_Decoding/simulations/1UAO.pdb -n 500000000 --tremd

echo "Ended at time $(date)"
