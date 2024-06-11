#!/bin/bash
#SBATCH --job-name=protein_sims
#SBATCH --output=%x.out
#SBATCH --partition condo
#SBATCH --qos condo
#SBATCH --constraint jm217
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=120:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module load gcc/11.2.1
module load mkl/19.0.5
module load nvhpc/22.7
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.12-anaconda
conda activate openmm_sim

cd /scratch/${SLURM_JOB_ID}
mkdir output
cd output

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
python ~/bin/sidechain-decoding/simulations/openmm_sims.py ~/Sidechain_Decoding/simulations/1UAO.pdb -n 500000000 -o './' --tremd --restrain
unset LD_LIBRARY_PATH

mv * $SLURM_SUBMIT_DIR
cd ../
rm -r output

echo "Ended at time $(date)"
