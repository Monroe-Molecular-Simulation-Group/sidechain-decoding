#!/bin/bash
#SBATCH --job-name=cgflow
#SBATCH --output=%x.out
#SBATCH --partition agpu72
#SBATCH --qos gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=72:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module load gcc/11.2.1
module load mkl/19.0.5
module load nvhpc/22.7
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.12-anaconda
conda activate new_tf_protein_env

simdir="${HOME}/Sidechain_Decoding/simulations"

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

# Perform training
python -m scdecode.flow_model_CG train ${simdir}/1UAO.pdb ${simdir}/openmm_tremd/1UAO/1UAO_tremd.nc --h_bonds

# Generate samples
python -m scdecode.flow_model_CG sample 1UAO_CG_BAT_object.pkl 1UAO_CG_flow/1UAO_CG_flow_weights.ckpt 1UAO_CG.pdb -n 100000 --h_bonds

echo "Ended at time $(date)"
