#!/bin/bash
#SBATCH --job-name=full_decode
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
conda activate old_tf_protein_env

inputdir="/storage/jm217/data_Sidechain_Decoding/energy_min_training_inputs"
modeldir="${HOME}/Sidechain_Decoding/energy_min_trained_models"

cd /local_scratch/$SLURM_JOB_ID
mkdir output
cd output

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

# python -m scdecode.full_protein_decoding trajectory ~/Sidechain_Decoding/simulations/1UAO.pdb ~/Sidechain_Decoding/simulations/openmm_restrained/1UAO/1UAO_restrained.nc -b $inputdir -m $modeldir

python -m scdecode.full_protein_decoding trajectory ~/Sidechain_Decoding/simulations/1UAO.pdb ~/Sidechain_Decoding/simulations/openmm_restrained_tremd/1UAO/1UAO_restrained_tremd.nc -b $inputdir -m $modeldir

# python -m scdecode.full_protein_decoding trajectory ~/Sidechain_Decoding/simulations/1UAO.pdb ~/Sidechain_Decoding/simulations/openmm_tremd/1UAO/1UAO_tremd.nc -b $inputdir -m $modeldir

# python -m scdecode.full_protein_decoding dataset ~/Sidechain_Decoding/Jones_PDB_test_pdbs/energy_min_pdbs -b $inputdir -m $modeldir

# python -m scdecode.full_protein_decoding decode ~/Sidechain_Decoding/simulations/1UAO.pdb ~/Sidechain_Decoding/simulations/flow_CG/gen_CG_traj_1UAO_CG.pdb ~/Sidechain_Decoding/simulations/flow_CG/gen_CG_traj_1UAO_CG.pdb -b $inputdir -m $modeldir -n 5

# python -m scdecode.full_protein_decoding decode ~/Sidechain_Decoding/simulations/1UAO.pdb ~/Sidechain_Decoding/simulations/pyrosetta/1UAO/cg_traj_1UAO.pdb ~/Sidechain_Decoding/simulations/pyrosetta/1UAO/cg_traj_1UAO.pdb -b $inputdir -m $modeldir -n 5

unset LD_LIBRARY_PATH

cp * $SLURM_SUBMIT_DIR
cd ../
rm -r output 

echo "Ended at time $(date)"
