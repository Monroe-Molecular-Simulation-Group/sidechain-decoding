#!/bin/bash
#SBATCH --job-name=make_train_SER
#SBATCH --output=make_train_SER.out
#SBATCH --partition tres288
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --time=144:00:00

cd $SLURM_SUBMIT_DIR

module purge
module load python/3.12-anaconda
conda activate tf_protein_env

base_dir="/home/jm217/Sidechain_Decoding"
data_dir="/storage/jm217/data_Sidechain_Decoding"
save_dir="/scratch/${SLURM_JOB_ID}/SER"

python ${base_dir}/data_io.py ${data_dir}/clean_pdbs/residue_modifications.json SER -r ${data_dir}/clean_pdbs -s $save_dir

mv $save_dir ${data_dir}/training_inputs
