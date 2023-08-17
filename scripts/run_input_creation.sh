#!/bin/bash
#SBATCH --job-name=gen_inputs_SER
#SBATCH --output=gen_inputs_SER.out
#SBATCH --partition tres288
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --time=144:00:00

cd $SLURM_SUBMIT_DIR

# module purge
# module load python/3.12-anaconda
conda activate tf_protein_env

data_dir="/storage/jm217/data_Sidechain_Decoding"
save_dir="/scratch/${SLURM_JOB_ID}/SER"

python -m scdecode.data_io ${data_dir}/clean_pdbs/residue_modifications.json SER -r ${data_dir}/clean_pdbs -s $save_dir

mv $save_dir ${data_dir}/training_inputs
