#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=%x_%a.out
#SBATCH --partition agpu72
#SBATCH --qos gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH --array=7,12

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module load gcc/11.2.1
module load mkl/19.0.5
module load nvhpc/22.7
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.12-anaconda
conda activate old_tf_protein_env

datadir="/storage/jm217/data_Sidechain_Decoding/energy_min_training_inputs"
modeldir="${HOME}/Sidechain_Decoding/energy_min_trained_models"
resnames=("ALA" "ARG" "ASH" "ASN" "ASP" "CYS" "GLH" "GLN" "GLU" "HID" "HIE" "HIP" "ILE" "LEU" "LYS" "MET" "PHE" "PRO" "SER" "THR" "TRP" "TYR" "VAL")

cp -r "${datadir}/${resnames[$SLURM_ARRAY_TASK_ID]}" /local_scratch/$SLURM_JOB_ID
cd /local_scratch/$SLURM_JOB_ID
mkdir output
cd output

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
echo "Analyzing sidechain ${resnames[$SLURM_ARRAY_TASK_ID]} decoding"
python -m scdecode.analysis_tools analyze_model "${resnames[$SLURM_ARRAY_TASK_ID]}" -r "/local_scratch/${SLURM_JOB_ID}/${resnames[$SLURM_ARRAY_TASK_ID]}" -m "${modeldir}/${resnames[$SLURM_ARRAY_TASK_ID]}_decoder/${resnames[$SLURM_ARRAY_TASK_ID]}_weights.ckpt" --cg_target --h_bonds
unset LD_LIBRARY_PATH

cp * $SLURM_SUBMIT_DIR
cd ../
rm -r output ${resnames[$SLURM_ARRAY_TASK_ID]}

echo "Ended at time $(date)"
