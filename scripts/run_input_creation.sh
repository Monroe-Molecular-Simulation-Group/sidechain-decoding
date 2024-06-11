#!/bin/bash
#SBATCH --job-name=gen_inputs
#SBATCH --output=gen_inputs.out
#SBATCH --partition condo
#SBATCH --qos condo
#SBATCH --constraint jm217
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=144:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

module load gcc/11.2.1
module load mkl/19.0.5
module load nvhpc/22.7
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.12-anaconda
conda activate new_tf_protein_env

datadir="/storage/jm217/data_Sidechain_Decoding"
savedir="/scratch/${SLURM_JOB_ID}"

# Exclude caps NHE, NME, and ACE
# And cysteines forming disulfide bonds, CYX
# For each of those, will train custom model if decide we want/need one
# resnames=("ALA" "ARG" "ASH" "ASN" "ASP" "CYM" "CYS" "GLH" "GLN" "GLU" "GLY" "HID" "HIE" "HIP" "HYP" "ILE" "LEU" "LYN" "LYS" "MET" "PHE" "PRO" "SER" "THR" "TRP" "TYR" "VAL")
resnames=("ASH" "CYM" "GLH" "HID" "HIE" "HIP" "HYP" "LYN")

for r in ${resnames[@]}
do
    mkdir ${savedir}/${r}
    srun -n 1 -c 1 -o gen_inputs_${r}.out --exact --export=ALL python -m scdecode.data_io ${datadir}/clean_pdbs/residue_modifications.json ${r} -r ${datadir}/clean_pdbs -s ${savedir}/${r} &
    sleep 2
done
wait

for r in ${resnames[@]}
do
    mv ${savedir}/${r} ${datadir}/training_inputs
done

echo "Ended at time $(date)"
