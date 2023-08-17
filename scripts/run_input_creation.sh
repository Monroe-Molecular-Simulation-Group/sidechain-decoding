#!/bin/bash
#SBATCH --job-name=gen_inputs
#SBATCH --output=gen_inputs.out
#SBATCH --partition tres288
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --time=144:00:00

echo "Starting time is $(date)"

cd $SLURM_SUBMIT_DIR

# module purge
# module load python/3.12-anaconda
conda activate tf_protein_env

datadir="/storage/jm217/data_Sidechain_Decoding"
savedir="/scratch/${SLURM_JOB_ID}"

# Excluding GLY since no real sidechain, only hydrogen
# Also exclude caps NHE, NME, and ACE
# And cysteines forming disulfide bonds, CYX
# For each of those, will train custom model if decide we want/need one
resnames=(ALA ARG ASH ASN ASP CYM CYS GLH GLN GLU HID HIE HIP HYP ILE LEU LYN LYS MET PHE PRO SER THR TRP TYR VAL )

for r in resnames
do
    mkdir ${savedir}/${r}
    srun -n 1 -c 1 -o gen_inputs_${r}.out --exclusive python -m scdecode.data_io ${datadir}/clean_pdbs/residue_modifications.json ${r} -r ${datadir}/clean_pdbs -s ${savedir}/${r}
    sleep 2
done
wait

for r in resnames
do
    mv ${savedir}/${r} ${datadir}/training_inputs
done

echo "Ended at time $(date)"
