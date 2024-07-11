"""
Builds a CG model based on clustering, samples from it, and saves a CG trajectory.
"""

import sys, os
# os.environ["OMP_NUM_THREADS"] = "8"

import argparse

import numpy as np

import parmed as pmd
import MDAnalysis as mda
import mdtraj

from scipy import cluster as spcluster
from scipy import spatial as spspatial

import tensorflow_probability as tfp

from scdecode import analysis_tools


def main(pdb_file, traj_file, out_name=None, sim_stride=10, n_samples=100000):
    """
    Performs clustering analysis and generates a CG model trajectory based on sampling
    from the clustered configurations. Sampling occurs based on two models, one with
    weights of clusters proportional to their population, the other with uniform weights.
    """

    # Create an output file name
    if out_name is None:
        out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    # Load the all-atom trajectory
    traj = mdtraj.load(traj_file, top=pdb_file, stride=sim_stride)
    
    # Identify alpha carbons - will only use these for RMSD distances and clustering
    ca_inds = traj.top.select("name CA")

    # Calculate the distance matrix
    dist_mat = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        if i % 1000 == 0:
            print(i)
        # Superposing all frames to the current frame i, then computing RMSD
        traj.superpose(traj, frame=i, atom_indices=ca_inds)
        dist_mat[i] = mdtraj.rmsd(traj, traj, frame=i, atom_indices=ca_inds)

    # Perform clustering using scipy
    # First just building graph (hierarchy), then flattening into clusters based on distance
    reduced_dist_mat = spspatial.distance.squareform(dist_mat, checks=False)
    linkage = spcluster.hierarchy.linkage(reduced_dist_mat, method='average')
    clust_inds = spcluster.hierarchy.fcluster(linkage, 0.20, criterion='distance')

    # Parse through assignment of clusters
    unique_clusts = np.sort(np.unique(clust_inds))
    n_counts = np.zeros_like(unique_clusts)
    per_cluster_inds = []
    for i, c in enumerate(unique_clusts):
        this_bool = (clust_inds == c)
        n_counts[i] = np.sum(this_bool)
        per_cluster_inds.append(np.arange(len(clust_inds))[this_bool])

    # Create a categorical distribution for choosing a cluster
    # Do once with sample proportional to counts
    # Then just sample uniformly
    by_pop_clust_dist = tfp.distributions.Categorical(probs=(n_counts / np.sum(n_counts)))
    uniform_clust_dist = tfp.distributions.Categorical(probs=(1.0 / len(unique_clusts)) * np.ones(len(unique_clusts)))
    for clust_dist, label in zip([by_pop_clust_dist, uniform_clust_dist],
                                 ['by_pop', 'uniform']):

        # Obtain a sample, first sampling cluster index, then sampling uniformly from each cluster
        clust_sample = clust_dist.sample(n_samples).numpy()
        ind_sample = [np.random.choice(per_cluster_inds[c]) for c in clust_sample]

        # Calculate and save log_probabilities
        clust_probs = clust_dist.log_prob(clust_sample).numpy()
        ind_probs = np.array([np.log(1.0 / n_counts[c]) for c in clust_sample])
        sample_probs = clust_probs + ind_probs
        np.save('CG_log_probs_%s_%s.npy'%(out_name, label), sample_probs)

        # Pull those configurations from the trajectory
        aa_sample_pos = traj.xyz[ind_sample]

        # Create a parmed structure, set its coordinates, convert to CG, and save
        pdb_obj, sim_obj = analysis_tools.sim_from_pdb(pdb_file)
        pmd_struc = pmd.openmm.load_topology(pdb_obj.topology, system=sim_obj.system, xyz=pdb_obj.positions)
        mda_uni = mda.Universe(pmd_struc, aa_sample_pos * 10.0) # Converting from nm to Angstroms
        cg_struc = analysis_tools.create_cg_structure(pmd_struc, mda_uni=mda_uni)
        cg_struc.save('gen_CG_traj_%s_%s.pdb'%(out_name, label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cluster_cg_model.py',
                                     description='Clusters a trajectory and samples from it as a CG model.'
                                     )
    parser.add_argument('pdb_file', help='pdb or structure file of all-atom representation')
    parser.add_argument('traj_file', help='all-atom trajectory to cluster')
    parser.add_argument('--out_name', '-o', default=None, help='naming convention for output')
    parser.add_argument('--n_samples', '-n', type=int, default=100000, help='number of samples to draw')
    parser.add_argument('--sim_stride', type=int, default=10, help='take every sim_stride frames from the trajectory')

    args = parser.parse_args(sys.argv[1:])

    main(args.pdb_file,
         args.traj_file,
         out_name=args.out_name,
         sim_stride=args.sim_stride,
         n_samples=args.n_samples,
        )

