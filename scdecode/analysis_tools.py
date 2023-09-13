"""Functions for performing analysis of backmapping models."""

import sys, os
import glob
import argparse
import pickle

import numpy as np
import tensorflow as tf

import openmm as mm
import parmed as pmd

import vaemolsim

from . import data_io, model_training, unconditional


def build_bat_histograms(all_bat):
    """
    Creates and collects histograms for each column of a numpy array of BAT coordinates.
    """
    num_atoms = int(all_bat.shape[1] / 3)

    all_hist = {}
    all_edges = {}
    for i in range(all_bat.shape[1]):
        this_hist, this_edges = np.histogram(all_bat[:, i], bins='auto')
        if i < num_atoms:
            label = "bond_%i"%(i + 1)
        elif i >= num_atoms and i < 2*num_atoms:
            label = "angle_%i"%(i - num_atoms + 1)
        elif i >= 2*num_atoms:
            label = "dihedral_%i"%(i - 2*num_atoms + 1)
        all_hist[label] = this_hist
        all_edges["%s_edges"%label] = this_edges

    return all_hist, all_edges


def bat_summary_statistics(files):
    """
    Performs summary statistics over the dataset of BAT coordinates.
    (as a list of .npy files)

    Focuses on building histograms of all BAT degrees of freedom to be predicted.
    """
    all_bat = []
    for f in files:
        all_bat.append(np.load(f)[:, 9:])

    all_bat = np.vstack(all_bat)

    return build_bat_histograms(all_bat)


def compute_bat_stats(arg_list):
    """
    Command line tool for computing BAT distributions.
    """
    parser = argparse.ArgumentParser(prog='analysis_tools.compute_bat_stats',
                                     description='Computes histograms for target BAT coordinates from npy files.',
                                    )
    parser.add_argument('--read_dir', '-r', default='./', help='directory to read files from')
    parser.add_argument('--save_file', '-s', default='bat_stats.npz', help='location/name of file to save to')

    args = parser.parse_args(arg_list)

    files = glob.glob('%s/*.npy'%args.read_dir)

    hist_dict, edges_dict = bat_summary_statistics(files)

    np.savez(args.save_file, **hist_dict, **edges_dict)


# To see which residues having trouble, or if atomic overlaps, helps to have forces
# Have optional flag to return
# That way, can visualize where have high energies/forces by coloring by force if want
def config_energy(coords, sim_obj, compute_forces=False):
    """
    Computes energy of a configuration given a simulation object.
    """
    if not isinstance(coords, mm.unit.Quantity):
        coords = coords * mm.unit.angstrom
    sim_obj.context.setPositions(coords)
    state = sim_obj.context.getState(getEnergy=True, getForces=compute_forces)
    if compute_forces:
        return (state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole),
                np.array(state.getForces().value_in_unit(mm.unit.kilojoules_per_mole / mm.unit.angstrom)),
               )
    else:
        return state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)


def sim_from_pdb(pdb_file):
    """
    Creates an OpenMM simulation object from a pdb file.
    Also returns pdb object.
    """
    pdb = mm.app.PDBFile(pdb_file)
    forcefield = data_io.ff
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=mm.app.NoCutoff)
    integrator = mm.LangevinIntegrator(300*mm.unit.kelvin, 1/mm.unit.picosecond, 0.004*mm.unit.picoseconds)
    simulation = mm.app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    return pdb, simulation


def pdb_energy(pdb_file, compute_forces=False):
    """
    Energy from a pdb file.
    """
    pdb, simulation = sim_from_pdb(pdb_file)
    return config_energy(pdb.positions, simulation, compute_forces=compute_forces)


def pdb_energy_decomp(pdb_file):
    """
    Full decomposition of energies in pdb file using Parmed
    """
    pdb, sim = sim_from_pdb(pdb_file)
    struc = pmd.openmm.load_topology(pdb.topology, system=sim.system, xyz=pdb.positions)
    return pmd.openmm.energy_decomposition_system(struc, sim.system, nrg=mm.unit.kilojoules_per_mole)


def xyz_from_bat(bat_coords, bat_obj):
    """
    Loops over many BAT coordinates to convert back to Cartesian.

    Parameters
    ----------
    bat_coords : NumPy array
        The full set of BAT coordinates for a specific residue/sidechain.
    bat_obj : MDAnalysis BAT analysis object
        The BAT analysis object that can convert between BAT and Cartesian.

    Returns
    -------
    xyz_coords : NumPy array
        The Cartesian coordinates of the residue/sidechain.
    """
    xyz_coords = []
    for bc in bat_coords:
        xyz_coords.append(bat_obj.Cartesian(bc))
    return np.array(xyz_coords)


def fill_in_bat(partial_bat, root_pos):
    """
    Recreates a full set of BAT coordinates from a partial set and the root atom positions.

    Parameters
    ----------
    partial_bat : NumPy array
        The partial set of BAT coordinates, not including the root atom (first 3) coordinates.
    root_pos : NumPy array
        A N_frames by 3 by 3 array of the positions of the root atoms. For most residues,
        this will be C, CA, and CB, but may be different for something like GLY.

    Returns
    -------
    full_bat : NumPy array
        The full set of BAT coordinates, including information on the CA and CB atom
        locations, which is needed for converting back to XYZ coordinates for all
        atoms in a sidechain.
    """
    if len(root_pos.shape) == 2:
        root_pos = np.expand_dims(root_pos, 0)
    elif len(root_pos.shape) == 3:
        pass
    else:
        raise ValueError('Positions of root atoms must be N_batchx3x3 or 3x3 (if have no batch dimension).')

    if len(partial_bat.shape) == 1:
        partial_bat = np.expand_dims(partial_bat, 0)

    n_batch = root_pos.shape[0]
    p0 = root_pos[:, 0, :]
    p1 = root_pos[:, 1, :]
    p2 = root_pos[:, 2, :]
    v01 = p1 - p0
    v21 = p1 - p2
    r01 = np.sqrt(np.sum(v01 * v01, axis=-1))
    r12 = np.sqrt(np.sum(v21 * v21, axis=-1))
    a012 = np.arccos(np.sum(v01 * v21, axis=-1) / (r01 * r12))
    polar = np.arccos(v01[:, 2] / r01)
    azimuthal = np.arctan2(v01[:, 1], v01[:, 0])
    cp = np.cos(azimuthal)
    sp = np.sin(azimuthal)
    ct = np.cos(polar)
    st = np.sin(polar)
    Rz = np.array([[cp * ct, ct * sp, -st], [-sp, cp, np.zeros(n_batch)], [cp * st, sp * st, ct]])
    Rz = np.transpose(Rz, axes=(2, 0, 1))
    pos2 = np.squeeze(Rz @ np.expand_dims(p2 - p1, -1), axis=-1)
    omega = np.arctan2(pos2[:, 1], pos2[:, 0])
    full_bat = np.hstack([p0, azimuthal[:, None], polar[:, None], omega[:, None],
                          r01[:, None], r12[:, None], a012[:, None], partial_bat])
    
    return np.squeeze(full_bat)


def check_cg(xyz_coords, bat_obj):
    """
    Calculates the coarse-grained site position given Cartesian coordinates of the BAT atoms.
    """
    working_ag = bat_obj.atoms.convert_to('PARMED')
    coms = []
    for xyz in xyz_coords:
        working_ag.coordinates = xyz
        sc_ag = working_ag['!(%s)'%data_io.backbone_atoms]
        masses = np.array([a.mass for a in sc_ag])
        coms.append(pmd.geometry.center_of_mass(sc_ag.coordinates, masses))
    return np.array(coms)


def check_cg_from_bat(bat_coords, bat_obj):
    """
    Calculates the coarse-grained site position from BAT coordinates.
    """
    xyz_coords = xyz_from_bat(bat_coords, bat_obj)
    return check_cg(xyz_coords, bat_obj)


def analyze_model(arg_list):
    """
    Command line tool for analyzing a decoding model..
    """
    parser = argparse.ArgumentParser(prog='analysis_tools.analyze_model',
                                     description='Computes histograms for various decoded BAT coordinate distributions and CG coords as well.',
                                    )
    parser.add_argument('res_type', help='residue type to work with')
    parser.add_argument('--read_dir', '-r', default='./', help='directory to read files from')
    parser.add_argument('--model_ckpt', '-m', default='./', help='model checkpoint file path')
    parser.add_argument('--save_prefix', '-s', default=None, help='prefix for files that will be saved')
    parser.add_argument('--rng_seed', default=42, type=int, help='random number seed for selecting configs')
    parser.add_argument('--unconditional', action='store_true', help='whether or not model is unconditional')

    args = parser.parse_args(arg_list)

    if args.save_prefix is None:
        args.save_prefix = args.res_type

    batch_size = 64

    # Start by identifying files with full BAT distributions and read in
    full_bat_files = glob.glob('%s/*.npy'%args.read_dir)
    full_bat_files.sort()
    full_bat_files = full_bat_files
    full_bat = np.vstack([np.load(f) for f in full_bat_files]).astype('float32')

    # Save BAT data as reference distribution
    ref_hists, ref_edges = build_bat_histograms(full_bat[:, 9:])
    np.savez('%s_BAT_data.npz'%args.save_prefix, **ref_hists, **ref_edges)

    # Next identify training files and load dataset without batching
    if args.unconditional:
        # Unless working with unconditional stuff...
        # Then dset is just the full_bat stuff
        dset = tf.data.Dataset.from_tensor_slices((full_bat, full_bat)).batch(batch_size)
    else:
        train_files = glob.glob('%s/*.tfrecord'%args.read_dir)
        train_files.sort()
        train_files = train_files
        dset = data_io.read_dataset(train_files)
        # Want CG coordinates for comparison
        cg_only_dset = dset.map(lambda x, y : x[0]).batch(1000)
        cg_refs = np.vstack([cg for cg in cg_only_dset])
        dset = dset.ragged_batch(batch_size)

    # Load the BAT analysis object
    bat_obj_file = glob.glob('%s/*.pkl'%args.read_dir)[0]
    with open(bat_obj_file, 'rb') as f:
        bat_obj = pickle.load(f)

    # Create the appropriate model
    n_atoms = len(bat_obj._torsions) # Will also be number of bonds, angles, and torsions
    if args.unconditional:
        model = unconditional.build_model(n_atoms)
    else:
        model = model_training.build_model(n_atoms)

    # Compile, build by passing through one sample, and load weights
    model.compile(tf.keras.optimizers.Adam(),
                  loss=vaemolsim.losses.LogProbLoss(),
                 )
    build_data = next(iter(dset))[0]
    _ = model(build_data)
    model.load_weights(args.model_ckpt).expect_partial()

    # Predict for full dataset
    # Note will need custom predict_step so that draw samples
    # tfp.distribution objects are not tensors, so will break default predict
    # NEED TO FIGURE OUT PREDICT WITH DIFFERENT DISTRIBUTIONS
    # For just a flow with a static distribution, need to create static to match batch_size (within the make_distribution_fn)
    # Then for others, should get batch size right, but be careful when sampling one distribution extra times
    samples = model.predict(dset, verbose=2)

    # Produce and save distributions of BAT samples
    m_hists, m_edges = build_bat_histograms(samples)
    np.savez('%s_BAT_model.npz'%args.save_prefix, **m_hists, **m_edges)

    if not args.unconditional:
        # Get CG bead positions of predictions and compare to references
        cg_pos = check_cg_from_bat(np.hstack([full_bat[:, :9], samples]), bat_obj)
        cg_diffs = cg_pos - cg_refs
        diff_hist_x, diff_edges_x = np.histogram(cg_diffs[:, 0], bins='auto')
        diff_hist_y, diff_edges_y = np.histogram(cg_diffs[:, 1], bins='auto')
        diff_hist_z, diff_edges_z = np.histogram(cg_diffs[:, 2], bins='auto')
        np.savez('%s_CG_diffs.npz'%args.save_prefix,
                 x_hist=diff_hist_x, y_hist=diff_hist_y, z_hist=diff_hist_z,
                 x_edges=diff_edges_x, y_edges=diff_edges_y, z_edges=diff_edges_z)

    # Want to select 10 random training examples to produce MANY samples from
    # Do by selecting indices
    rng = np.random.default_rng(seed=args.rng_seed)
    extra_batch_inds = rng.choice(full_bat.shape[0] // batch_size, size=10, replace=False).tolist()
    print('Batches with indices will be sampled extra: %s'%str(extra_batch_inds))
    extra_sample_inds = rng.choice(batch_size, size=10, replace=False).tolist()
    if full_bat.shape[0] - 1 in extra_batch_inds:
        extra_sample_inds[extra_batch_inds.index(full_bat.shape[0] - 1)] = rng.choice(full_bat.shape[0] % batch_size, replace=False)
    print('Within those batches, samples with indices will be sampled extra: %s'%str(extra_sample_inds))

    # Then looping over dataset and only stopping on selected indices
    extra_sample_hists = {}
    extra_sample_edges = {}
    extra_sample_count = 1
    for i, (inputs, target) in enumerate(dset):
        if i in extra_batch_inds:
            this_sample_ind = extra_sample_inds[extra_batch_inds.index(i)]
            if args.unconditional:
                this_input = tf.gather(inputs, [this_sample_ind,], axis=0)
            else:
                this_input = [tf.gather(d, [this_sample_ind,], axis=0) for d in inputs]
            dist = model(this_input)
            extra_samples = np.squeeze(dist.sample(10000))
            this_hist, this_edges = build_bat_histograms(extra_samples)
            # Once have histograms, label uniquely with keys
            for k in this_hist.keys():
                extra_sample_hists['example%i_%s'%(extra_sample_count, k)] = this_hist[k]
                extra_sample_edges['example%i_%s_edges'%(extra_sample_count, k)] = this_edges[k+'_edges']
            extra_sample_count += 1

    # And save extra sample histograms
    np.savez('%s_BAT_model_extra_samples.npz'%args.save_prefix, **extra_sample_hists, **extra_sample_edges)


if __name__ == "__main__":
    if sys.argv[1] == 'bat_stats':
        compute_bat_stats(sys.argv[2:])
    elif sys.argv[1] == 'analyze_model':
        analyze_model(sys.argv[2:])
    else:
        print("Argument \'%s\' unrecognized. For the first argument, select \'bat_stats\' or \'analyze_model\'.")
