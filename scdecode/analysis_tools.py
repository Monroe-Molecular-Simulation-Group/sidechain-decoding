"""Functions for performing analysis of backmapping models."""

import sys, os
import glob
import argparse
import pickle

import numpy as np
import tensorflow as tf

from scipy import stats as spstats
import matplotlib.pyplot as plt

import openmm as mm
import parmed as pmd
import MDAnalysis as mda
import mdtraj

import vaemolsim

from . import data_io, coord_transforms, model_training, unconditional


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
def config_energy(coords, sim_obj, compute_forces=False, constrain_H_bonds=False):
    """
    Computes energy of a configuration given a simulation object.
    """
    if not isinstance(coords, mm.unit.Quantity):
        coords = coords * mm.unit.angstrom
    sim_obj.context.setPositions(coords)
    if constrain_H_bonds:
        sim_obj.context.applyConstraints(sim_obj.integrator.getConstraintTolerance())
    state = sim_obj.context.getState(getEnergy=True, getForces=compute_forces)
    if compute_forces:
        return (state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole),
                np.array(state.getForces().value_in_unit(mm.unit.kilojoules_per_mole / mm.unit.angstrom)),
               )
    else:
        return state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)


def sim_from_pdb(pdb_file, implicit_solvent=True):
    """
    Creates an OpenMM simulation object from a pdb file.
    Also returns pdb object.
    """
    pdb = mm.app.PDBFile(pdb_file)
    forcefield = data_io.ff # Should be AMBER ff14sb
    if implicit_solvent:
        forcefield.loadFile('implicit/gbn2.xml') # igb8
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=mm.app.NoCutoff,
                                     constraints=mm.app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300*mm.unit.kelvin, 1/mm.unit.picosecond, 0.002*mm.unit.picoseconds)
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
    xyz_coords = coord_transforms.xyz_from_bat_numpy(bat_coords, bat_obj)
    return check_cg(xyz_coords, bat_obj)


def map_to_cg_configs(mda_uni):
    """
    Given an MDAnalysis universe, maps the trajectory to a CG trajectory.
    """
    # Define atom groups
    atom_cg = mda_uni.select_atoms('name %s'%data_io.cg_atoms[1:].replace(',', ' or name '))
    atom_sc = mda_uni.select_atoms('not (name %s)'%data_io.backbone_atoms[1:].replace(',', ' or name '))

    cg_traj = []
    for frame in mda_uni.trajectory:
        cg_traj.append(np.vstack([atom_cg.positions, atom_sc.center_of_mass(compound='residues')]))
    
    return np.array(cg_traj)


def create_cg_structure(pmd_struc, mda_uni=None):
    """
    Given a ParmEd protein structure, generates a CG ParmEd structure.
 
    The CG atoms (mostly backbone) will be present, along with CG sidechain beads.
    """
    # Get positions of atoms and beads in CG configuration
    if mda_uni is None:
        cg_pos = map_to_cg_configs(mda.Universe(pmd_struc))
    else:
        cg_pos = map_to_cg_configs(mda_uni)

    # Get atoms present in CG structure
    cg_struc = pmd_struc[data_io.cg_atoms]

    # Loop over sidechain beads/residues and create own residues and atoms
    num_res = len(cg_struc.residues)
    for i in range(num_res):
        res = cg_struc.residues[i]
        this_cg_atom = pmd.Atom(name='SC', type=res.name)
        # Add bond from beta carbon to CG bead unless residue is glycine
        if res.name == 'GLY':
            bond_atom = [a for a in res.atoms if a.name == 'CA'][0]
        else:
            bond_atom = [a for a in res.atoms if a.name == 'CB'][0]

        # Add CG residue and atom to end of structure, sharing same residue name it came from
        cg_struc.add_atom(this_cg_atom, res.name, num_res + i + 1)

        # Set up and add bond, too
        this_bond = pmd.Bond(this_cg_atom, bond_atom)
        cg_struc.bonds.append(this_bond) 

    # Set all coordinates from trajectory if have them
    cg_struc.coordinates = cg_pos

    # Set up so does not indicate changes
    cg_struc.unchange()

    return cg_struc


def residue_coordination(res_sites, cutoff=10.0):
    """
    Given C-alpha positions or CG sites, assess their degree of coordination
    (i.e., the number of neighbors within a specified cutoff (in Angstroms)
    res_sites should be an (N_res, 3) array
    """
    dist_sq_mat = np.sum((res_sites[None, :, :] - res_sites[:, None, :])**2, axis=-1)
    cut_bool = dist_sq_mat <= (cutoff**2)
    coord_nums = np.sum(cut_bool, axis=-1) - 1 # Subtract one to exclude residue distance with self
    return coord_nums


def residue_burial(res_sites, cutoff=10.0, buried_num=4):
    """
    Calculates the coordinate of all residue sites to other sites and classifies as buried or not
    """
    coord_nums = residue_coordination(res_sites, cutoff=cutoff)
    buried_res = (coord_nums <= buried_num)
    return buried_res


def get_bonded_mask(pmd_struc, nb_cut=0):
    """
    Creates a mask of whether or not an atom is within nb_cut bonds of other atoms.
    Input is a ParmEd structure. Note that nb_cut can only be 0, 1, 2, or 3 since 
    that covers atom itself, bond partners, angle partners, and dihedral partners.
    """
    if nb_cut > 3:
        "WARNING: Cannot consider more than 3 bond partners to exclude, setting to 3 (dihedrals)."
        nb_cut = 3

    # Get list of atoms within bond cutoff for each atom in the ParmEd structure
    bonded_atom_inds = []
    for atom in pmd_struc.atoms:
        to_consider = [atom,]
        if nb_cut >= 1:
            to_consider.extend(atom.bond_partners)
        if nb_cut >= 2:
            to_consider.extend(atom.angle_partners)
        if nb_cut == 3:
            to_consider.extend(atom.dihedral_partners)
        bonded_atom_inds.append([a.idx for a in to_consider])

    # Create mask
    mask = np.ones((len(pmd_struc.atoms), len(pmd_struc.atoms)), dtype=bool)
    for i, inds in enumerate(bonded_atom_inds):
        mask[i, inds] = False # Exclude only the bonded atoms

    return mask


def clash_score(configs, cutoff=1.2, higher_cut=5.0, bond_mask=None):
    """
    Computes clash score, but instead of as a fraction of residues with clashes, as a fraction
    of atoms within 5 Angstroms of each other, which is in line with the original description
    in Yang, 2023..
    Returns score and number of distances within each cutoff (so can recompute).
    BUT, should exclude atoms with bonds, so need optional mask of N_atoms x N_atoms.
    Can create this mask with get_bonded_mask.
    NEED MASK EVEN TO EXCLUDE ATOMS FROM THEMSELVES!
    """
    dist_sq_mat = np.sum((configs[:, None, :, :] - configs[:, :, None, :])**2, axis=-1)
    if bond_mask is not None:
        dist_sq_mat = dist_sq_mat[:, bond_mask] # Flattens, but that's ok
    within_cut = np.sum(dist_sq_mat < (cutoff**2))
    within_max = np.sum(dist_sq_mat < (higher_cut**2))
    return within_cut/within_max, within_cut, within_max


def clash_score_res(trj, thresh=0.12, Ca_cut=2.0, include_H=False):
    """
    Residue-based clash score adapted from Jones, 2023 (https://github.com/Ferg-Lab/DiAMoNDBack/blob/237fc5f4b08feadf15bd1b5a7040392e0e22bcac/scripts/utils.py#L1139)
    Needed to change so that can optionally also consider hydrogens.
    Also changed to consider full trajectory all at once and output a fraction and the numerator and denominator of it.
    Note that mdtraj works in nanometers rather than Angstroms.
    """
    Ca_idxs = []
    for i, atom in enumerate(trj.top.atoms):
        if 'CA' in atom.name:
            Ca_idxs.append(i)
    Ca_idxs = np.array(Ca_idxs)
    Ca_xyzs = trj.xyz[0, Ca_idxs]
    n_res = trj.n_residues
    pairs = []
    for i in range(n_res):
        for j in range(i-1):
            if np.linalg.norm(Ca_xyzs[i]-Ca_xyzs[j]) < Ca_cut:
                pairs.append((i, j))

    if include_H:
        dist, pairs = mdtraj.compute_contacts(trj, contacts=pairs, scheme='closest')
        neighbor_pairs = [(i, i+1) for i in range(trj.n_residues-1)]
        neighbor_dist, neighbor_pairs = mdtraj.compute_contacts(trj, contacts=neighbor_pairs, scheme='sidechain')
    else:
        dist, pairs = mdtraj.compute_contacts(trj, contacts=pairs, scheme='closest-heavy')
        neighbor_pairs = [(i, i+1) for i in range(trj.n_residues-1) if (
            trj.top.residue(i).name != 'GLY' and trj.top.residue(i+1).name != 'GLY')]
        neighbor_dist, neighbor_pairs = mdtraj.compute_contacts(trj, contacts=neighbor_pairs, scheme='sidechain-heavy')
    
    dist = np.concatenate([dist, neighbor_dist], axis=-1)
    pairs = np.concatenate([pairs, neighbor_pairs], axis=0)
    res_closes = list()
    for n_res in range(trj.top.n_residues):
        pair_mask = np.array([n_res in i for i in pairs])
        res_close = np.any(dist[:, pair_mask] < thresh, axis=1)
        res_closes.append(res_close)
    res_closes = np.array(res_closes)
    within_cut = np.sum(res_closes)
    tot_considered = res_closes.size
    
    return within_cut/tot_considered, within_cut, tot_considered


def bond_score(topology, ref_config, configs, cutoff=0.10):
    """
    Determines number of bonds within cutoff*100% of their reference configuration lengths.
    Uses a topology to identify pairs and for computing bonds.
    Takes fraction of such bonds to report as score.
    Returns score, number of bonds within cutoff, and total number of bonds.
    """
    bond_pairs = [[b[0].index, b[1].index] for b in topology.bonds()]
    bond_pairs = np.array(bond_pairs)
    ref_diffs = ref_config[bond_pairs[:, 1]] - ref_config[bond_pairs[:, 0]]
    ref_bonds = np.sqrt(np.sum(ref_diffs*ref_diffs, axis=-1))
    gen_diffs = configs[:, bond_pairs[:, 1], :] - configs[:, bond_pairs[:, 0], :]
    gen_bonds = np.sqrt(np.sum(gen_diffs*gen_diffs, axis=-1))
    within_cutoff = (np.abs(gen_bonds - ref_bonds) < cutoff*ref_bonds)
    return np.sum(within_cutoff) / np.size(within_cutoff), np.sum(within_cutoff), np.size(within_cutoff)


def diversity_score_raw(ref_config, configs):
    """
    Calculates the diversity score (see Jones, 2023). Essentially the ratio of RMSD between
    the generated configurations and the references to the RMSD between all generated configs to each other..
    """
    rmsd_to_ref = np.sqrt(np.average(np.sum((configs - ref_config)**2, axis=-1), axis=-1))
    avg_rmsd_to_ref = np.average(rmsd_to_ref)
    rmsd_combos = []
    for i in range(configs.shape[0]):
        for j in range(i+1, configs.shape[0]):
            this_rmsd = np.sqrt(np.average(np.sum((configs[i] - configs[j])**2, axis=-1), axis=-1))
            rmsd_combos.append(this_rmsd)
    avg_rmsd_combos = np.sum(rmsd_combos) / len(rmsd_combos)
    return 1.0 - (avg_rmsd_combos / avg_rmsd_to_ref)


def diversity_score(ref_config, configs):
    """
    Same as diversity score, but computes both value and bootstrap estimate of variance
    (from 100 resamples)
    """
    raw_score = diversity_score_raw(ref_config, configs)
    bootstrap_inds = np.random.choice(configs.shape[0], size=(100, configs.shape[0]))
    bootstrap_resamples = []
    for inds in bootstrap_inds:
        bootstrap_resamples.append(diversity_score_raw(ref_config, configs[inds]))
    return raw_score, np.var(bootstrap_resamples)


def end_end_distance(pdb_file, traj_file):
    """
    Computes end-to-end distance with MDAnalysis for a trajectory.
    """
    from MDAnalysis.analysis import atomicdistances
    uni = mda.Universe(pdb_file, traj_file)
    ca_res_first = uni.select_atoms('resid 1 and name CA')
    ca_res_last = uni.select_atoms('resid %i and name CA'%len(uni.residues))
    dist_obj = atomicdistances.AtomicDistances(ca_res_first, ca_res_last, pbc=False)
    dist_obj.run()
    end_end_dist = dist_obj.results[:, 0]
    return end_end_dist


def rmsd_from_native(pdb_file, traj_file):
    """
    Given a pdb of the native structure and a trjectory, uses MDAnalysis to compute RMSD for all frames.
    (only considers the backbone for computing RMSD)
    """
    from MDAnalysis.analysis import rms
    ref_uni = mda.Universe(pdb_file, pdb_file)
    ref_bb_atoms = ref_uni.select_atoms('backbone')
    uni = mda.Universe(pdb_file, traj_file)
    bb_atoms = uni.select_atoms('backbone')
    rmsd_obj = rms.RMSD(bb_atoms, reference=ref_bb_atoms)
    rmsd_obj.run()
    rmsd = rmsd_obj.results.rmsd[:, -1]
    return rmsd


# Need to add arguments --include_cg and --h_bonds analagously to train_model
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
    parser.add_argument('--cg_target', action='store_true', help='whether or not to penalize CG bead distance') 
    parser.add_argument('--h_bonds', action='store_true', help='whether or not to constrain bonds with hydrogens')

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

    # Load the BAT analysis object
    bat_obj_file = glob.glob('%s/*.pkl'%args.read_dir)[0]
    with open(bat_obj_file, 'rb') as f:
        bat_obj = pickle.load(f)

    # Create the appropriate model
    n_atoms = len(bat_obj._torsions) # Will also be number of bonds, angles, and torsions
    # Get number of H-bonds that will be constrained, along with H bond info
    if args.h_bonds:
        h_inds, non_h_inds, h_bond_lengths = coord_transforms.get_h_bond_info(bat_obj)
        n_H_bonds = len(h_inds)
    else:
        h_inds = []
        non_h_inds = list(range(len(bat_obj._torsions)*3))
        h_bond_lengths = []
        n_H_bonds = 0

    if args.unconditional:
        model = unconditional.build_model(n_atoms, n_H_bonds=n_H_bonds)
    else:
        model = model_training.build_model(n_atoms, n_H_bonds=n_H_bonds)

    # Select loss
    if args.cg_target and not args.unconditional:
        loss = model_training.LogProbPenalizedCGLoss(bat_obj, mask_H=args.h_bonds)
    else:
        loss = vaemolsim.losses.LogProbLoss()

    # Next identify training files and load dataset without batching
    if args.unconditional:
        # Unless working with unconditional stuff...
        # Then dset is just the full_bat stuff
        dset = tf.data.Dataset.from_tensor_slices((full_bat[:, non_h_inds], full_bat[:, non_h_inds])).batch(batch_size)
    else:
        train_files = glob.glob('%s/*.tfrecord'%args.read_dir)
        train_files.sort()
        train_files = train_files
        dset = data_io.read_dataset(train_files, include_cg_target=args.cg_target)
        # Want CG coordinates for comparison
        cg_only_dset = dset.map(lambda x, y : x[0]).batch(1000)
        cg_refs = np.vstack([cg for cg in cg_only_dset])
        dset = dset.ragged_batch(batch_size)

    # Compile, build by passing through one sample, and load weights
    model.compile(tf.keras.optimizers.Adam(),
                  loss=loss,
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

    # Fill in hydrogens (works whether constrained or not)
    samples = coord_transforms.fill_in_h_bonds(samples, h_inds, non_h_inds, h_bond_lengths)

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
    num_batches = full_bat.shape[0] // batch_size
    if num_batches >= 10:
        extra_batch_inds = rng.choice(num_batches, size=10, replace=False).tolist()
    else:
        extra_batch_inds = np.arange(num_batches).tolist()
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
            extra_samples = coord_transforms.fill_in_h_bonds(extra_samples, h_inds, non_h_inds, h_bond_lengths)
            this_hist, this_edges = build_bat_histograms(extra_samples)
            # Once have histograms, label uniquely with keys
            for k in this_hist.keys():
                extra_sample_hists['example%i_%s'%(extra_sample_count, k)] = this_hist[k]
                extra_sample_edges['example%i_%s_edges'%(extra_sample_count, k)] = this_edges[k+'_edges']
            extra_sample_count += 1

    # And save extra sample histograms
    np.savez('%s_BAT_model_extra_samples.npz'%args.save_prefix, **extra_sample_hists, **extra_sample_edges)


# Define functions to help with sorting when plotting
def get_num(s):
    return int(s.split('_')[-1])


def sort_BAT_keys(keys):
    new_list = []
    for lab in ['bond', 'angle', 'dihedral']:
        this_list = [k for k in keys if lab in k]
        this_list.sort(key=get_num)
        new_list.extend(this_list)
    return new_list


# And for normalizing histograms
def norm_hist(hist, edges):
    new_hist = hist / np.sum(hist)
    new_hist /= (edges[1:] - edges[:-1])
    return new_hist


def plot_history(hist_list, labels=None):
    if labels is None:
        if len(hist_list) > 1:
            labels = ['Model %i '%(i+1) for i in range(len(hist_list))]
        else:
            labels = ['',]

    fig, ax = plt.subplots()

    for i, history in enumerate(hist_list):
        epochs = 1 + np.arange(history['loss'].shape[0])
        ax.plot(epochs, history['loss'], label=labels[i]+'Training Loss')
        ax.plot(epochs, history['val_loss'], label=labels[i]+'Validation Loss', linestyle='--')
        # ax.plot(epochs, history['mean_log_prob'], label=labels[i]+'Log-Prob')
        # ax.plot(epochs, history['loss'] - history['mean_log_prob'], label=labels[i]+'CG Penalty')
        # ax.plot(epochs, history['val_mean_log_prob'], label=labels[i]+'Val Log-Prob', linestyle='--')
        # ax.plot(epochs, history['val_loss'] - history['val_mean_log_prob'], label=labels[i]+'Val CG Penalty', linestyle='--')

    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Loss')

    ax.legend()

    fig.tight_layout()
    return fig


def plot_dofs(ref_dat, pred_dat, labels=None, extra_samples=False):
    if len(ref_dat) != len(pred_dat):
        raise ValueError("Lengths of data lists must match!")

    if labels is None:
        if len(ref_dat) > 1:
            labels = ['Dataset %i '%(i+1) for i in range(len(ref_dat))]
        else:
            labels = ['',]

    keys = [k for k in ref_dat[0].keys() if 'edges' not in k]
    keys = sort_BAT_keys(keys)

    n_col = 5
    n_row = int(np.ceil(len(keys) / n_col))
    
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*3.0, n_row*2.5))

    ref_colors = ['gray', 'black']
    
    for a, key in zip(ax.flatten()[:len(keys)], keys):

        for i in range(len(ref_dat)):
            
            if extra_samples:
                for j in range(1, 11):
                    hist = pred_dat[i]['example%i_%s'%(j, key)]
                    edges = pred_dat[i]['example%i_%s_edges'%(j, key)]
                    a.stairs(norm_hist(hist, edges),
                             edges=edges,
                             label=labels[i]+'Config %i'%j,
                            ) 
            else:
                a.stairs(norm_hist(pred_dat[i][key], pred_dat[i]['%s_edges'%key]),
                         edges=pred_dat[i]['%s_edges'%key],
                         label=labels[i]+'Model',
                        )
            a.stairs(norm_hist(ref_dat[i][key], ref_dat[i]['%s_edges'%key]),
                     edges=ref_dat[i]['%s_edges'%key],
                     label=labels[i]+'Ref',
                     color=ref_colors[i],
                     linestyle='--',
                    )
       
        a.annotate(key, xy=(0.05, 0.90), xycoords='axes fraction')
    
    ax[0, 0].legend(loc='lower left')
    
    fig.tight_layout()
    return fig


def plot_cg_diffs(cg_dat):

    fig, ax = plt.subplots(3, 2)
     
    for i, dim in enumerate(['x', 'y', 'z']):
        hist = cg_dat['%s_hist'%dim]
        edges = cg_dat['%s_edges'%dim]
        ax[i, 0].stairs(norm_hist(hist, edges), edges=edges)
    
        # Check if can fit to Gaussian
        centers = 0.5*(edges[1:] + edges[:-1])
        mean = np.sum((hist / np.sum(hist)) * centers)
        std = np.sqrt(np.sum((hist / np.sum(hist)) * (centers - mean)**2))
        ax[i, 0].plot(centers, spstats.norm.pdf(centers, mean, std), 'k--')
    
        # Seems like a Cauchy distribution fits much better, though hard to determine best-fit parameters
        # Note not fitting based on likelihood, just noting that parameters match median and mad
        cum_dist = np.cumsum(hist / np.sum(hist))
        median = centers[np.argmin(np.abs(cum_dist - 0.5))]
        mad = np.sum((hist / np.sum(hist)) * np.abs(centers - median))
        ax[i, 0].plot(centers, spstats.cauchy.pdf(centers, median, mad), 'c--')
    
        # Or a Laplace distribution? This is probably the best fit
        abs_mean = np.sum((hist / np.sum(hist)) * np.abs(centers - median))
        ax[i, 0].plot(centers, spstats.laplace.pdf(centers, median, abs_mean), 'r--')
   
        # And plot cummulative distributions, too
        ax[i, 1].plot(centers, cum_dist)
        ax[i, 1].plot(centers, spstats.norm.cdf(centers, mean, std), 'k--')
        ax[i, 1].plot(centers, spstats.cauchy.cdf(centers, median, mad), 'c--')
        ax[i, 1].plot(centers, spstats.laplace.cdf(centers, median, abs_mean), 'r--')
    
    fig.tight_layout()
    return fig


def get_laplace_scale_from_hists(hist_file):
    """
    Given a .npz file containing histograms of x, y, and z differences of samples from the reference CG bead,
    computes an approximate best-fit scale (or b) parameter for a Laplace distribution. Assumes median is zero
    and averages over scale parameters for each coordinate dimension since all very similar.
    """
    dat = np.load(hist_file)
    scale_fit = []
    for coord in ['x', 'y', 'z']:
        hist = dat['%s_hist'%coord]
        edges = dat['%s_edges'%coord]
        centers = 0.5*(edges[1:] + edges[:-1])
        cum_dist = np.cumsum(hist / np.sum(hist))
        abs_mean = np.sum((hist / np.sum(hist)) * np.abs(centers))
        scale_fit.append(abs_mean)
    return np.average(scale_fit)


def plot_analysis(arg_list):
    """
    Plots results of analyze_model.
    """
    parser = argparse.ArgumentParser(prog='analysis_tools.plot_analysis',
                                     description='Generates plots of analysis results.',
                                    )
    parser.add_argument('res_type', help='residue type to work with')
    parser.add_argument('--read_dirs', '-r', default=['./',], nargs='*', help='directories to read files from')
    parser.add_argument('--model_dirs', '-m', default=['./',], nargs='*', help='directories where models and training histories found')
    parser.add_argument('--save_dir', '-s', default='./', help='directory to save figures to')
    parser.add_argument('--labels', '-l', default=None, nargs='*', help='labels for data from directories read')

    args = parser.parse_args(arg_list)

    if args.labels is None:
        if len(args.read_dirs) == 1:
            labels = ['',]
        else:
            labels = ['', 'E-Min ']
    else:
        labels = args.labels[:len(args.read_dirs)]
    
    # Start by plotting training history
    history_list = [np.load('%s/%s_history.npz'%(d, args.res_type)) for d in args.model_dirs]
    hist_fig = plot_history(history_list, labels=labels)
    hist_fig.savefig('%s/%s_train_history.png'%(args.save_dir, args.res_type))

    # Next look at decoded distributions compared to reference
    ref_dat = [np.load('%s/%s_BAT_data.npz'%(d, args.res_type)) for d in args.read_dirs]
    pred_dat = [np.load('%s/%s_BAT_model.npz'%(d, args.res_type)) for d in args.read_dirs]

    # Start with plot of single prediction for every sample compared to reference
    dofs_fig = plot_dofs(ref_dat, pred_dat, labels=labels)
    dofs_fig.savefig('%s/%s_DOFs.png'%(args.save_dir, args.res_type))

    # Plots of the reference compared to 10 static configurations with many draws
    # Plot for each dataset
    # Only do this if can load data (some residues with small amounts of data may not have generated)
    try:
        extra_dat = [np.load('%s/%s_BAT_model_extra_samples.npz'%(d, args.res_type)) for d in args.read_dirs]
        for i in range(len(args.read_dirs)):
            extra_dofs_fig = plot_dofs([ref_dat[i]], [extra_dat[i]], extra_samples=True)
            extra_dofs_fig.savefig('%s/%s_extra_sample_DOFs_%s.png'%(args.save_dir, args.res_type, labels[i].strip()))
    except FileNotFoundError:
        print("Skipping single config distribution analysis for residue %s because could not find 'extra_sample' files."%args.res_type)

    # Examine differences of decoded structure from original CG bead location
    # May fail if CG bead location not defined or irrelevant
    try:
        cg_dat = [np.load('%s/%s_CG_diffs.npz'%(d, args.res_type)) for d in args.read_dirs]
        for i in range(len(cg_dat)):
            cg_fig = plot_cg_diffs(cg_dat[i])
            cg_fig.savefig('%s/%s_CG_diffs_%s.png'%(args.save_dir, args.res_type, labels[i].strip()))
    except FileNotFoundError:
        print("Skipping CG bead difference analysis for residue %s because could not find 'CG_diffs' files."%args.res_type)


if __name__ == "__main__":
    if sys.argv[1] == 'bat_stats':
        compute_bat_stats(sys.argv[2:])
    elif sys.argv[1] == 'analyze_model':
        analyze_model(sys.argv[2:])
    elif sys.argv[1] == 'plot_analysis':
        plot_analysis(sys.argv[2:])
    else:
        print("Argument \'%s\' unrecognized. For the first argument, select \'bat_stats\', \'analyze_model\', or \'plot_analysis\'."%sys.argv[1])
