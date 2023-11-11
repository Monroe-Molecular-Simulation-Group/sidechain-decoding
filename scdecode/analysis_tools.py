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
        sim_obj.context.applyConstraints()
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
 
    # Loop over residues and create residue and atom lists
    res_list = pmd.ResidueList()
    atom_list = pmd.AtomList()
    for i, res in enumerate(pmd_struc.residues):
        res_list.add_atom(pmd.Atom(name='CG', type=res.name), 'SC%s'%res.name, i + 1, 'A')
        atom_list.append(res_list[i].atoms[0])

    # Create just sidechain bead structure
    sc_beads = pmd.Structure()
    sc_beads.residues = res_list
    sc_beads.atoms = atom_list
    sc_beads.coordinates = cg_pos[0, -len(pmd_struc.residues):, :]
    
    # Combine CG atoms and CG sidechain beads
    combined_struc = cg_struc + sc_beads

    # Set all coordinates from trajectory if have them
    combined_struc.coordinates = cg_pos

    return combined_struc


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
        h_inds, non_h_inds, h_bond_lengths = coord_transforms.get_h_bond_info(bat_dict[res])
        n_H_bonds = len(h_inds)
    else:
        h_inds = []
        non_h_inds = list(range(len(bat_dict[res]._torsions)))
        h_bond_lengths = []
        n_H_bonds = 0

    if args.unconditional:
        model = unconditional.build_model(n_atoms, n_H_bonds=n_H_bonds)
    else:
        model = model_training.build_model(n_atoms, n_H_bonds=n_H_bonds)

    # Select loss
    if args.cg_target and not args.unconditional:
        loss = LogProbPenalizedCGLoss(bat_obj, mask_H=args.h_bonds)
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
    samples = coord_transforms.fill_in_h_bonds(samples.numpy(), h_inds, non_h_inds, h_bond_lengths)

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
        ax.plot(epochs, history['val_loss'], label=labels[i]+'Validation Loss')

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
        cum_dist = np.cumsum(hist / np.sum(hist))
        median = centers[np.argmin(np.abs(cum_dist - 0.5))]
        mad = np.sum((hist / np.sum(hist)) * np.abs(centers - median))
        ax[i, 0].plot(centers, spstats.cauchy.pdf(centers, median, mad), 'c--')
    
        # Or a Laplace distribution? This is probably the best fit
        ax[i, 0].plot(centers, spstats.laplace.pdf(centers, mean, std/np.sqrt(2)), 'r--')
   
        # And plot cummulative distributions, too
        ax[i, 1].plot(centers, cum_dist)
        ax[i, 1].plot(centers, spstats.norm.cdf(centers, mean, std), 'k--')
        ax[i, 1].plot(centers, spstats.cauchy.cdf(centers, median, mad), 'c--')
        ax[i, 1].plot(centers, spstats.laplace.cdf(centers, mean, std/np.sqrt(2)), 'r--')
    
    fig.tight_layout()
    return fig


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
        print("Argument \'%s\' unrecognized. For the first argument, select \'bat_stats\', \'analyze_model\', or \'plot_analysis\'.")
