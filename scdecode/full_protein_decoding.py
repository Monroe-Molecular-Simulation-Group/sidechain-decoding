"""
Applies residue-based decodings to an entire protein coarse-grained model.
"""

import sys, os
import datetime
import argparse
import pickle
import glob
import copy

import numpy as np
import tensorflow as tf

import parmed as pmd
import MDAnalysis as mda
import mdtraj
import openmm as mm
from openmm import app as mmapp

import vaemolsim

from . import data_io, coord_transforms, model_training, unconditional, analysis_tools 


def get_unique_res(sequence):
    """
    From a list of residue names (matching the AMBER ff14sb force field),
    assigns names matching trained models. Cannot deal with capping groups
    and will automatically add the N-terminal type Nterm. For any N-terminal
    residue, will also add the residue itself (e.g., if have NALA, also adds
    (ALA), if not already present, and removes NALA. Same with C-terminal
    residues.
    """

    # Check for N-terminal residues and add normal residue as well
    # Do the same thing for C-terminal residues (don't need special decoding for these, though)
    unique_res = []
    for res in sequence:
        if len(res) == 4:
            unique_res.append(res[1:])
            if res[0] == 'N':
                if res == 'NPRO':
                    unique_res.append(res)
                else:
                    unique_res.append('Nterm')
        else:
            unique_res.append(res)
    
    # Select out only the unique residue names
    unique_res = np.unique(unique_res)

    return unique_res


def gather_bat_objects(res_types, search_dir='./'):
    """
    Loads MDAnalysis BAT analysis objects for each residue type.

    Residue types should already be unique.
    """

    # Will return a dictionary of BAT objects matched to residue names
    bat_dict = {}

    for res in res_types:
        this_bat_file = '%s/%s/%s_BAT_object.pkl'%(search_dir, res, res)
        with open(this_bat_file, 'rb') as f:
            this_bat_obj = pickle.load(f)
        bat_dict[res] = this_bat_obj

    return bat_dict


# Need to add arguments --include_cg and --h_bonds analagously to train_model
def gather_models(res_types,
                  bat_dict,
                  model_dir='./',
                  include_cg_target=False,
                  constrain_H_bonds=False,
                  unconditional_types=['GLY', 'NPRO', 'Nterm'], # ['NPRO', 'Nterm'],
                 ):
    """
    Given a sequence and dictionary of BAT objects, loads models for every residue type.

    Note that residue types should match the force field used for training (AMBER ff14sb).
    AND, they should already be unique.

    Also returns dictionary of info for bonds involving hydrogens.
    (will be important for decoding and sampling... don't want to rely on model loss)
    """

    # Loop over unique residue types and create dictionary mapping to decoding models
    decode_dict = {}
    h_info_dict = {}

    for res in res_types:

        print("Loading model for residue type %s"%res)

        # Get number of atoms for this residue type based on BAT object
        n_atoms = len(bat_dict[res]._torsions) # Will also be number of bonds, angles, and torsions
       
        # And number of H-bonds that will be constrained, along with H bond info
        if constrain_H_bonds:
            h_inds, non_h_inds, h_bond_lengths = coord_transforms.get_h_bond_info(bat_dict[res])
            n_H_bonds = len(h_inds)
        else:
            h_inds = []
            non_h_inds = list(range(len(bat_dict[res]._torsions)*3))
            h_bond_lengths = []
            n_H_bonds = 0

        # Add to dictionary of bonds involving hydrogens
        h_info_dict[res] = [h_inds, non_h_inds, h_bond_lengths]

        # Build the model depending on whether it should be unconditional or not
        if res in unconditional_types:
            this_model = unconditional.build_model(n_atoms, n_H_bonds=n_H_bonds)
            build_data = tf.zeros((1, n_atoms*3), dtype=tf.float32)
        else:
            this_model = model_training.build_model(n_atoms, n_H_bonds=n_H_bonds)
            # Note that have 112 FG and CG bead types total (when creating one-hot encoding of type)
            # That should show up in data_io.all_ref_types
            build_data = (tf.zeros((1, 3), dtype=tf.float32),
                          tf.ones((1, 1, 3), dtype=tf.float32),
                          tf.reshape(tf.cast(tf.range(len(data_io.all_ref_types)) < 1, tf.float32), (1, 1, -1)),
                         )
    
        # Set up right loss
        if include_cg_target and res not in unconditional_types:
            # Setting one_over_cg_var to zero so acts like normal log-probability loss if use here
            # Safer because need this loss if contraining H bonds when training, but may have set
            # the CG penalization weight (i.e., one_over_cg_var) to zero
            # Either way, should not need loss here
            loss = model_training.LogProbPenalizedCGLoss(bat_dict[res], mask_H=constrain_H_bonds,
                                                         one_over_cg_var=0.0)
        else:
            loss = vaemolsim.losses.LogProbLoss()

        # Compile, build by passing through one sample, and load weights
        this_model.compile(tf.keras.optimizers.Adam(),
                           loss=loss,
                          )
        _ = this_model(build_data)
        this_model_ckpt = '%s/%s_decoder/%s_weights.ckpt'%(model_dir, res, res)
        this_model.load_weights(this_model_ckpt).expect_partial()

        decode_dict[res] = this_model

    return decode_dict, h_info_dict


def one_hot_from_types(types):
    one_hot = np.zeros((len(types), len(data_io.all_ref_types)))
    for i, t in enumerate(types):
        one_hot[i, :] = np.array(data_io.all_ref_types == t, dtype='float32')
    return one_hot


class ProteinDecoder(tf.Module):
    """
    Class for decoding a protein.
    """

    def __init__(self, pdb_file, bat_dir='./', model_dir='./',
                 bat_dict=None, model_dict=None, h_info_dict=None,
                 include_cg=False, h_bonds=False):
        """
        Given a cleaned up PDB file, creates a class instance.
        """

        # Will load the PDB file with OpenMM and create a parametrized system
        mm_pdb = mmapp.PDBFile(pdb_file)
        # Follow procedure in openmm.app.ForceField.createSystem() to correctly apply
        # forcefield information to this pdb
        ff_data = data_io.ff._SystemData(mm_pdb.topology)
        templates_for_residues = data_io.ff._matchAllResiduesToTemplates(ff_data, mm_pdb.topology, dict(), False)
        res_types = [templates_for_residues[r.index].name for r in mm_pdb.topology.residues()]
        self.sequence = res_types

        # Helpful to have ParmEd structure object for understanding topology and atom indices
        self.pmd_struc = pmd.openmm.load_topology(mm_pdb.topology, xyz=mm_pdb.positions)

        # Can generate one-hot encodings for CG configuration now
        # Consists of all atoms preserved in CG config plus sidechain beads
        cg_only = self.pmd_struc.view[data_io.cg_atoms]
        cg_inds = [a.idx for a in cg_only.atoms]
        self.cg_inds = cg_inds # Will be helpful for stitching configurations back together
        atom_types = np.array([ff_data.atomType[a] for a in mm_pdb.topology.atoms()])
        cg_types = [atom_types[i] for i in cg_inds]
        # Add on sidechain beads
        cg_types = np.array(cg_types + res_types)
        self.one_hot_cg = one_hot_from_types(cg_types)

        # Get unique residue list for loading appropriate BAT objects and models
        unique_res = get_unique_res(res_types)
        # Load dictionaries of BAT objects and decoding models, if not already provided
        if bat_dict is None:
            self.bat_dict = gather_bat_objects(unique_res, search_dir=bat_dir)
        else:
            self.bat_dict = bat_dict
        if model_dict is None or h_info_dict is None:
            self.model_dict, self.h_info_dict = gather_models(unique_res,
                                                              self.bat_dict,
                                                              model_dir=model_dir,
                                                              include_cg_target=include_cg,
                                                              constrain_H_bonds=h_bonds,
                                                             )
        else:
            self.model_dict = model_dict
            self.h_info_dict = h_info_dict

        # Will need structure of only CG atoms (for getting those indices specifically, etc.)
        cg_struc = self.pmd_struc[data_io.cg_atoms] # Positions of only CG atoms (without sidechain beads)

        # ONLY for N-terminal model Nterm, remove H hydrogen from CG_config
        # The Nterm decoding model predicts all hydrogens on the nitrogen, even though one is in the CG configuration
        # Here store indices in CG configuration so can remove from input
        # Also, need to modify CG information created to date, including the cg_struc structure
        # Necessary so BAT root indices will be correct when identify later
        remove_Nterm_H_inds = []
        for i, res in enumerate(self.sequence):
            if len(res) == 4 and res[0] == 'N' and res[1:] != 'PRO':
                this_root_names = [a.name for a in self.bat_dict['Nterm']._root]
                this_decode_names = [a.name for a in self.bat_dict['Nterm']._ag if a.name not in this_root_names]
                exclude_ind = [cg_struc.view[':%i@%s'%(i + 1, a[0])].atoms[0].idx for a in this_decode_names if a == 'H1']
                remove_Nterm_H_inds.append(exclude_ind[0])
                
        # Need indices of all atoms except N-terminal hydrogens
        self.cg_non_Nterm_H_inds = np.delete(np.arange(len(cg_struc.atoms) + len(self.sequence)), remove_Nterm_H_inds)
        # Remove one-hot indices for N-terminal H atoms
        self.one_hot_cg = np.delete(self.one_hot_cg, remove_Nterm_H_inds, axis=0)
        # And adjust self.cg_inds
        self.cg_inds = np.delete(self.cg_inds, remove_Nterm_H_inds)
        # Finally, get rid of those atoms in cg_struc as well
        exclude_str = '!(@' + ','.join(['%i'%(ind + 1) for ind in remove_Nterm_H_inds]) + ')'
        cg_struc = cg_struc[exclude_str]

        # To save time, it will also be nice to have all of the atom indices necessary for
        # recreating XYZ coords from BAT
        # But these should be indices in the COARSE-GRAINED configuration (use cg_struc)
        # Also, once have xyz coordinates, need to map those to indices in overall protein
        # (i.e., from bat.Cartesian, get xyz coords in some order, then for each atom need overall index)
        # For simplicity, will do decoding in two steps
        # First for all unconditional distributions
        # Then for all conditional
        # So need separate lists of residue names and atom indices of decoding for each
        # At the same time, can also prepare one-hot encodings for each decoding model
        # (only requires decoded atom types)
        # And for conditional models, need indices of CG sidechain beads in CG configuration
        # Note that CG sidechain beads are last len(self.sequence) entries in cg_config coordinates
        uncond_root_inds = []
        uncond_seq = []
        uncond_decode_inds = []
        uncond_one_hot = []
        cond_root_inds = []
        cond_cg_ref_inds = []
        cond_seq = []
        cond_decode_inds = []
        cond_one_hot = []
        for i, res in enumerate(self.sequence):
            # If N-terminal add to unconditional sets
            # If not, just set residue name
            # Remember, even if N-terminal, need to add to conditional set (if not GLY) as well
            if len(res) == 4:
                if res[0] == 'C':
                    # For C-terminal, just get residue name and proceed
                    res_name = res[1:]
                elif res[0] == 'N':
                    res_name = res[1:]
                    # N-proline is a special case
                    if res == 'NPRO':
                        uncond_seq.append(res)
                        this_root_names = [a.name for a in self.bat_dict['NPRO']._root]
                        this_decode_names = [a.name for a in self.bat_dict['NPRO']._ag if a.name not in this_root_names]
                    else:
                        uncond_seq.append('Nterm')
                        this_root_names = [a.name for a in self.bat_dict['Nterm']._root]
                        this_decode_names = []
                        for a in self.bat_dict['Nterm']._ag:
                            if a.name not in this_root_names:
                                if '1' not in a.name:
                                    this_decode_names.append(a.name)
                                else:
                                    this_decode_names.append(a.name[0])
                    this_root_inds = [cg_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_root_names]
                    uncond_root_inds.append(this_root_inds)
                    this_decode_inds = [self.pmd_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_decode_names]
                    uncond_decode_inds.append(this_decode_inds)
                    this_one_hot = one_hot_from_types(atom_types[this_decode_inds])
                    uncond_one_hot.append(this_one_hot)
            else:
                res_name = res
            
            # Use BAT object to get names of root atoms, then get indices from a view over the CG structure
            this_root_names = [a.name for a in self.bat_dict[res_name]._root]
            this_root_inds = [cg_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_root_names]
            # Also obtain names of atoms that will be decoded, get their indices, then their one-hot encoding
            # Note that the XYZ coordinates produced by the bat object will contain the root atoms
            # However, we can exclude those based on _root_XYZ_inds, so here also exclude the root atoms
            this_decode_names = [a.name for a in self.bat_dict[res_name]._ag if a.name not in this_root_names]
            this_decode_inds = [self.pmd_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_decode_names]
            this_one_hot = one_hot_from_types(atom_types[this_decode_inds])
            if res_name == 'GLY':
                uncond_seq.append(res_name)
                uncond_root_inds.append(this_root_inds)
                uncond_decode_inds.append(this_decode_inds)
                uncond_one_hot.append(this_one_hot)
            else:
                cond_seq.append(res_name)
                cond_root_inds.append(this_root_inds)
                cond_one_hot.append(this_one_hot)
                cond_decode_inds.append(this_decode_inds)
                cond_cg_ref_inds.append(i - len(self.sequence))

        self.uncond_root_inds = uncond_root_inds
        self.uncond_seq = uncond_seq
        self.uncond_decode_inds = uncond_decode_inds
        self.uncond_one_hot = uncond_one_hot
        self.cond_root_inds = cond_root_inds
        self.cond_seq = cond_seq
        self.cond_decode_inds = cond_decode_inds
        self.cond_one_hot = cond_one_hot
        self.cond_cg_ref_inds = cond_cg_ref_inds

        # Need way to sort decoded indices back to order that is expected by protein structure
        self.full_decode_inds = np.hstack([self.cg_inds,] + self.uncond_decode_inds + self.cond_decode_inds)
        self.sort_inds = np.argsort(self.full_decode_inds)

        # Handy to have number of unconditional and conditional to decode handy
        self.num_uncond = len(self.uncond_seq)
        self.num_cond = len(self.cond_seq)

        # Also need lists of non-root BAT indices for all residue types
        self.bat_non_root_inds = {}
        for key, val in self.bat_dict.items():
            self.bat_non_root_inds[key] = np.delete(np.arange(len(val.atoms)), val._root_XYZ_inds)

    # input_signature helps prevent retracing of graph
    # (even for numpy inputs, it seems - to be safe, maybe pass in tensors instead)
    # Actually know the shape of the second dimension as self.one_hot_cg.shape[0]
    # However, cannot access self in the tf.function decorator call
    # So just ensure the shape is correct within the function
    # To parallelize sample generation, tile the input configuration before passing
    # That way, flexibly allows different configurations or the same configuration repeated multiple times
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),))
    def decode_config(self, cg_config):
        """
        Given CG configuration(s) of shape (N_frames, N_particles, 3), decodes to atomistic.
        Should consist of all atomistic "CG" coordinates (i.e., the backbone-ish) plus the
        sidechain beads, in that order so that it matches up with self.one_hot_cg.
        """
        # Prep configuration by removing N-terminal H hydrogens
        cg_config = tf.gather(tf.cast(cg_config, tf.float32), self.cg_non_Nterm_H_inds, axis=1)

        # Check shape of configuration
        cg_config = tf.ensure_shape(cg_config, [None, self.one_hot_cg.shape[0], 3])
        batch_shape = tf.shape(cg_config)[0]

        # Need to tile the one-hot indicators to match the input cg configuration
        cg_one_hot = tf.tile(tf.expand_dims(tf.cast(self.one_hot_cg, tf.float32), axis=0), (batch_shape, 1, 1))

        # Will loop over the residues and decode each
        # Can keep track of decoded positions in a list to concatenate with the CG config at each pass
        # The decoding models do not care about atom order, so can stitch together correctly at end
        # Same with one-hot encodings
        # Also make sure to keep track of probability for each generated sample
        decoded_coords = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                        clear_after_read=False, infer_shape=False)
        decoded_one_hot = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                         clear_after_read=False, infer_shape=False)
        decoded_probs = tf.TensorArray(tf.float32, size=(self.num_uncond + self.num_cond),
                                       clear_after_read=False, infer_shape=True)

        # Decode with unconditional models first
        for i, res in enumerate(self.uncond_seq):
            # Could move below outside loop and save something like self.bat_input as dictionary of tf.Tensor objects
            bat = self.bat_dict[res]
            bat_input = bat.results.bat[:, 9:][:, self.h_info_dict[res][1]]
            bat_input = tf.tile(tf.cast(bat_input, tf.float32), (batch_shape, 1))

            # If want probabilities, cannot use predict_on_batch...
            # Need distribution, then compute log_probabilities of sample
            dist = self.model_dict[res](bat_input)
            sample = dist.sample()
            prob = dist.log_prob(sample)
            # Note that writing to TensorArray objects works by assigning to themselves
            # Doing this ensures functioning in graph mode and prevents errors in eager mode
            decoded_probs = decoded_probs.write(i, prob)

            # Fill in hydrogens (with tensorflow ops)
            full_bat_sample = coord_transforms.fill_in_h_bonds_tf(sample, *self.h_info_dict[res])

            # Convert BAT coordinates to xyz
            full_bat_sample = coord_transforms.fill_in_bat(full_bat_sample,
                                                           tf.gather(cg_config, self.uncond_root_inds[i], axis=1))
            sample_xyz = coord_transforms.bat_cartesian_tf(full_bat_sample, bat)
            # But want to exclude the root atom indices, which are already in the CG config
            sample_xyz = tf.gather(sample_xyz, self.bat_non_root_inds[res], axis=1)
            # Write to TensorArray, but must transpose so .concat works!
            # Means each entry in decoded_coords will be (n_atoms, n_confs, 3)
            decoded_coords = decoded_coords.write(i, tf.transpose(sample_xyz, perm=[1, 0, 2]))
            
            # Get one-hot encoding for this set of decoded atoms
            one_hot = tf.tile(tf.expand_dims(tf.cast(self.uncond_one_hot[i], tf.float32), axis=0), (batch_shape, 1, 1))
            decoded_one_hot = decoded_one_hot.write(i, tf.transpose(one_hot, perm=[1, 0, 2]))

        # Decode conditional models
        for i, res in enumerate(self.cond_seq):
            
            # Add coordinates and one-hot encodings decoded up to this point to CG configuration
            if decoded_coords.size() != 0:
                current_decoded = tf.transpose(decoded_coords.concat(), perm=[1, 0, 2])
                this_config = tf.concat([cg_config, current_decoded], axis=1)
                current_one_hot = tf.transpose(decoded_one_hot.concat(), perm=[1, 0, 2])
                this_one_hot = tf.concat([cg_one_hot, current_one_hot], axis=1)
            else:
                this_config = cg_config
                this_one_hot = cg_one_hot

            # Grab reference sidechain location to decode
            this_ref = cg_config[:, self.cond_cg_ref_inds[i], :]

            # Collect input
            this_input = (this_ref, this_config, this_one_hot)
            
            # Apply decoding model
            dist = self.model_dict[res](this_input)
            sample = dist.sample()
            prob = dist.log_prob(sample)
            decoded_probs = decoded_probs.write(i + self.num_uncond, prob)

            # Fill in hydrogens
            full_bat_sample = coord_transforms.fill_in_h_bonds_tf(sample, *self.h_info_dict[res])

            # Convert BAT coordinates to xyz
            full_bat_sample = coord_transforms.fill_in_bat(full_bat_sample,
                                                           tf.gather(cg_config, self.cond_root_inds[i], axis=1))
            sample_xyz = coord_transforms.bat_cartesian_tf(full_bat_sample, self.bat_dict[res])
            # But want to exclude the root atom indices, which are already in the CG config
            sample_xyz = tf.gather(sample_xyz, self.bat_non_root_inds[res], axis=1)
            decoded_coords = decoded_coords.write(i + self.num_uncond, tf.transpose(sample_xyz, perm=[1, 0, 2]))

            # Get one-hot encoding for this set of decoded atoms
            one_hot = tf.tile(tf.expand_dims(tf.cast(self.cond_one_hot[i], tf.float32), axis=0), (batch_shape, 1, 1))
            decoded_one_hot = decoded_one_hot.write(i + self.num_uncond, tf.transpose(one_hot, perm=[1, 0, 2]))

        # Finally, need to stitch the configuration back together
        decoded_coords_tensor = tf.transpose(decoded_coords.concat(), perm=[1, 0, 2])
        decoded_configs = tf.concat([cg_config[:, :-len(self.sequence), :], decoded_coords_tensor], axis=1)
        decoded_configs = tf.gather(decoded_configs, self.sort_inds, axis=1)

        # Sum over log probabilities (for each decoding model)
        decoded_probs = tf.math.reduce_sum(decoded_probs.stack(), axis=0)

        # Close decoded_one_hot and mark as used to avoid error in eager execution mode
        decoded_one_hot = decoded_one_hot.close()

        return decoded_configs, decoded_probs


def analyze_trajectory(pdb_file, traj_file, bat_dir='./', model_dir='./', out_name=None, n_samples=1):
    """
    Assesses the ability of a full protein decoding to recreate an atomistic trajectory.
    First evaluates all energies of atomistic configurations, then converts to a CG
    trajectory and decodes some number of times for each frame, evaluating energies and
    forces of decoded configurations.
    """

    # First, create a full protein decoding model
    # Do this first because collects lots of useful information
    full_decode = ProteinDecoder(pdb_file,
                                 bat_dir=bat_dir,
                                 model_dir=model_dir,
                                 h_bonds=True,
                                 include_cg=True, # Need if constraining bonds involving hydrogens
                                )

    # Create an OpenMM simulation object
    pdb_obj, sim = analysis_tools.sim_from_pdb(pdb_file)
    
    # Helpful to have ParmEd structure
    pmd_struc = pmd.openmm.load_topology(pdb_obj.topology, system=sim.system, xyz=pdb_obj.positions)

    # Load in the trajectory to get coordinates from
    uni = mda.Universe(pmd_struc.topology, traj_file)

    # Get atom indices for every residue
    res_atom_inds = []
    for res in uni.residues:
        this_inds = [a.ix for a in res.atoms]
        res_atom_inds.append(this_inds)

    print("\nDone with prep.")
    print(datetime.datetime.now())

    # Compute all energies and forces of simulated trajectory
    # For forces, store max force in each residue
    sim_energies = []
    sim_forces = []
    sim_decomp = {}
    for frame in uni.trajectory:
        this_energy, this_forces = analysis_tools.config_energy(frame.positions,
                                                                sim,
                                                                compute_forces=True,
                                                                constrain_H_bonds=True,
                                                                )
        sim_energies.append(this_energy)
        this_force_mags = np.linalg.norm(this_forces, axis=-1)
        this_max_f = [np.max(this_force_mags[inds]) for inds in res_atom_inds]
        sim_forces.append(this_max_f)
        # Add energy decomposition as well
        pmd_struc.coordinates = frame.positions
        this_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, sim.system, nrg=mm.unit.kilojoules_per_mole)
        for key, eng in this_decomp:
            if key in sim_decomp:
                sim_decomp[key].append(eng)
            else:
                sim_decomp[key] = [eng]

    sim_energies = np.array(sim_energies)
    sim_forces = np.array(sim_forces)
    out_sim_decomp = {}
    for key, eng in sim_decomp.items():
        out_sim_decomp['sim_'+key] = np.array(eng)
    out_sim_decomp.pop('sim_CMMotionRemover', None)

    print("\nSimulation energies computed. Working with decoding.")
    print(datetime.datetime.now())

    # Generate a CG trajectory from the all-atom one
    cg_traj = analysis_tools.map_to_cg_configs(uni)

    # Since have CG trajectory, interesting to assess the burial of residues
    # Can base on C-alpha atoms or CG sites - using C-alpha atoms for consistency
    # with other literature
    res_coordination = []
    c_alpha_sel = uni.select_atoms('name CA')
    for frame in uni.trajectory:
        this_coordination = analysis_tools.residue_coordination(c_alpha_sel.positions)
        res_coordination.append(this_coordination)

    res_coordination = np.array(res_coordination)

    print("\nGenerated CG trajectory from all-atom simulation trajectory.")
    print(datetime.datetime.now())

    # Generate a decoded trajectory with n_samples per frame
    cg_traj = np.tile(cg_traj, (n_samples, 1, 1))
    decoded_traj = []
    decoded_probs = []
    n_chunk = 200
    for i in range(0, cg_traj.shape[0], n_chunk):
        decoded_configs, this_probs = full_decode.decode_config(cg_traj[i:(i+n_chunk)])
        decoded_traj.append(decoded_configs.numpy())
        decoded_probs.append(this_probs.numpy())

    decoded_traj = np.concatenate(decoded_traj, axis=0)
    decoded_probs = np.concatenate(decoded_probs, axis=0)

    print("\nDecoding complete. Calculating stats and energies of decoded configurations.")
    print(datetime.datetime.now())

    # Obtain energies and max forces for the decoded trajectory
    decoded_energies = []
    decoded_forces = []
    decoded_decomp = {}
    for config in decoded_traj:
        this_energy, this_forces = analysis_tools.config_energy(config,
                                                                sim,
                                                                compute_forces=True,
                                                                constrain_H_bonds=True,
                                                                )
        decoded_energies.append(this_energy)
        this_force_mags = np.linalg.norm(this_forces, axis=-1)
        this_max_f = [np.max(this_force_mags[inds]) for inds in res_atom_inds]
        decoded_forces.append(this_max_f)
        # Add energy decomposition as well
        pmd_struc.coordinates = config
        this_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, sim.system, nrg=mm.unit.kilojoules_per_mole)
        for key, eng in this_decomp:
            if key in decoded_decomp:
                decoded_decomp[key].append(eng)
            else:
                decoded_decomp[key] = [eng]

    decoded_energies = np.array(decoded_energies)
    decoded_forces = np.array(decoded_forces)
    out_decoded_decomp = {}
    for key, eng in decoded_decomp.items():
        out_decoded_decomp['decoded_'+key] = np.array(eng)
    out_decoded_decomp.pop('decoded_CMMotionRemover', None)

    print("\nDecoded energies computed. Saving.")
    print(datetime.datetime.now())

    # Save everything as npz file
    # To do that nicely, want forces as both arrays and dictionaries categorized by residue type
    unique_res = np.unique(full_decode.sequence).tolist()
    unique_inds = [[i for i, r in enumerate(full_decode.sequence) if r == res] for res in unique_res]
    res_force_dict = {}
    for res, inds in zip(unique_res, unique_inds):
        res_force_dict['sim_%s'%res] = sim_forces[:, inds].flatten()
        res_force_dict['decoded_%s'%res] = decoded_forces[:, inds].flatten()

    # Create an output file name
    if out_name is None:
        out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    # Actually save
    np.savez('energy_analysis_%s.npz'%out_name,
             sim_energy=sim_energies,
             decoded_energy=decoded_energies,
             sim_max_force=sim_forces,
             decoded_max_force=decoded_forces,
             res_coordination=res_coordination,
             decoded_probs=decoded_probs,
             **out_sim_decomp,
             **out_decoded_decomp,
             **res_force_dict,
            )

    print("\nPerforming BAT distribution analysis on sidechain of each residue.")
    print(datetime.datetime.now())

    # Also want to compare distributions of BAT coordinates between original trajectory and decoded structures
    # Loop over each residue type, skipping GLY, and compute BAT coordinates with MDAnalysis BAT object
    # Start with simulation trajectory
    # Annoyingly, the MDAnalysis BAT analysis cannot handle pulling a selection from a Universe without modifying bonds
    # (unlike ParmEd, bonds are not dropped when atoms are excluded)
    # So need to use ParmEd to select out the atoms, then still use MDAnalysis to extract specific coordinates
    for i, res_type in enumerate(full_decode.sequence):
        if 'GLY' in res_type:
            continue
        this_bat_atoms = pmd_struc['(:%i)&(!(%s))'%(i + 1, data_io.not_bat_atoms)]
        this_sel = uni.select_atoms('resnum %i and not (name %s)'%(i+1, data_io.not_bat_atoms[1:].replace(',', ' or name ')))
        this_traj = uni.trajectory.timeseries(this_sel, order='fac')
        this_bat_uni = mda.Universe(this_bat_atoms, this_traj)
        bat_analysis = mda.analysis.bat.BAT(this_bat_uni.select_atoms('all'), initial_atom=this_bat_uni.select_atoms('name C')[0])
        bat_analysis.run()
        # this_hists, this_edges = analysis_tools.build_bat_histograms(bat_analysis.results.bat[:, 9:])
        # np.savez('sim_BAT_stats_%s%i.npz'%(res_type, i+1), **this_hists, **this_edges)
        np.save('sim_BAT_%s%i.npy'%(res_type, i+1), bat_analysis.results.bat, allow_pickle=False)

    # Redo with universe created from decoded trajectory
    decoded_uni = mda.Universe(pmd_struc.topology, decoded_traj) #, format=mda.coordinates.memory.MemoryReader)
    for i, res_type in enumerate(full_decode.sequence):
        if 'GLY' in res_type:
            continue
        this_bat_atoms = pmd_struc['(:%i)&(!(%s))'%(i + 1, data_io.not_bat_atoms)]
        this_sel = decoded_uni.select_atoms('resnum %i and not (name %s)'%(i+1, data_io.not_bat_atoms[1:].replace(',', ' or name ')))
        this_traj = decoded_uni.trajectory.timeseries(this_sel, order='fac')
        this_bat_uni = mda.Universe(this_bat_atoms, this_traj)
        bat_analysis = mda.analysis.bat.BAT(this_bat_uni.select_atoms('all'), initial_atom=this_bat_uni.select_atoms('name C')[0])
        bat_analysis.run()
        # this_hists, this_edges = analysis_tools.build_bat_histograms(bat_analysis.results.bat[:, 9:])
        # np.savez('decoded_BAT_stats_%s%i.npz'%(res_type, i+1), **this_hists, **this_edges)
        np.save('decoded_BAT_%s%i.npy'%(res_type, i+1), bat_analysis.results.bat, allow_pickle=False)

    # Also a good idea to save the decoded configurations (can then use for re-weighting properties computed from them
    # along with energies and log-probabilities)
    # Note that will reweight with non-restrained energy, assuming that restraint effectively constrains backbone configuration
    # (or that backbone configs can be drawn from a Gaussian distribution exactly that cancels with restrained probability)
    out_traj_name = 'decoded_' + traj_file.split('/')[-1].split('.')[0] + '.nc'
    decoded_uni.select_atoms('all').write(out_traj_name, frames='all')


def analyze_individual_residues(pdb_file, bat_dir='./', model_dir='./', out_name=None, n_samples=100000):
    """
    For a single configuration, performs n_samples decodings for each residue separately.
    This assesses the performance of a trained decoding model for specific residues with all
    other atomic positions provided rather than fully decoding an entire protein. This procedure
    is amenable to reweighting individual residue decodings rather than an entire protein.
    """

    # Set name for outputs associated with pdb file
    if out_name is None:
        out_name = pdb_file.split('.pdb')[0].split('/')[-1]

    # Create an OpenMM simulation object
    pdb_obj, sim = analysis_tools.sim_from_pdb(pdb_file)
   
    # Collect residue types and one-hot encoding for them
    ff_data = data_io.ff._SystemData(pdb_obj.topology)
    templates_for_residues = data_io.ff._matchAllResiduesToTemplates(ff_data, pdb_obj.topology, dict(), False)
    res_types = [templates_for_residues[r.index].name for r in pdb_obj.topology.residues()]
    one_hot_cg = one_hot_from_types(res_types)

    # And atom types and their one-hot encoding
    atom_types = np.array([ff_data.atomType[a] for a in pdb_obj.topology.atoms()])
    one_hot_aa = one_hot_from_types(atom_types)

    # Get unique residue list for loading appropriate BAT objects and models
    unique_res = get_unique_res(res_types)

    # Load dictionaries of BAT objects and decoding models, if not already provided
    bat_dict = gather_bat_objects(unique_res, search_dir=bat_dir)
    model_dict, h_info_dict = gather_models(unique_res,
                                            bat_dict,
                                            model_dir=model_dir,
                                            include_cg_target=True,
                                            constrain_H_bonds=True,
                                           )

    # Also need ParmEd structure
    pmd_struc = pmd.openmm.load_topology(pdb_obj.topology, system=sim.system, xyz=pdb_obj.positions)

    # Preserve copy of coordinates that won't be changed
    fixed_coords = copy.deepcopy(pmd_struc.coordinates)

    # Generate a CG structure from the all-atom one
    # Note that CG center of mass beads will be last N_residues coordinates
    cg_config = analysis_tools.map_to_cg_configs(mda.Universe(pmd_struc))

    # Store outputs in dictionaries
    all_energies = {}
    all_decomp = {}
    all_logprobs = {}

    # For reference structure, get energy and decomposition
    ref_energy = analysis_tools.config_energy(fixed_coords,
                                              sim,
                                              compute_forces=False,
                                              constrain_H_bonds=True,
                                              )
    all_energies['ref_energy'] = ref_energy
    ref_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, sim.system, nrg=mm.unit.kilojoules_per_mole)
    for key, eng in ref_decomp:
        all_decomp['ref_%s'%key] = np.array(eng)
    all_decomp.pop('ref_CMMotionRemover', None)

    # Loop over each residue
    num_res = len(pmd_struc.residues)
    for i, res in enumerate(pmd_struc.residues):

        res_key = "%s%i"%(res.name, i+1)

        # Use BAT object to get names of root atoms, then get indices from a view over the CG structure
        this_root_names = [a.name for a in bat_dict[res.name]._root]
        this_root_inds = [pmd_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_root_names]
        this_root_coords = fixed_coords[this_root_inds]

        # Also obtain names of atoms that will be decoded, get their indices
        # Note that the XYZ coordinates produced by the bat object will contain the root atoms
        # So also excluding those root atoms and getting indices within BAT object excluding them as well
        this_decode_names = [a.name for a in bat_dict[res.name]._ag if a.name not in this_root_names]
        this_decode_inds = [pmd_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_decode_names]
        this_bat_non_root_inds = np.delete(np.arange(len(bat_dict[res.name].atoms)), bat_dict[res.name]._root_XYZ_inds)

        # Generate inputs for decoding model
        if res.name == 'GLY':
            this_model_inputs = bat_dict[res.name].results.bat[:, 9:][:, h_info_dict[res.name][1]]
        else:
            # Get CG bead location of this residue
            this_cg_bead = cg_config[0, -(num_res - i)]
            # And get other atom and CG bead locations and one-hot encodings
            this_other_input = np.concatenate([np.delete(fixed_coords, this_decode_inds, axis=0),
                                               cg_config[0, -num_res:]]
                                             )
            this_other_onehot = np.concatenate([np.delete(one_hot_aa, this_decode_inds, axis=0),
                                               one_hot_cg,]
                                              )
            this_model_inputs = (this_cg_bead[None, ...], this_other_input[None, ...], this_other_onehot[None, ...])
        
        # Generate predicted probability distribution for BAT coordinates
        this_dist = model_dict[res.name](this_model_inputs)

        # Sample n_samples times from it, in batches to avoid overflows
        sample = []
        logprobs = []
        n_sub_sample = 10000
        tot_sampled = 0
        while tot_sampled < n_samples:
            if (n_samples - tot_sampled) >= n_sub_sample:
                this_n_sample = n_sub_sample
            else:
                this_n_sample = (n_samples % n_sub_sample)

            tot_sampled += this_n_sample

            this_sample = tf.squeeze(this_dist.sample(this_n_sample))
            logprobs.append(this_dist.log_prob(this_sample))
            sample.append(this_sample)

        sample = tf.concat(sample, axis=0)
        logprobs = tf.concat(logprobs, axis=0)
        
        # Fill in hydrogens
        full_bat_sample = coord_transforms.fill_in_h_bonds_tf(sample, *h_info_dict[res.name])

        # Convert BAT coordinates to xyz
        full_bat_sample = coord_transforms.fill_in_bat(full_bat_sample,
                                np.tile(this_root_coords, (n_samples, 1, 1)).astype('float32'))


        # Save sample BAT coordinates, including root positions
        np.save("BAT_sample_%s_%s.npy"%(out_name, res_key), np.array(full_bat_sample), allow_pickle=False)

        # Convert to XYZ
        sample_xyz = coord_transforms.bat_cartesian_tf(full_bat_sample, bat_dict[res.name]).numpy()

        # Only extract non-root coordinates
        sample_xyz = sample_xyz[:, this_bat_non_root_inds]

        # Insert decoded coordinates into structure and calculate energies
        decoded_energies = []
        decoded_decomp = {}
        for j in range(n_samples):
            this_config = copy.deepcopy(fixed_coords)
            this_config[this_decode_inds] = sample_xyz[j]
            this_energy = analysis_tools.config_energy(this_config,
                                                       sim,
                                                       compute_forces=False,
                                                       constrain_H_bonds=True,
                                                       )
            decoded_energies.append(this_energy)
            # Add energy decomposition as well
            pmd_struc.coordinates = this_config
            this_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, sim.system, nrg=mm.unit.kilojoules_per_mole)
            for key, eng in this_decomp:
                if key in decoded_decomp:
                    decoded_decomp[key].append(eng)
                else:
                    decoded_decomp[key] = [eng]
        
        # Store energies, decomposition, and log_probabilities
        decoded_energies = np.array(decoded_energies)
        for key, eng in decoded_decomp.items():
            all_decomp['decoded_%s_%s'%(key, res_key)] = np.array(eng)
        all_decomp.pop('decoded_CMMotionRemover_%s'%res_key, None)
        all_energies['decoded_energy_%s'%res_key] = decoded_energies
        all_logprobs['decoded_probs_%s'%res_key] = np.array(logprobs)
        
    # Save everything as npz file
    np.savez('individual_res_energy_analysis_%s.npz'%out_name,
             **all_energies,
             **all_decomp,
             **all_logprobs,
            )


def analyze_protein_dataset(pdb_files, bat_dir='./', model_dir='./', out_name='dataset_decoding_analysis', n_samples=100):
    """
    Given a list of pdb files (paths), generates full decoding models from provided BAT and
    model data directories. For each pdb file, n_samples decodings full protein decodings are
    generated and assessed.
    """

    # Since will only evaluate each full protein decoding model once, tracing the graph is just extra time
    tf.config.run_functions_eagerly(True)

    # Avoid loading all models, BAT dictionaries, etc. multiple times
    # Get unique residue list for loading all BAT objects and models
    unique_res = get_unique_res(data_io.ref_res_types).tolist()
    # Need to remove some residue types we do not have models for
    # (so we don't try and fail to load them)
    # NHE, NME, and ACE are capping groups
    # CYX is a cysteine participating in a disulfide bond
    # CYM, HYP, and LYN are special modifications of residues that had no data in training dataset
    for r in ['NHE', 'NME', 'ACE', 'CYX', 'CYM', 'HYP', 'LYN']:
        unique_res.remove(r)
    print(unique_res)
    unique_res = np.array(unique_res)
    # Load dictionaries of BAT objects and decoding models
    bat_dict = gather_bat_objects(unique_res, search_dir=bat_dir)
    model_dict, h_info_dict = gather_models(unique_res,
                                            bat_dict,
                                            model_dir=model_dir,
                                            include_cg_target=False,
                                            constrain_H_bonds=True,
                                           )

    energies = []
    energy_decomps = {}
    decoded_energies = []
    decoded_energy_decomps = {}
    decoded_probs = []
    max_forces_by_res = {}
    decoded_max_forces_by_res = {}
    coordination_by_res = {}
    cg_diffs_by_res = {}
    for rtype in data_io.ref_res_types:
        max_forces_by_res[rtype] = []
        decoded_max_forces_by_res[rtype] = []
        coordination_by_res[rtype] = []
        cg_diffs_by_res[rtype] = []

    # Other metrics for decoded configurations
    # Will save on a per-protein basis
    clash_score_stats = []
    clash_score_res_stats = []
    bond_score_stats = []
    diversity_score_stats = []
    clash_score_noH_stats = []
    clash_score_res_noH_stats = []
    bond_score_noH_stats = []
    diversity_score_noH_stats = []

    # And general stats on proteins
    num_res = []
    num_atoms = []
    pdb_ids = []

    # Loop over pdb files
    for file in pdb_files:
        print("\nOn pdb %s"%(file.split('/')[-1]))
        pdb_ids.append(file.split('/')[-1].split('.pdb')[0])
        
        # Obtain a ParmEd structure and an OpenMM simulation object
        this_pdb_obj, this_sim = analysis_tools.sim_from_pdb(file)
        pmd_struc = pmd.openmm.load_topology(this_pdb_obj.topology, system=this_sim.system, xyz=this_pdb_obj.positions)

        # And need mdtraj topology for residue-based clash score calculation
        md_top = mdtraj.Topology.from_openmm(this_pdb_obj.topology)

        # Get information on pmd structure now, including bond mask
        num_res.append(len(pmd_struc.residues))
        num_atoms.append(len(pmd_struc.atoms))
        this_bond_mask = analysis_tools.get_bonded_mask(pmd_struc, nb_cut=3)

        # Helpful to have a structure without hydrogens as well
        pmd_struc_noH = pmd_struc['!(@H=)']
        noH_inds = [a.idx for a in pmd_struc.view['!(@H=)'].atoms]
        this_bond_mask_noH = analysis_tools.get_bonded_mask(pmd_struc_noH, nb_cut=3)

        # Get coordination of all residues
        this_res_coordination = analysis_tools.residue_coordination(pmd_struc['@CA'].coordinates)

        # Create a full decoding model - useful to do this earlier since collects information we want
        full_decode = ProteinDecoder(file,
                                     bat_dir=bat_dir,
                                     model_dir=model_dir,
                                     bat_dict=bat_dict,
                                     model_dict=model_dict,
                                     h_info_dict=h_info_dict,
                                     h_bonds=True,
                                     include_cg=True, # Need if constraining bonds involving hydrogens
                                    )

        # Collect energy of input structure
        ref_energy, ref_forces = analysis_tools.config_energy(pmd_struc.coordinates,
                                                              this_sim,
                                                              compute_forces=True,
                                                              constrain_H_bonds=True,
                                                             )
        energies.append(ref_energy)
        ref_force_mags = np.linalg.norm(ref_forces, axis=-1)

        # Need to get max force for atoms in each residue and add to dictionary
        for i, res in enumerate(full_decode.sequence):
            this_max_f = np.max(ref_force_mags[[a.idx for a in pmd_struc.residues[i].atoms]])
            max_forces_by_res[res].append(this_max_f)
            
        # Add energy decomposition as well
        ref_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, this_sim.system, nrg=mm.unit.kilojoules_per_mole)
        for key, eng in ref_decomp:
            if key in energy_decomps:
                energy_decomps[key].append(eng)
            else:
                energy_decomps[key] = [eng]

        # Generate a CG configuration
        cg_config = analysis_tools.map_to_cg_configs(mda.Universe(pmd_struc))
        
        # Decode n_samples of configurations
        cg_config = np.tile(cg_config, (n_samples, 1, 1))
        gen_configs, gen_probs = full_decode.decode_config(cg_config)
        gen_configs = gen_configs.numpy()
        gen_cg_configs = analysis_tools.map_to_cg_configs(mda.Universe(pmd_struc, gen_configs)) 
        
        decoded_probs.append(gen_probs.numpy())

        # Get bond, clash, and diversity scores now (just need generated configurations and reference)
        bond_score_stats.append(analysis_tools.bond_score(pmd_struc.topology, pmd_struc.coordinates, gen_configs))
        clash_score_stats.append(analysis_tools.clash_score(gen_configs, bond_mask=this_bond_mask))
        clash_score_res_stats.append(analysis_tools.clash_score_res(mdtraj.Trajectory(gen_configs/10.0, md_top), include_H=True))
        diversity_score_stats.append(analysis_tools.diversity_score(pmd_struc.coordinates, gen_configs))
        bond_score_noH_stats.append(analysis_tools.bond_score(pmd_struc_noH.topology, pmd_struc_noH.coordinates, gen_configs[:, noH_inds, :]))
        clash_score_noH_stats.append(analysis_tools.clash_score(gen_configs[:, noH_inds, :], bond_mask=this_bond_mask_noH))
        clash_score_res_noH_stats.append(analysis_tools.clash_score_res(mdtraj.Trajectory(gen_configs/10.0, md_top), include_H=False))
        diversity_score_noH_stats.append(analysis_tools.diversity_score(pmd_struc_noH.coordinates, gen_configs[:, noH_inds, :]))

        # Obtain energies, forces and decompositions of decoded configurations
        for j, config in enumerate(gen_configs):
        
            this_energy, this_forces = analysis_tools.config_energy(config,
                                                                    this_sim,
                                                                    compute_forces=True,
                                                                    constrain_H_bonds=True,
                                                                    )
            decoded_energies.append(this_energy)
            this_force_mags = np.linalg.norm(this_forces, axis=-1)
            this_cg_diffs = gen_cg_configs[j, -len(full_decode.sequence):, :] - cg_config[j, -len(full_decode.sequence):, :]

            # As loop, will also add coordination
            # That way, make sure that they align with each other
            # And can match up to reference structure max forces because know n_samples
            for i, res in enumerate(full_decode.sequence):
                atom_inds = [a.idx for a in pmd_struc.residues[i].atoms]
                this_max_f = np.max(this_force_mags[atom_inds])
                decoded_max_forces_by_res[res].append(this_max_f)
                atom_mask = np.zeros(config.shape[0], dtype=bool)
                atom_mask[atom_inds] = True
                coordination_by_res[res].append(this_res_coordination[i])
                cg_diffs_by_res[res].append(this_cg_diffs[i, :])
            
            # Add energy decomposition as well
            pmd_struc.coordinates = config
            this_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, this_sim.system, nrg=mm.unit.kilojoules_per_mole)
            for key, eng in this_decomp:
                if key in decoded_energy_decomps:
                    decoded_energy_decomps[key].append(eng)
                else:
                    decoded_energy_decomps[key] = [eng]
    
    # Clean up output and save
    energies = np.array(energies)
    decoded_energies = np.array(decoded_energies)
    decoded_probs = np.concatenate(decoded_probs)
    
    out_energy_decomps = {}
    for key, eng in energy_decomps.items():
        out_energy_decomps['ref_'+key] = np.array(eng)
    out_energy_decomps.pop('ref_CMMotionRemover', None)
    
    out_decoded_energy_decomps = {}
    for key, eng in decoded_energy_decomps.items():
        out_decoded_energy_decomps['decoded_'+key] = np.array(eng)
    out_decoded_energy_decomps.pop('decoded_CMMotionRemover', None)

    out_max_forces_by_res = {}
    for key, val in max_forces_by_res.items():
        out_max_forces_by_res['ref_'+key] = np.array(val)

    out_decoded_max_forces_by_res = {}
    for key, val in decoded_max_forces_by_res.items():
        out_decoded_max_forces_by_res['decoded_'+key] = np.array(val)

    out_coordination_by_res = {}
    for key, val in coordination_by_res.items():
        out_coordination_by_res['coordination_'+key] = np.array(val)

    out_cg_diffs_by_res = {}
    for key, val in cg_diffs_by_res.items():
        out_cg_diffs_by_res['cg_diffs_'+key] = np.array(val)

    # Note that cannot easily compare residue coordination (burial) to reference max forces
    # Can only compare for max forces on decoded residues since those match up
    np.savez('%s.npz'%out_name,
             n_samples=n_samples,
             pdb_ids=np.array(pdb_ids),
             n_res=np.array(num_res),
             n_atoms=np.array(num_atoms),
             ref_energies=energies,
             decoded_energies=decoded_energies,
             decoded_probs=decoded_probs,
             bond_scores=np.array(bond_score_stats),
             clash_scores=np.array(clash_score_stats),
             clash_scores_res=np.array(clash_score_res_stats),
             diversity_scores=np.array(diversity_score_stats),
             bond_scores_noH=np.array(bond_score_noH_stats),
             clash_scores_noH=np.array(clash_score_noH_stats),
             clash_scores_res_noH=np.array(clash_score_res_noH_stats),
             diversity_scores_noH=np.array(diversity_score_noH_stats),
             **out_energy_decomps,
             **out_decoded_energy_decomps,
             **out_max_forces_by_res,
             **out_decoded_max_forces_by_res,
             **out_coordination_by_res,
             **out_cg_diffs_by_res,
            )


def decode_CG_traj(aa_pdb_file, cg_pdb_file, cg_traj_file, bat_dir='./', model_dir='./', n_samples=1):
    """
    Decodes n_samples configurations per frame of a CG trajectory.
    As input, an all-atom pdb file, a CG pdb file, and a CG trajectory are required.
    To correctly load the full decoding model, directories for BAT objects and trained
    models are also required.
    """

    # First, create a full protein decoding model
    full_decode = ProteinDecoder(aa_pdb_file,
                                 bat_dir=bat_dir,
                                 model_dir=model_dir,
                                 h_bonds=True,
                                 include_cg=True, # Need if constraining bonds involving hydrogens
                                )

    # Load in the CG trajectory
    cg_uni = mda.Universe(cg_pdb_file, cg_traj_file)

    # Loop over frames to get all CG coordinates
    cg_traj = []
    for frame in cg_uni.trajectory:
        cg_traj.append(frame.positions.copy())
    cg_traj = np.array(cg_traj)

    # Generate a decoded trajectory with n_samples per frame
    cg_traj = np.tile(cg_traj, (n_samples, 1, 1))
    decoded_traj = []
    decoded_probs = []
    n_chunk = 200
    for i in range(0, cg_traj.shape[0], n_chunk):
        decoded_configs, this_probs = full_decode.decode_config(cg_traj[i:(i+n_chunk)])
        decoded_traj.append(decoded_configs.numpy())
        decoded_probs.append(this_probs.numpy())

    decoded_traj = np.concatenate(decoded_traj, axis=0)
    decoded_probs = np.concatenate(decoded_probs, axis=0)

    # Also obtain energy for every decoded configuration
    # Including energy decomposition
    aa_pdb_obj, sim = analysis_tools.sim_from_pdb(aa_pdb_file)
    pmd_struc = pmd.openmm.load_topology(aa_pdb_obj.topology, system=sim.system, xyz=aa_pdb_obj.positions)
    decoded_energies = []
    decoded_decomp = {}
    for config in decoded_traj:
        this_energy = analysis_tools.config_energy(config,
                                                   sim,
                                                   constrain_H_bonds=True,
                                                  )
        decoded_energies.append(this_energy)
        # Add energy decomposition as well
        pmd_struc.coordinates = config
        this_decomp = pmd.openmm.energy_decomposition_system(pmd_struc, sim.system, nrg=mm.unit.kilojoules_per_mole)
        for key, eng in this_decomp:
            if key in decoded_decomp:
                decoded_decomp[key].append(eng)
            else:
                decoded_decomp[key] = [eng]

    decoded_energies = np.array(decoded_energies)
    for key, eng in decoded_decomp.items():
        decoded_decomp[key] = np.array(eng)
    decoded_decomp.pop('CMMotionRemover', None)

    # Save the decoded trajectory, probabilities, and energies
    # For the probabilities, also indicate which CG frame the decoded config corresponds to
    aa_uni = mda.Universe(aa_pdb_file, decoded_traj)
    out_traj_name = 'decoded_' + cg_traj_file.split('/')[-1].replace('cg', 'aa').replace('pdb', 'nc')
    aa_uni.select_atoms('all').write(out_traj_name, frames='all')
    np.savez('decoded_logprobs_energies_%s.npz'%aa_pdb_file.split('.pdb')[0].split('/')[-1],
             logprobs=decoded_probs,
             energies=decoded_energies,
             cginds=np.tile(np.arange(cg_uni.trajectory.n_frames), (n_samples,)),
             **decoded_decomp,
            )


def run_traj_analysis(arg_list):
    parser = argparse.ArgumentParser(prog='full_protein_decoding.run_traj_analysis',
                                     description='Performs analysis using trained residue models to decode full proteins',
                                    )
    parser.add_argument('pdb_file', help='path to pdb file of full-atom reference configuration')
    parser.add_argument('traj_file', help='path to trajectory file')
    parser.add_argument('--bat_dir', '-b', help='path to directory containing directories named by residue type with pickled BAT objects')
    parser.add_argument('--model_dir', '-m', help='path to directory containing trained model directories')
    parser.add_argument('--output', '-o', help='path of output file (WILL OVERWRITE)')
    parser.add_argument('--n_samples', '-n', type=int, default=1, help='number of decoded samples per frame')

    args = parser.parse_args(arg_list)

    analyze_trajectory(args.pdb_file,
                       args.traj_file,
                       bat_dir=args.bat_dir,
                       model_dir=args.model_dir,
                       out_name=args.output,
                       n_samples=args.n_samples,
                      )


def run_individual_residue_analysis(arg_list):
    parser = argparse.ArgumentParser(prog='full_protein_decoding.run_individual_residue_analysis',
                                     description='Performs analysis using trained residue models to decode one residue at a time',
                                    )
    parser.add_argument('pdb_file', help='path to pdb file of full-atom reference configuration')
    parser.add_argument('--bat_dir', '-b', help='path to directory containing directories named by residue type with pickled BAT objects')
    parser.add_argument('--model_dir', '-m', help='path to directory containing trained model directories')
    parser.add_argument('--output', '-o', help='path of output file (WILL OVERWRITE)')
    parser.add_argument('--n_samples', '-n', type=int, default=1, help='number of decoded samples for each residue')

    args = parser.parse_args(arg_list)

    analyze_individual_residues(args.pdb_file,
                                bat_dir=args.bat_dir,
                                model_dir=args.model_dir,
                                out_name=args.output,
                                n_samples=args.n_samples,
                               )


def run_dataset_analysis(arg_list):
    parser = argparse.ArgumentParser(prog='full_protein_decoding.run_dataset_analysis',
                                     description='Performs analysis over dataset of proteins',
                                    )
    parser.add_argument('pdb_dir', help='path to directory containing pdb files')
    parser.add_argument('--bat_dir', '-b', help='path to directory containing directories named by residue type with pickled BAT objects')
    parser.add_argument('--model_dir', '-m', help='path to directory containing trained model directories')
    parser.add_argument('--output', '-o', default='dataset_decoding_analysis', help='path of output file (WILL OVERWRITE)')
    parser.add_argument('--n_samples', '-n', type=int, default=100, help='number of decoded samples per frame')

    args = parser.parse_args(arg_list)

    pdb_files = glob.glob('%s/*.pdb'%args.pdb_dir)


    analyze_protein_dataset(pdb_files,
                            bat_dir=args.bat_dir,
                            model_dir=args.model_dir,
                            out_name=args.output,
                            n_samples=args.n_samples,
                           )


def run_cg_traj_decoding(arg_list):
    parser = argparse.ArgumentParser(prog='full_protein_decoding.run_cg_traj_decoding',
                                     description='Produces an all-atom trajectory and log-probabilities from a CG trajectory',
                                    )
    parser.add_argument('aa_pdb_file', help='path to pdb file of full-atom reference configuration')
    parser.add_argument('cg_pdb_file', help='path to pdb file of a CG configuration')
    parser.add_argument('cg_traj_file', help='path to CG trajectory file')
    parser.add_argument('--bat_dir', '-b', help='path to directory containing directories named by residue type with pickled BAT objects')
    parser.add_argument('--model_dir', '-m', help='path to directory containing trained model directories')
    parser.add_argument('--n_samples', '-n', type=int, default=1, help='number of decoded samples per frame')

    args = parser.parse_args(arg_list)

    decode_CG_traj(args.aa_pdb_file,
                   args.cg_pdb_file,
                   args.cg_traj_file,
                   bat_dir=args.bat_dir,
                   model_dir=args.model_dir,
                   n_samples=args.n_samples,
                  )


if __name__ == '__main__':
    if sys.argv[1] == 'trajectory':
        run_traj_analysis(sys.argv[2:])
    elif sys.argv[1] == 'dataset':
        run_dataset_analysis(sys.argv[2:])
    elif sys.argv[1] == 'decode':
        run_cg_traj_decoding(sys.argv[2:])
    elif sys.argv[1] == 'residue':
        run_individual_residue_analysis(sys.argv[2:])
    else:
        print("Argument \'%s\' unrecognized. For the first argument, select \'trajectory\', \'dataset\', \'decode\', or \'residue\'."%sys.argv[1])
