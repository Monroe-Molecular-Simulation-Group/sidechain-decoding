"""
Applies residue-based decodings to an entire protein coarse-grained model.
"""

import sys, os
import pickle

import numpy as np
import tensorflow as tf

import parmed as pmd
from openmm import app as mmapp

import vaemolsim

from . import data_io, coord_transforms, model_training, unconditional 


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
                  unconditional_types=['NPRO', 'Nterm'], # ['GLY', 'NPRO', 'Nterm'],
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
            loss = model_training.LogProbPenalizedCGLoss(bat_dict[res], mask_H=constrain_H_bonds)
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


class ProteinDecoder(object):
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
                exclude_ind = [cg_struc.view[':%i@%s'%(i + 1, a)].atoms[0].idx for a in this_decode_names if a == 'H']
                remove_Nterm_H_inds.append(exclude_ind[0])
                
        self.remove_Nterm_H_inds = remove_Nterm_H_inds
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
                        this_decode_names = [a.name for a in self.bat_dict['Nterm']._ag if a.name not in this_root_names]
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
            # if res_name == 'GLY':
            #     uncond_seq.append(res_name)
            #     uncond_root_inds.append(this_root_inds)
            #     uncond_decode_inds.append(this_decode_inds)
            #     uncond_one_hot.append(this_one_hot)
            # else:
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

    def decode_config(self, cg_config, n_samples=1):
        """
        Given a CG configuration of shape (N_particles, 3), decodes to atomistic.
        Should consist of all atomistic "CG" coordinates (i.e., the backbone-ish) plus the
        sidechain beads, in that order so that it matches up with self.one_hot_cg.
        """

        # Prep configuration by removing N-terminal H hydrogens
        cg_config = np.delete(cg_config, self.remove_Nterm_H_inds, axis=0)

        # Check shape of configuration
        if cg_config.shape != (self.one_hot_cg.shape[0], 3):
            raise ValueError('Shape of input configuration (%s)'
                             'must match structures used for initialization (%s).'%(str(cg_config.shape),
                                                                                    str((self.one_hot_cg.shape[0], 3))))

        # Parallelize sample generation by making copies of the CG configuration.
        cg_config = tf.tile(tf.expand_dims(tf.cast(cg_config, tf.float32), axis=0), (n_samples, 1, 1))
        cg_one_hot = tf.tile(tf.expand_dims(tf.cast(self.one_hot_cg, tf.float32), axis=0), (n_samples, 1, 1))

        # Will loop over the residues and decode each
        # Can keep track of decoded positions in a list to concatenate with the CG config at each pass
        # The decoding models do not care about atom order, so can stitch together correctly at end
        # Same with one-hot encodings
        # Also make sure to keep track of probability for each generated sample
        decoded_coords = []
        decoded_one_hot = []
        decoded_probs = []

        # Decode with unconditional models first
        for i, res in enumerate(self.uncond_seq):
            bat = self.bat_dict[res]
            bat_input = tf.tile(tf.cast(bat.results.bat[:, 9:], tf.float32), (n_samples, 1))
            bat_input = bat_input[:, self.h_info_dict[res][1]]

            # If want probabilities, cannot use predict_on_batch...
            # Need distribution, then compute log_probabilities of sample
            dist = self.model_dict[res](bat_input)
            sample = dist.sample()
            prob = dist.log_prob(sample)
            decoded_probs.append(prob)

            # Fill in hydrogens
            full_bat_sample = coord_transforms.fill_in_h_bonds(sample.numpy(), *self.h_info_dict[res])

            # Convert BAT coordinates to xyz
            full_bat_sample = coord_transforms.fill_in_bat(full_bat_sample,
                                                           cg_config.numpy()[:, self.uncond_root_inds[i], :])
            sample_xyz = coord_transforms.xyz_from_bat(full_bat_sample, bat)
            # But want to exclude the root atom indices, which are already in the CG config
            sample_xyz = np.delete(sample_xyz, self.bat_dict[res]._root_XYZ_inds, axis=1)
            decoded_coords.append(tf.convert_to_tensor(sample_xyz, dtype=tf.float32))
            
            # Get one-hot encoding for this set of decoded atoms
            one_hot = tf.tile(tf.expand_dims(tf.cast(self.uncond_one_hot[i], tf.float32), axis=0), (n_samples, 1, 1))
            decoded_one_hot.append(one_hot)

        # Decode conditional models
        for i, res in enumerate(self.cond_seq):
            
            # Add coordinates and one-hot encodings decoded up to this point to CG configuration
            if len(decoded_coords) != 0:
                this_config = tf.concat([cg_config,] + decoded_coords, axis=1)
                this_one_hot = tf.concat([cg_one_hot,] + decoded_one_hot, axis=1)

            # Grab reference sidechain location to decode
            this_ref = cg_config[:, self.cond_cg_ref_inds[i], :]

            # Collect input
            this_input = (this_ref, this_config, this_one_hot)
            
            # Apply decoding model
            dist = self.model_dict[res](this_input)
            sample = dist.sample()
            prob = dist.log_prob(sample)
            decoded_probs.append(prob)

            # Fill in hydrogens
            full_bat_sample = coord_transforms.fill_in_h_bonds(sample.numpy(), *self.h_info_dict[res])

            # Convert BAT coordinates to xyz
            full_bat_sample = coord_transforms.fill_in_bat(full_bat_sample,
                                                           cg_config.numpy()[:, self.cond_root_inds[i], :])
            sample_xyz = coord_transforms.xyz_from_bat(full_bat_sample, self.bat_dict[res])
            # But want to exclude the root atom indices, which are already in the CG config
            sample_xyz = np.delete(sample_xyz, self.bat_dict[res]._root_XYZ_inds, axis=1)
            decoded_coords.append(tf.convert_to_tensor(sample_xyz, dtype=tf.float32))

            # Get one-hot encoding for this set of decoded atoms
            one_hot = tf.tile(tf.expand_dims(tf.cast(self.cond_one_hot[i], tf.float32), axis=0), (n_samples, 1, 1))
            decoded_one_hot.append(one_hot)

        # Finally, need to stitch the configuration back together
        decode_inds = np.hstack([self.cg_inds,] + self.uncond_decode_inds + self.cond_decode_inds)
        sort_inds = np.argsort(decode_inds)
        decoded_configs = tf.concat([cg_config[:, :-len(self.sequence), :]] + decoded_coords, axis=1).numpy()[:, sort_inds, :]

        # Sum over log probabilities (for each decoding model)
        decoded_probs = np.array(decoded_probs)
        decoded_probs = np.sum(decoded_probs, axis=0)

        return decoded_configs, decoded_probs


