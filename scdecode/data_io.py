"""
Defines routines for generating training inputs and reading that data for training.
"""

import os
import json
import pickle
import argparse
import glob

import numpy as np

import parmed as pmd

from openmm import app as mmapp

import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT

import tensorflow as tf


# Define some global variables, like the forcefield used
ff = mmapp.ForceField('amber14/protein.ff14SB.xml')
ref_res_types = [key for key in ff._templates]
ref_atom_types = [key for key in ff._atomTypes]
all_ref_types = ref_atom_types + ref_res_types
# bat_exclude_dofs = [0, 1, 2, 3, 4, 6]
# XYZ of root atom, angles for first bond in spherical coords, first bond length
# Will be same for ANY residue except GLY


def inputs_from_pdb(pdb_file, res_name, mod_info,
                    bb_atom_str='@N,CA,C,O,H,HA',
                    not_bat_atom_str='@N,O,H',
                    cg_atom_list=['N', 'CA', 'C', 'O', 'H', 'CB'],
                    rng=np.random.default_rng(),
                    ):
    """
    Generates inputs needed for training sidechain backmapping models from pdb files.

    Parameters
    ----------
    pdb_file : str
        Full path file name for the PDB file to process
    res_name : str
        The name of the residue that will be used for training. All such residues in the
        protein are identified with input/target data for training created for each.
    mod_info : str
        A list of residue indices for which mutations and/or modifications, such as
        adding heavy atoms, have been made. These will not be considered for preparing
        training inputs, even if they match res_name.
    bb_atom_str : str, default '@N,CA,C,O,H,HA'
        A string following the ParmEd atom selection format specifying backbone atoms.
    not_bat_atom_str : str, default '@C,O,H'
        A string following ParmEd atom selection format specifying which atoms in a residue
        will NOT be included when generating BAT coordinates.
    cg_atom_list : list, default ['N', 'CA', 'C', 'O', 'H', 'CB']
        A list of atom names that will be included in the "CG" representation of a protein.
    rng : object, default np.random.default_rng()
        Random number generator

    Returns
    -------
    cg_inputs : list of NumPy arrays
        A list of the CG reference beads for each residue to be included in the training set.
        The number of arrays in the list will match the number of residues matching res_name
        and not excluded by mod_info. Each array will be of shape (3,), containing the x, y, z
        coordinates of the CG bead to decode.
    coord_inputs : list of NumPy arrays
        A list of protein coordinates to inform the decoding, including both atomistic and CG.
        Will NOT include the coordinates of the atoms that the model will be trained to decode.
        The number of arrays will be the same as in cg_inputs, but the shape of each array
        will be (N_particles, 3), where N_particles may vary.
    one_hot_inputs : list of NumPy arrays
        Matching one-hot atom/CG bead type information to complement coord_inputs. Generally,
        this will be (N_particles, N_types), where each N_particles entry should match that
        in coord_inputs.
    full_bat : NumPy array
        Array of the full BAT representation for the residues to train on. This includes the
        location of the backbone nitrogen (N), alpha carbon (CA) and beta carbon (CB), which
        are not part of what is decoded. However, these locations are necessary to convert
        back from BAT to XYZ coordinates.
    bat_targets : list of NumPy arrays
        Coordinates, in a BAT reference frame, of the atoms to be decoded. The number of
        arrays in the list will match that in cg_inputs, with the shape of each array
        being (N_bat,), where N_bat will be the same for a specific residue type.
    bat_analysis : MDAnalysis BAT analysis object
        The object used to convert between BAT and XYZ coordinates for the residue type
        that will be trained on.
    """
    # Get pdb id from file
    pdb_id = os.path.split(pdb_file)[-1].split('.pdb')[0]

    # Load structure with openmm first
    mm_pdb = mmapp.PDBFile(pdb_file)

    # Load into ParmEd
    struc = pmd.openmm.load_topology(mm_pdb.topology, xyz=mm_pdb.positions)

    # Separate out sidechain atoms and compute CG bead locations
    sc_struc = struc['!(%s)'%bb_atom_str]
    cg_coords = []
    for i, r in enumerate(sc_struc.residues):
        masses = np.array([a.mass for a in r])
        cg_coords.append(pmd.geometry.center_of_mass(sc_struc[':%i'%(i+1)].coordinates, masses))
    cg_coords = np.array(cg_coords)

    # Follow procedure in openmm.app.ForceField.createSystem() to correctly apply
    # forcefield information to this pdb
    ff_data = ff._SystemData(mm_pdb.topology)
    templates_for_residues = ff._matchAllResiduesToTemplates(ff_data, mm_pdb.topology, dict(), False)
    atom_types = [ff_data.atomType[a] for a in mm_pdb.topology.atoms()]
    res_types = [templates_for_residues[r.index].name for r in mm_pdb.topology.residues()]

    # Identify target residues (of right type, not modified)
    target_res_num = []
    for i, r in enumerate(res_types):
        if (r == res_name) and (i not in mod_info):
            target_res_num.append(str(i + 1))

    # If have no residues of desired type, stop now
    if len(target_res_num) == 0:
        return None

    # Use .view to preserve residue and atom indexing
    target_res = struc.view[':'+','.join(target_res_num)]

    # Will need list of atom indices of sidechain atoms in every residue
    # Makes easier to randomly select residues for exclusion to emulate random backmapping
    res_atom_inds = []
    for res in struc.residues:
        res_atom_inds.append([a.idx for a in res.atoms if a.name not in cg_atom_list])

    # Create one-hot encodings for atoms and residues
    # But using overall set of combined atom and residue types
    one_hot_atoms = np.zeros((len(atom_types), len(all_ref_types)))
    for i, t in enumerate(atom_types):
        if t not in all_ref_types:
            raise ValueError("The atom type %s does not match any type in the force field."%t)
        one_hot_atoms[i, :] = np.array(all_ref_types == t, dtype='int32')

    one_hot_res = np.zeros((len(res_types), len(all_ref_types)))
    for i, t in enumerate(res_types):
        if t not in all_ref_types:
            raise ValueError("The residue type %s does not match any template in the force field."%t)
        one_hot_res[i, :] = np.array(all_ref_types == t, dtype='int32')

    # Now have all information for entire protein
    # Will loop over residues matching target type and select out for each
    cg_inputs = []
    coord_inputs = []
    one_hot_inputs = []
    full_bat = []
    bat_targets = []

    # Also return BAT analysis object from last target residue
    # Should behave the same way for all residues of same type
    bat_analysis = None

    for res in target_res.residues:
    
        # When selecting atoms, only exclude non-CG representation atoms
        # To be consitent with Rosetta, the CG atoms are the backbone atoms plus the beta carbon (and a CG bead)
        # So only want to remove the FG coordinates of other atoms in the residue
        this_atom_inds = [a.idx for a in res.atoms if a.name not in cg_atom_list]

        # To improve ability to decode structures in any order (and save memory)
        # Randomly select a number of residues, then specific residues, to exclude atoms for
        num_res_exclude = rng.integers(0, len(res_types) + 1)
        res_exclude = rng.choice(len(res_types), size=num_res_exclude, replace=False)
        for i in res_exclude:
            if i != res.idx:
                this_atom_inds.extend(res_atom_inds[i])

        # And gather Cartesian coordinate and type information we want
        this_fg_coords = np.delete(struc.coordinates, this_atom_inds, axis=0)
        this_cg_coords = np.delete(cg_coords, res.idx, axis=0)
        this_fg_one_hot = np.delete(one_hot_atoms, this_atom_inds, axis=0)
        this_cg_one_hot = np.delete(one_hot_res, res.idx, axis=0)
        cg_inputs.append(cg_coords[res.idx, :])
        coord_inputs.append(np.vstack([this_fg_coords, this_cg_coords]))
        one_hot_inputs.append(np.vstack([this_fg_one_hot, this_cg_one_hot]))

        # Also need BAT coordinates for just this residue
        this_bat_atoms = struc['(:%i)&(!(%s))'%(res.idx + 1, not_bat_atom_str)]
        uni = mda.Universe(this_bat_atoms.topology, np.array(this_bat_atoms.coordinates, dtype='float32'))
        bat_analysis = BAT(uni.select_atoms('all'), initial_atom=uni.select_atoms('name C')[0])
        bat_analysis.run()
        this_bat = bat_analysis.results.bat[0] # Includes N, CA, and CB atom information
        full_bat.append(this_bat)
        # bat_targets.append(np.delete(this_bat, bat_exclude_dofs))
        bat_targets.append(this_bat[9:])

    # Clean up and return
    # Only convert full_bat to array - rest better off in list
    # That way easier to combine inputs from multiple pdbs as ragged tensors if want to
    full_bat = np.array(full_bat, dtype='float32')

    return cg_inputs, coord_inputs, one_hot_inputs, full_bat, bat_targets, bat_analysis


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


# Need to fix this function, carefully thinking about how should work
# Want to instead use N, CA, and CB positions as root atoms for BAT analysis
# Trickier with N
# But also, need to think about when need to fill in full BAT coords obtain XYZ coords
# What is use case?
def fill_in_bat(partial_bat, ca_pos, cb_pos):
    """
    Recreates a full set of BAT coordinates from a partial set and CA and CB atom positions.

    Parameters
    ----------
    partial_bat : NumPy array
        The partial set of BAT coordinates, not including the CA or CB atom positions.
    ca_pos : NumPy array
        The XYZ coordinate of the alpha carbon, which is the root of the BAT analysis.
    cb_pos : NumPy array
        The XYZ coordinate of the beta carbon, which is the secondary atom in the BAT
        analysis.

    Returns
    -------
    full_bat : NumPy array
        The full set of BAT coordinates, including information on the CA and CB atom
        locations, which is needed for converting back to XYZ coordinates for all
        atoms in a sidechain.
    """
    diff = cb_pos - ca_pos
    r = np.sqrt(np.sum(diff*diff, axis=-1))
    polar = np.arccos(diff[..., 2] / r)
    azimuthal = np.arctan2(diff[..., 1], diff[..., 0])
    full_bat = np.zeros((partial_bat.shape[0], 6+partial_bat.shape[1]), dtype='float32')
    full_bat = np.hstack([ca_pos, azimuthal, polar, partial_bat[..., 0], r, partial_bat[..., 1:]])
    return full_bat


def _create_example(refs, coords, infos, targets):
    """
    A function to create a serialized tf.train.Example object describing training data.

    Parameters
    ----------
    refs : tf.Tensor
        A tensor describing the reference CG positions to be decoded.
    coords : tf.Tensor
        All coordinate information to be passed as inputs (excluding the atoms to be decoded).
    infos : tf.Tensor
        Information, like one-hot encoding or parameters, associated with coords.
    targets : tf.Tensor
        The BAT coordinates to be predicted by the model (targets for training).

    Returns
    -------
    serialized string of tf.train.Example object
    """
    input_list = [tf.io.serialize_tensor(t).numpy() for t in [refs, coords, infos]]
    input_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=input_list))
    target_serialized = tf.io.serialize_tensor(targets)
    target_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_serialized.numpy()]))
    features = tf.train.Features(feature={'inputs': input_feature, 'targets': target_feature})
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def _tf_serialize_example(inputs, targets):
    """
    Returns a serialized string representing a tf.train.Example object of training inputs and targets.

    This function is intened to be used to map a dataset of inputs and targets to a serialized
    format that can be saved as a TFRecord.

    Parameters
    ----------
    inputs : list of tf.Tensor objects
        A list of tf.Tensor objects containing the reference CG positions to be decoded, all XYZ
        coordinate information provided for decoding, and particle type information, in that order.
    targets : tf.Tensor
        A tensor containing all training targets (the BAT coordinates to predict).

    Returns
    -------
    example_str : tf.Tensor with dtype str
        Serialized string wrapped in a tensor representing a training example.
    """
    example_str = tf.py_function(_create_example, (inputs[0], inputs[1], inputs[2], targets), tf.string)
    return tf.reshape(example_str, ())


def _create_data_from_serialized_strs(input_str, targets_str):
    """
    Parses a single serialized example, recovering tensors from serialized strings.

    Parameters
    ----------
    input_str : tf.Tensor of dtype str
        The serialized string representing the inputs for training (reference CG
        coordinates, all other coordinates, and particle info).
    targets_str : tf.Tensor of dtype str
        The serialized string representing the targets for training (BAT coords).

    Returns
    -------
    refs_read : tf.Tensor
        Reference CG particle coordinates to decode
    coords_read : tf.Tensor
        Coordinates to be used as inputs the decoding (excludes coordinates
        that are being predicted).
    info_read : tf.Tensor
        One-hot or other information concerning particle types for the supplied
        coordinates.
    targets_read : tf.Tensor
        Targets parsed from the serialized string (BAT coordinates to predict).
    """
    np_inputs = input_str.numpy()
    refs_read = tf.io.parse_tensor(np_inputs[0], out_type='float32')
    coords_read = tf.io.parse_tensor(np_inputs[1], out_type='float32')
    info_read = tf.io.parse_tensor(np_inputs[2], out_type='float32')
    targets_read = tf.io.parse_tensor(targets_str[0].numpy(), out_type='float32')
    return refs_read, coords_read, info_read, targets_read


def _tf_parse_example(example_str):
    """
    Takes a single serialized example string and parses the data from it.

    This is intended to be applied with tf.data.Dataset.map to convert all
    serialized string data into training inputs and targets.

    Parameters
    ----------
    example_str : serialized str
        A serialized string representing a tf.train.Example object.

    Returns
    -------
    tuple of tf.Tensors
        The inputs to the decoding model, as a tuple.
    tf.Tensor
        The targets for the decoding model.
    """
    example_read = tf.io.parse_single_example(
                    example_str,
                    features = {
                                'inputs': tf.io.RaggedFeature(dtype=tf.string),
                                'targets': tf.io.RaggedFeature(dtype=tf.string)}
                    )
    data = tf.py_function(_create_data_from_serialized_strs,
                          (example_read['inputs'], example_read['targets']),
                          (tf.float32, tf.float32, tf.float32, tf.float32),
                         )
    return (data[0], data[1], data[2]), data[3]


def _data_as_tensors(refs, coords, info, targets):
    """
    Takes lists of NumPy arrays and converts them into tensors.

    Parameters
    ----------
    refs : list of NumPy arrays
        The CG reference coordinates to decode.
    coords : list of NumPy arrays
        Atomic and CG coordinates for the decoder to use.
    info : list of NumPy arrays
        A list of one-hot particle types corresponding to coords.
    targets : list of NumPy arrays
        A list of targets (BAT coordinates) for the model to predict.

    Returns
    -------
    refs : tf.Tensor
        Converted refs
    coords : tf.RaggedTensor
        Converted coords, noting that may be ragged with different
        numbers of particles in each entry.
    info : tf.RaggedTensor
        Converted info, matching numbers of particles in coords.
    targets : tf.Tensor
        Converted targets
    """
    refs = tf.convert_to_tensor(np.array(refs, dtype='float32'))
    targets = tf.convert_to_tensor(np.array(targets, dtype='float32'))

    coords = [np.array(c, dtype='float32') for c in coords]
    info = [np.array(i, dtype='float32') for i in info]
    lens = [c.shape[0] for c in coords]

    coords = np.vstack(coords)
    info = np.vstack(info)

    coords = tf.RaggedTensor.from_row_splits(coords, np.hstack([0, np.cumsum(lens)]))
    info = tf.RaggedTensor.from_row_splits(info, np.hstack([0, np.cumsum(lens)]))

    return refs, coords, info, targets


def save_dataset(refs, coords, info, targets, save_file_name):
    """
    Saves a TFRecord-based dataset from tensors of inputs and targets.

    Parameters
    ----------
    refs : tf.Tensor
        The CG reference coordinates to decode.
    coords : tf.RaggedTensor
        Atomic and CG coordinates for the decoder to use.
    info : tf.RaggedTensor
        One-hot particle types corresponding to coords.
    targets : tf.Tensors
        Targets (BAT coordinates) for the model to predict.
    save_file_name : str
        The name of the file to save the dataset to.

    Returns
    -------
    Writes file save_file_name

    """
    inputs_dset = tf.data.Dataset.from_tensor_slices((refs, coords, info))
    targets_dset = tf.data.Dataset.from_tensor_slices(targets)
    full_dset = tf.data.Dataset.zip((inputs_dset, targets_dset))
    serialized_full_dset = full_dset.map(_tf_serialize_example)
    writer = tf.io.TFRecordWriter(save_file_name, options='GZIP')
    for seri_example in serialized_full_dset:
        writer.write(seri_example.numpy())
    writer.close()


def read_dataset(files):
    """
    Reads a TFRecord-based dataset, producing a parsed dataset.

    Parsing is necessary to deserialize the serialized strings representing tf.train.Example objects.

    Parameters
    ----------
    files : list of str
        A list of file names to load data from.

    Returns
    -------
    parsed_dset : tf.data.Dataset
        A tf dataset of parsed examples.
    """
    raw_dset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    parsed_dset = raw_dset.map(_tf_parse_example)
    return parsed_dset


def main():
    parser = argparse.ArgumentParser(prog='create_inputs.py',
                                     description='Generates training inputs for all pdb files in a directory',
                                    )
    parser.add_argument('mod_file', help="json file containing dictionary of residue modifications for each pdb")
    parser.add_argument('res_type', help="residue type to prepare inputs for")
    parser.add_argument('--read_dir', '-r', default='./', help="directory to read files from")
    parser.add_argument('--save_dir', '-s', default='./', help="directory to save outputs to")

    args = parser.parse_args()

    # Get dictionary of modifications
    with open(args.mod_file, 'r') as f:
        mod_info = json.load(f)

    # Get a list of all pdb files to process
    pdb_list = glob.glob(os.path.join(args.read_dir, '*.pdb'))
    print('Found %i pdb files to process.'%len(pdb_list))

    # Get a fixed seed for this run and report it
    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed=seed)
    print('\nSeed entropy: %i'%seed.entropy)
    print('For reproducible results, create a seed with np.random.SeedSequence(entropy=%i)'%seed.entropy)
    print('Then pass that seed to np.random.default_rng(seed=seed), replacing the appropriate lines of code.\n')

    saved_bat_obj = False
    for p in pdb_list:
        pdb_id = os.path.split(p)[-1].split('.pdb')[0]
        this_mod = mod_info[pdb_id]

        inputs = inputs_from_pdb(p, args.res_type, this_mod, rng=rng)

        # Check to make sure had residues to work with
        if inputs is not None:

            # Want to save as something that tensorflow can read easily and quickly without memory issues
            # Best way is to save as serialized tf.train.Example objects in a tf.data.Dataset in TFRecord file
            refs, coords, info, bat_targets = _data_as_tensors(inputs[0], inputs[1], inputs[2], inputs[4])
            save_dataset(refs, coords, info, bat_targets,
                         os.path.join(args.save_dir, '%s_%s.tfrecord'%(pdb_id, args.res_type))
                        )

            # np.savez_compressed(os.path.join(args.save_dir, '%s_%s_inputs.npz'%(pdb_id, args.res_type)),
            #                     cg=inputs[0],
            #                     coords=inputs[1],
            #                     one_hot=inputs[2],
            #                     full_bat=inputs[3],
            #                     bat=inputs[4],
            #                    )

            # Specifically save the full set of BAT coordinates
            np.save(os.path.join(args.save_dir, '%s_%s_full_BAT.npy'%(pdb_id, args.res_type)),
                    inputs[3],
                    allow_pickle=False
                   )

            # Pickle the most recent BAT object
            # Only need to do once per residue type
            # So do as soon as inputs not None
            if not saved_bat_obj:
                pickle.dump(inputs[-1], open(os.path.join(args.save_dir, '%s_BAT_object.pkl'%(args.res_type)), 'wb'))
                saved_bat_obj = True


if __name__ == '__main__':
    main()
